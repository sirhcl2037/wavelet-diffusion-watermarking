import io
import time
import torch
import numpy as np
import pywt
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, roc_auc_score
import os

# Setup device and pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16 if device.type == 'cuda' else torch.float32

# Load Stable Diffusion pipeline (add this if not already loaded)
pipe = StableDiffusionPipeline.from_pretrained(
'runwayml/stable-diffusion-v1-5',
torch_dtype=dtype
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Create debug directory for visual verification
os.makedirs("attack_debug", exist_ok=True)

# Core watermarking functions
def embed_watermark_wavelet(noise, watermark_array, strength=0.02, wavelet='haar'):
noise_np = noise.detach().cpu().numpy()
coeffs = pywt.dwt2(noise_np, wavelet)
cA, (cH, cV, cD) = coeffs

len_a = cA.size
len_h = cH.size
len_v = cV.size
len_d = cD.size
total_required = len_a + len_h + len_v + len_d

if len(watermark_array) < total_required:
raise ValueError(f"Watermark too short: requires {total_required} values")

w_a = watermark_array[:len_a]
w_h = watermark_array[len_a:len_a+len_h]
w_v = watermark_array[len_a+len_h:len_a+len_h+len_v]
w_d = watermark_array[len_a+len_h+len_v:len_a+len_h+len_v+len_d]

cA.flat[:] += strength * w_a
cH.flat[:] += strength * w_h
cV.flat[:] += strength * w_v
cD.flat[:] += strength * w_d

watermarked_np = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
return torch.tensor(watermarked_np, device=noise.device, dtype=noise.dtype)

def DetectWatermark(noise, watermark_array, wavelet='haar'):
noise_np = noise.detach().cpu().numpy()
coeffs = pywt.dwt2(noise_np, wavelet)
cA, (cH, cV, cD) = coeffs

len_a = cA.size
len_h = cH.size
len_v = cV.size
len_d = cD.size
total_required = len_a + len_h + len_v + len_d

if len(watermark_array) < total_required:
raise ValueError(f"Watermark too short: needs {total_required}, got {len(watermark_array)}")

w_a = watermark_array[:len_a]
w_h = watermark_array[len_a:len_a + len_h]
w_v = watermark_array[len_a + len_h:len_a + len_h + len_v]
w_d = watermark_array[len_a + len_h + len_v:len_a + len_h + len_v + len_d]

def safe_corr(x, y):
x = (x - np.mean(x)) / (np.std(x) + 1e-10)
y = (y - np.mean(y)) / (np.std(y) + 1e-10)
return np.dot(x, y) / len(x)

corr_a = safe_corr(cA.flat, w_a)
corr_h = safe_corr(cH.flat, w_h)
corr_v = safe_corr(cV.flat, w_v)
corr_d = safe_corr(cD.flat, w_d)

return (corr_a + corr_h + corr_v + corr_d) / 4.0

def ddim_inversion_fixed(image, prompt, pipe, steps=50):
inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
inverse_scheduler.set_timesteps(steps)

if isinstance(image, Image.Image):
transform = transforms.Compose([
transforms.Resize((512, 512)),
transforms.ToTensor(),
transforms.Normalize([0.5], [0.5])
])
image_tensor = transform(image).unsqueeze(0).to(pipe.device).to(pipe.dtype)
else:
image_tensor = image.to(pipe.device).to(pipe.dtype)

with torch.no_grad():
latents = pipe.vae.encode(image_tensor).latent_dist.sample()
latents = latents * pipe.vae.config.scaling_factor

text_input = pipe.tokenizer(
[prompt],
padding="max_length",
max_length=pipe.tokenizer.model_max_length,
truncation=True,
return_tensors="pt"
)
text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]

current_latents = latents
for t in inverse_scheduler.timesteps:
latent_model_input = inverse_scheduler.scale_model_input(current_latents, t)

with torch.no_grad():
noise_pred = pipe.unet(
latent_model_input,
t,
encoder_hidden_states=text_embeddings
).sample

current_latents = inverse_scheduler.step(
noise_pred, t, current_latents
).prev_sample

return current_latents

# Enhanced Attack Functions with Visual Verification
def verify_attack_effectiveness(original_img, attacked_img, attack_type, strength):
"""Verify that the attack actually modified the image"""
orig_array = np.array(original_img)
attack_array = np.array(attacked_img)

# Calculate pixel-level differences
diff = np.abs(orig_array.astype(float) - attack_array.astype(float))
mean_diff = np.mean(diff)
max_diff = np.max(diff)
pixels_changed = np.sum(diff > 0)
total_pixels = orig_array.size
percent_changed = (pixels_changed / total_pixels) * 100

# Calculate PSNR (Peak Signal-to-Noise Ratio)
mse = np.mean((orig_array.astype(float) - attack_array.astype(float)) ** 2)
if mse == 0:
psnr = float('inf')
else:
psnr = 20 * np.log10(255.0 / np.sqrt(mse))

print(f" Attack Verification for {attack_type} (strength={strength}):")
print(f" Mean pixel difference: {mean_diff:.2f}")
print(f" Max pixel difference: {max_diff:.2f}")
print(f" Pixels changed: {percent_changed:.2f}%")
print(f" PSNR: {psnr:.2f} dB")

# Warning thresholds
if mean_diff < 1.0:
print(f" WARNING: Very small changes detected!")
if percent_changed < 10.0 and attack_type != "scaling":
print(f" WARNING: Less than 10% of pixels changed!")
if psnr > 40 and attack_type == "jpeg":
print(f" WARNING: PSNR too high for JPEG compression!")

return {
'mean_diff': mean_diff,
'max_diff': max_diff,
'percent_changed': percent_changed,
'psnr': psnr
}

def apply_attack_with_verification(image, attack_type, strength, save_debug=False, debug_id=""):
"""Apply attack with visual verification and optional debug saving"""
original = image.copy()

if attack_type == "jpeg":
buffer = io.BytesIO()
image.save(buffer, format="JPEG", quality=strength)
buffer.seek(0)
attacked = Image.open(buffer)

elif attack_type == "gaussian_noise":
img_array = np.array(image).astype(np.float32) / 255.0
noise = np.random.normal(0, strength, img_array.shape)
noisy = np.clip(img_array + noise, 0, 1)
attacked = Image.fromarray((noisy * 255).astype(np.uint8))

elif attack_type == "rotation":
attacked = image.rotate(strength, expand=False, fillcolor=(128, 128, 128))

elif attack_type == "scaling":
w, h = image.size
new_size = (int(w * strength), int(h * strength))
scaled = image.resize(new_size, Image.LANCZOS)
if strength < 1.0:
padded = Image.new("RGB", (w, h), (128, 128, 128))
padded.paste(scaled, ((w - new_size[0]) // 2, (h - new_size[1]) // 2))
attacked = padded
else:
left = (new_size[0] - w) // 2
top = (new_size[1] - h) // 2
attacked = scaled.crop((left, top, left + w, top + h))

# Verify attack effectiveness
verification_stats = verify_attack_effectiveness(original, attacked, attack_type, strength)

# Save debug images if requested
if save_debug and debug_id:
original.save(f"attack_debug/original_{attack_type}_{strength}_{debug_id}.png")
attacked.save(f"attack_debug/attacked_{attack_type}_{strength}_{debug_id}.png")

# Create side-by-side comparison
comparison = Image.new('RGB', (original.width * 2, original.height))
comparison.paste(original, (0, 0))
comparison.paste(attacked, (original.width, 0))
comparison.save(f"attack_debug/comparison_{attack_type}_{strength}_{debug_id}.png")

return attacked, verification_stats

def run_attack_evaluation_with_verification(attack_type, attack_strength, n_samples=25):
scores = []
labels = []
prompt = "a surreal painting of a jellyfish in space"
verification_stats = []

print(f"\nTesting {attack_type} attack (strength={attack_strength})...")

for i in range(n_samples):
# Use same seeds as baseline for consistency
torch.manual_seed(i) # CHANGED: Use same seeds as baseline
np.random.seed(i + 1000) # CHANGED: Use same seeds as baseline

original_noise = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
base_channel_0 = original_noise[0, 0].clone().float().cpu()
length = base_channel_0.numel()
watermark_array = np.random.choice([-1, 1], size=length)

# NEGATIVE SAMPLE (Clean image under attack)
with torch.no_grad():
image_clean = pipe(prompt=prompt, latents=original_noise, num_inference_steps=50).images[0]

# Apply attack with verification (save debug images for first sample)
save_debug = (i == 0)
debug_id = f"clean_sample{i}"
attacked_clean, clean_stats = apply_attack_with_verification(
image_clean, attack_type, attack_strength, save_debug, debug_id
)
verification_stats.append(clean_stats)

# Try empty prompt for inversion (like Tree-Ring)
inverted_clean = ddim_inversion_fixed(attacked_clean, "", pipe, steps=50) # CHANGED: Empty prompt
recovered_clean = inverted_clean[0, 0].float().cpu()
score_neg = DetectWatermark(recovered_clean, watermark_array)
scores.append(score_neg)
labels.append(0)

# POSITIVE SAMPLE (Watermarked image under attack)
watermarked_noise = original_noise.clone()
watermarked_noise[0, 0] = embed_watermark_wavelet(
base_channel_0, watermark_array, strength=0.1
).to(dtype).to(device)

with torch.no_grad():
image_watermarked = pipe(prompt=prompt, latents=watermarked_noise, num_inference_steps=50).images[0]

# Apply attack with verification
debug_id = f"watermarked_sample{i}"
attacked_watermarked, wm_stats = apply_attack_with_verification(
image_watermarked, attack_type, attack_strength, save_debug, debug_id
)
verification_stats.append(wm_stats)

# Try empty prompt for inversion (like Tree-Ring)
inverted_watermarked = ddim_inversion_fixed(attacked_watermarked, "", pipe, steps=50) # CHANGED: Empty prompt
recovered_watermarked = inverted_watermarked[0, 0].float().cpu()
score_pos = DetectWatermark(recovered_watermarked, watermark_array)
scores.append(score_pos)
labels.append(1)

# Print progress for first few samples
if i < 3:
print(f" Sample {i}: Clean score={score_neg:.4f}, Watermarked score={score_pos:.4f}")

torch.cuda.empty_cache()

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)

target_fpr = 0.001
idx = np.where(fpr <= target_fpr)[0]
tpr_at_target_fpr = tpr[idx[-1]] if len(idx) > 0 else 0.0

# Summary statistics for attack verification
avg_psnr = np.mean([stats['psnr'] for stats in verification_stats])
avg_pixel_change = np.mean([stats['percent_changed'] for stats in verification_stats])

print(f" Attack Summary: Avg PSNR={avg_psnr:.2f}dB, Avg pixel change={avg_pixel_change:.1f}%")

return auc, tpr_at_target_fpr, scores, labels


# Generate original noise
torch.manual_seed(1)
np.random.seed(1)
original_noise = torch.randn((1, 4, 64, 64))

# Create placeholder watermark signature
length = original_noise[0, 0].numel()
watermark_array = np.random.choice([-1, 1], size=length)

# Apply watermark to channel 0 only
watermarked_noise = original_noise.clone()
watermarked_noise[0, 0] = embed_watermark_wavelet(original_noise[0, 0], watermark_array, strength=0.1)

# Convert to device/dtype for SD
latents1 = watermarked_noise.to(device=device, dtype=pipe.unet.dtype)
latents2 = original_noise.to(device=device, dtype=pipe.unet.dtype)
import matplotlib.pyplot as plt
prompt = 'A digital painting of an astronaut in a jungle'
image = pipe(prompt=prompt, latents=latents2, num_inference_steps=50).images[0]

plt.imshow(image)
plt.axis('off'),
plt.title("")
plt.show()
image = pipe(prompt=prompt, latents=latents1, num_inference_steps=50).images[0]

plt.imshow(image)
plt.axis('off'),
plt.title("")
plt.show()


# Define attacks to test (some more aggressive parameters)
attacks_to_test = [
("jpeg", [10, 30, 50, 70]), # CHANGED: Added more aggressive JPEG compression
("gaussian_noise", [0.01, 0.05, 0.1, 0.2]), # CHANGED: Added higher noise level
("rotation", [1, 5, 15, 45]), # CHANGED: Added more aggressive rotation
("scaling", [0.5, 0.8, 1.2, 1.5]) # CHANGED: Added more aggressive scaling
]

print("=" * 70)
print("ATTACK ROBUSTNESS EVALUATION WITH VISUAL VERIFICATION")
print("=" * 70)
print("Baseline Performance: AUC = 1.0000, TPR @ 0.1% FPR = 1.0000")
print("Debug images will be saved to 'attack_debug/' directory")
print("=" * 70)

attack_results = {}
total_start_time = time.time()

for attack_type, strengths in attacks_to_test:
print(f"\nTesting {attack_type.upper()} attacks...")
attack_results[attack_type] = {}

for strength in strengths:
start_time = time.time()
auc, tpr, scores, labels = run_attack_evaluation_with_verification(attack_type, strength, n_samples=10) # CHANGED: Reduced samples for faster debugging
end_time = time.time()

attack_results[attack_type][strength] = {
'auc': auc,
'tpr': tpr,
'time': end_time - start_time
}

print(f" {attack_type} (strength={strength}): AUC={auc:.4f}, TPR@0.1%FPR={tpr:.4f} ({end_time-start_time:.1f}s)")

total_time = time.time() - total_start_time

# Summary Results
print("\n" + "=" * 70)
print("ATTACK ROBUSTNESS SUMMARY")
print("=" * 70)
print(f"Total attack testing time: {total_time/60:.1f} minutes")
print(f"Baseline (Clean): AUC=1.0000, TPR@0.1%FPR=1.0000")
print()

for attack_type, strengths_dict in attack_results.items():
print(f"{attack_type.upper()} Attack Results:")
for strength, results in strengths_dict.items():
print(f" Strength {strength}: AUC={results['auc']:.4f}, TPR@0.1%FPR={results['tpr']:.4f}")
print()

print("Attack evaluation complete!")
print("Check 'attack_debug/' directory for visual verification images")
print("=" * 70)
