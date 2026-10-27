import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, roc_auc_score
import time

# Auto device + dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16 if device.type == 'cuda' else torch.float32

# Load Stable Diffusion 1.5 with DDIM
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def visualize_wavelet_bands(coeffs_before, coeffs_after, title_prefix="Wavelet"):
    cA_before, (cH_before, cV_before, cD_before) = coeffs_before
    cA_after,  (cH_after,  cV_after,  cD_after)  = coeffs_after

    bands = ['Approx (cA)', 'Horiz (cH)', 'Vert (cV)', 'Diag (cD)']
    before = [cA_before, cH_before, cV_before, cD_before]
    after  = [cA_after,  cH_after,  cV_after,  cD_after]

    plt.figure(figsize=(12, 8))
    for i, (b, a) in enumerate(zip(before, after)):
        plt.subplot(4, 2, 2*i + 1)
        plt.imshow(b, cmap='gray')
        plt.title(f"{title_prefix} {bands[i]} Before")
        plt.axis('off')

        plt.subplot(4, 2, 2*i + 2)
        plt.imshow(a, cmap='gray')
        plt.title(f"{title_prefix} {bands[i]} After")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def embed_watermark_wavelet(
    noise: torch.Tensor,
    watermark_array,
    strength=0.02,
    wavelet='haar',
):
    noise_np = noise.detach().cpu().numpy()
    coeffs = pywt.dwt2(noise_np, wavelet)
    cA, (cH, cV, cD) = coeffs
    coeffs_before = (cA.copy(), (cH.copy(), cV.copy(), cD.copy()))

    # ---- Embed PN in all four bands: cA, cH, cV, cD ----
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

    flat_cA = cA.flatten()
    flat_cH = cH.flatten()
    flat_cV = cV.flatten()
    flat_cD = cD.flatten()

    flat_cA += strength * w_a
    flat_cH += strength * w_h
    flat_cV += strength * w_v
    flat_cD += strength * w_d

    cA_mod = flat_cA.reshape(cA.shape)
    cH_mod = flat_cH.reshape(cH.shape)
    cV_mod = flat_cV.reshape(cV.shape)
    cD_mod = flat_cD.reshape(cD.shape)

    coeffs_after = (cA_mod, (cH_mod, cV_mod, cD_mod))

    watermarked_np = pywt.idwt2(coeffs_after, wavelet)
    return torch.tensor(watermarked_np, device=noise.device, dtype=noise.dtype)

def DetectWatermark(noise: torch.Tensor, watermark_array, wavelet='haar') -> float:
    noise_np = noise.detach().cpu().numpy()
    coeffs = pywt.dwt2(noise_np, wavelet)
    cA, (cH, cV, cD) = coeffs

    # Flatten bands
    flat_cA = cA.flatten()
    flat_cH = cH.flatten()
    flat_cV = cV.flatten()
    flat_cD = cD.flatten()

    # Split watermark array for all 4 bands
    len_a = flat_cA.size
    len_h = flat_cH.size
    len_v = flat_cV.size
    len_d = flat_cD.size
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

    # Compute correlation scores for all 4 bands
    corr_a = safe_corr(flat_cA, w_a)
    corr_h = safe_corr(flat_cH, w_h)
    corr_v = safe_corr(flat_cV, w_v)
    corr_d = safe_corr(flat_cD, w_d)

    # Average correlation score
    score = (corr_a + corr_h + corr_v + corr_d) / 4.0
    return score

def ddim_inversion_fixed(image, prompt, pipe, steps=50):
    """
    Proper DDIM inversion implementation for watermarking applications
    """
    # Create inverse scheduler with same config as forward scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler.set_timesteps(steps)

    # Prepare image tensor with correct preprocessing
    if isinstance(image, Image.Image):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        image_tensor = transform(image).unsqueeze(0).to(pipe.device).to(pipe.dtype)
    else:
        image_tensor = image.to(pipe.device).to(pipe.dtype)

    # Encode to latent space using VAE with proper scaling
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor  # Use model-specific scaling

    # Prepare text embeddings properly
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]

    # Perform inversion: clean image (t=0) → noise (t=T)
    current_latents = latents
    for t in inverse_scheduler.timesteps:
        # Scale model input
        latent_model_input = inverse_scheduler.scale_model_input(current_latents, t)

        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

        # Step backwards (add noise) using inverse scheduler
        current_latents = inverse_scheduler.step(
            noise_pred, t, current_latents
        ).prev_sample

    return current_latents

# ===== EVALUATION LOOP - 100 ITERATIONS (200 TOTAL SAMPLES) =====
scores = []
labels = []
prompt = "a surreal painting of a jellyfish in space"
num_inference_steps = 50

print("Starting comprehensive watermarking evaluation...")
print("Testing 100 watermarked images + 100 original images (200 total samples)")
print("Estimated completion time: ~32 minutes")
print("=" * 60)

start_time = time.time()

for i in range(100):  # CHANGED FROM 25 TO 100 ITERATIONS
    torch.manual_seed(i)
    np.random.seed(i + 1000)

    # Generate base latent
    original_noise = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
    base_channel_0 = original_noise[0, 0].clone().float().cpu()
    length = base_channel_0.numel()
    watermark_array = np.random.choice([-1, 1], size=length)

    # NEGATIVE SAMPLE (Original/Non-watermarked)
    with torch.no_grad():
        image_clean = pipe(prompt=prompt, latents=original_noise, num_inference_steps=num_inference_steps).images[0]

    inverted_clean = ddim_inversion_fixed(image_clean, prompt, pipe, steps=num_inference_steps)
    recovered_clean = inverted_clean[0, 0].float().cpu()
    score_neg = DetectWatermark(recovered_clean, watermark_array)
    scores.append(score_neg)
    labels.append(0)

    # POSITIVE SAMPLE (Watermarked)
    watermarked_noise = original_noise.clone()
    watermarked_noise[0, 0] = embed_watermark_wavelet(
        base_channel_0, watermark_array, strength=0.1).to(torch.float16).to(device)

    with torch.no_grad():
        image_watermarked = pipe(prompt=prompt, latents=watermarked_noise, num_inference_steps=num_inference_steps).images[0]

    inverted_watermarked = ddim_inversion_fixed(image_watermarked, prompt, pipe, steps=num_inference_steps)
    recovered_watermarked = inverted_watermarked[0, 0].float().cpu()
    score_pos = DetectWatermark(recovered_watermarked, watermark_array)
    scores.append(score_pos)
    labels.append(1)

    #UPDATED PROGRESS REPORTING
    if (i + 1) % 10 == 0:
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (i + 1)) * 100
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"Completed {i + 1}/100 iterations... (Pos: {score_pos:.4f}, Neg: {score_neg:.4f})")
        print(f"  Progress: {((i + 1) / 100) * 100:.1f}% | Elapsed: {elapsed_time/60:.1f}min | ETA: {remaining_time/60:.1f}min")
        
    torch.cuda.empty_cache()

total_time = time.time() - start_time

# ===== COMPREHENSIVE METRICS CALCULATION =====
scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)

target_fpr = 0.001  # 0.1% 
idx = np.where(fpr <= target_fpr)[0]
tpr_at_target_fpr = tpr[idx[-1]] if len(idx) > 0 else 0.0

# ===== COMPREHENSIVE RESULTS OUTPUT =====
print("\n" + "=" * 60)
print("COMPREHENSIVE WATERMARKING EVALUATION RESULTS")
print("=" * 60)
print(f"Total Samples: {len(scores)} ({np.sum(labels)} watermarked + {len(scores) - np.sum(labels)} original)")
print(f"Total Runtime: {total_time/60:.1f} minutes")
print(f"Average time per iteration: {total_time/100:.1f} seconds")

print(f"\n PRIMARY METRICS:")
print(f"AUC = {auc:.4f}")
print(f"TPR @ {target_fpr*100}% FPR = {tpr_at_target_fpr:.4f}")

print(f"\n SCORE STATISTICS:")
print(f"Positive scores: mean={np.mean(scores[labels==1]):.4f}, std={np.std(scores[labels==1]):.4f}")
print(f"Negative scores: mean={np.mean(scores[labels==0]):.4f}, std={np.std(scores[labels==0]):.4f}")
print(f"Score difference (mean): {np.mean(scores[labels==1]) - np.mean(scores[labels==0]):.4f}")

print(f"\n SCORE RANGES:")
print(f"Positive: [{np.min(scores[labels==1]):.4f}, {np.max(scores[labels==1]):.4f}]")
print(f"Negative: [{np.min(scores[labels==0]):.4f}, {np.max(scores[labels==0]):.4f}]")

# Cohen's d (effect size)
pooled_std = np.sqrt(((np.std(scores[labels==1])**2 + np.std(scores[labels==0])**2) / 2))
cohens_d = (np.mean(scores[labels==1]) - np.mean(scores[labels==0])) / pooled_std
print(f"\n STATISTICAL ANALYSIS:")
print(f"Effect Size (Cohen's d): {cohens_d:.4f}")

# Performance interpretation with sample size consideration
if auc >= 0.95:
    performance = "Excellent"
elif auc >= 0.8:
    performance = "Good" 
elif auc >= 0.7:
    performance = "Fair"
else:
    performance = "Poor"

print(f"Performance Assessment: {performance}")
print(f"Statistical Confidence: High (n={len(scores)} samples)")

# Separation analysis
min_positive = np.min(scores[labels == 1])
max_negative = np.max(scores[labels == 0])
separation_gap = min_positive - max_negative

print(f"\n CLASS SEPARATION ANALYSIS:")
print(f"Minimum positive score: {min_positive:.6f}")
print(f"Maximum negative score: {max_negative:.6f}")
print(f"Separation gap: {separation_gap:.6f}")

if separation_gap > 0:
    print("PERFECT CLASS SEPARATION: All watermarked > all non-watermarked")
else:
    print(" CLASS OVERLAP EXISTS")

#print("=" * 60)
#print("EVALUATION COMPLETE - Ready for attack testing and further analysis!")
#print("=" * 60)


======================================================================
JPEG ATTACK ROBUSTNESS SUMMARY
======================================================================
Total testing time: 74.5 minutes

JPEG Attack Results:
  Quality 10: AUC=0.9523, TPR@1%FPR=0.5200
  Quality 30: AUC=0.9909, TPR@1%FPR=0.8600
  Quality 50: AUC=0.9978, TPR@1%FPR=0.9700
  Quality 70: AUC=0.9979, TPR@1%FPR=0.9900

JPEG attack evaluation complete!
======================================================================

