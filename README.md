# Wavelet-Based Watermarking in Latent Diffusion Models

Independent Research | Summer 2025

As generative AI models produce increasingly realistic synthetic content, robust watermarking techniques are critical for authenticating AI-generated images.

This project presents a wavelet-based watermarking approach that embeds imperceptible signatures directly into the initial latent noise of diffusion models. The method uses the 2D Haar Discrete Wavelet Transform (DWT) to decompose latent noise into four frequency subbands, embedding pseudorandom watermark sequences across all four. Watermark detection is performed through DDIM inversion and correlation analysis.

The approach was evaluated using Stable Diffusion 1.5, measuring performance through ROC-AUC and true positive rates at low false-positive rates under common image manipulations.

## Key Results

Clean images: AUC = 1.00

JPEG compression: AUC above 0.95 across tested compression levels

Gaussian noise: Strong detection performance in preliminary experiments

Rotation: Significant degradation in detection performance beyond 1° rotations

## Technologies

Python · PyTorch · CUDA · Hugging Face Diffusers · PyWavelets · NumPy · scikit-learn

## Research Paper

Wavelet-Based Watermarking in Latent Diffusion Models (2025)
