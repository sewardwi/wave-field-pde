---
name: Wave Field Diffusion — project overview
description: Core architecture, file layout, and design decisions for the wave-field diffusion research project
type: project
---

Research project: replace standard self-attention in a diffusion denoising backbone with FFT-convolution-based "wave field attention" (damped oscillation kernels). Goal is O(n log n) scaling + physically interpretable denoising.

**Why:** Exploratory ML research — test whether wave-equation dynamics in the backbone helps frequency specialization emerge across diffusion timesteps.

**How to apply:** When asked about the project architecture or extending it, use this file as the ground truth on structure and decisions.

## File layout

```
wave-field-diffusion/
├── wave_field/
│   ├── attention.py   — WaveFieldAttention (1D), WaveFieldAttention2D (2D radial)
│   ├── model.py       — WaveFieldDenoiser (patchify → N blocks → unpatchify)
│   ├── diffusion.py   — DDPMDiffusion (linear schedule, DDPM + DDIM sampling)
│   └── __init__.py
├── train_mnist.py     — Phase 1: MNIST 28×28
├── train_cifar.py     — Phase 2: CIFAR-10 32×32 (1D or 2D kernels, wave vs standard attn ablation)
├── validate_attention.py — Phase 0: all-pass validation + kernel diagnostic plots
└── requirements.txt
```

## Key design decisions

- **Kernel:** `k(t) = exp(-α|t|) · cos(ω·t + φ)` — non-causal (full symmetric), applied via FFT convolution on sequence dim. K projection unused in wave-field path; Q used for content gate.
- **Physics conditioning (Option B):** `ts_to_params` MLP maps timestep embedding → (Δlog_α, Δω, Δφ) per head. Zero-init → starts as identity perturbation. FFN gets FiLM scale/shift from t_emb.
- **AdaLN conditioning (Option A):** DiT AdaLN-Zero — 6 modulation values per block. Zero-init gate → identity residuals at start.
- **2D variant:** `WaveFieldAttention2D` uses radial kernel `k(r) = exp(-α·r)·cos(ω·r+φ)` + `rfft2`/`irfft2`. Used for Phase 2 CIFAR Approach B.
- **Model sizes:** MNIST ~475K params (physics) / ~550K (adaln). CIFAR ~6.5M (physics, dim=256, depth=6).
- **Patch sizes:** MNIST 4×4→49 tokens; CIFAR 4×4→64 tokens.
- **Validated:** All 24 Phase 0 checks pass (shapes, gradients through α/ω/φ, kernel structure, physics conditioning changes kernels with timestep, no NaNs).
