---
name: project-pde-pivot
description: "THIS REPO'S CURRENT DIRECTION — wave/FFT diffusion operator applied to scientific fields / PDEs / weather. Forked from wave-field-diffusion 2026-07-24. Read first."
metadata:
  type: project
---

**This repo (`wave-field-pde`) is a fork of `wave-field-diffusion`, pivoted on 2026-07-24 to a new goal: apply the wave-equation / FFT-convolution diffusion operator to scientific-field generation — PDE surrogates, fluid dynamics, weather/climate.** The image/audio code carried over is proven infrastructure + lineage, not the current target. See [[project_dynamic_wave_filter]] and [[project_wave_field_diffusion]] for the full prior-work history.

**Why this pivot (the reasoning, so it isn't re-litigated):**
- The wave operator is an FFT global convolution — literally an FNO (Fourier Neural Operator) cousin, and FNO is the dominant ML-for-PDE architecture. The "wave equation dynamics" framing stops being a metaphor and becomes real physics here.
- The prior work established (all against real baselines): (a) wave beats softmax on generation *quality* at matched budget (SC09 FSD ~16 vs ~26; CIFAR full stack 55.6 vs 57.4 FID); (b) its efficiency advantage is **long-context-only** — vs FlashAttention wave is a *speed* win only beyond ~8k tokens (up to ~10× at 65k) and **never** a memory win (uses ~1.5× more, FFT materializes complex spectra). At short context FlashAttention wins outright.
- So the operator needs a domain where sequences are genuinely long AND a convolutional/spectral inductive bias fits. Scientific fields are that domain: a hi-res 2D/3D field is 10k–1M+ tokens, long-range coupling is the point, non-causal is fine (modeling a field, not autoregressing), and diffusion-for-forecasting is a live, validated, non-saturated area (DeepMind GenCast beat the operational gold-standard ensemble in 2024).
- Competition here is NOT expensive attention (everyone's already sub-quadratic) — it's FNO, Mamba/S4, Hyena. So "sub-quadratic" is table stakes; the differentiator must be the physics-timestep conditioning or a quality edge. Benchmark against FNO + Mamba, not softmax.

**Reusable code carried over (proven, validated):** `wave_field/` (WaveFieldAttention 1D/2D, blocks, DDPMDiffusion with cosine/v-pred/min-SNR/EMA, DPM++ sampler, class-cond+CFG), `denoisers/` (image/audio DiT-style backbones — templates for a new field denoiser), `metrics/`, `scripts/` (RunPod autopush/EMA/shutdown/preflight scaffolding — reuse the pattern). The 1D/2D wave operator and diffusion machinery transfer directly; a PDE run needs a new field dataset loader + a field denoiser (adapt `denoisers/image.py`) + a domain metric (not FID/FSD).

**First experiment (agreed as the cheapest honest probe):** 2D Navier–Stokes (the standard FNO benchmark dataset, easy to obtain) with the existing diffusion backbone, benchmarked against a plain FNO baseline. Metric: field reconstruction error (relative L2 / spectral error), not FID. Goal: does the wave-diffusion operator match/beat FNO on a real PDE field, and does the physics-timestep conditioning help? This validates the direction before any scale-up.

**Hard-won lessons from prior work (DO NOT repeat) — full detail in [[project_dynamic_wave_filter]]:**
- **Always sanity-check the BASELINE before believing a win.** Two overclaims were self-caught, both from unfair comparisons: (1) a training-budget confound (bs256-FAST vs bs64), (2) benchmarking against naive materialized softmax instead of FlashAttention. Here that means: compare against a *real* FNO/Mamba, matched budget.
- AMP + `torch.no_grad()` self-cond bug: never run a model forward under `no_grad` inside an autocast region a grad pass shares (fixed in `wave_field/diffusion.py` p_losses; `cache_enabled=False` in trainers). Torch 2.4 on pods.
- The FAST preset raises batch at fixed epochs → 2–4× fewer optimizer steps → never use for cross-run quality comparisons.
- Pod workflow: RunPod, clone with PAT, `bash scripts/test_push.sh` first, always `AUTOPUSH=1 SHUTDOWN=1` for unattended runs, EMA-only checkpoints pushed (full ones die with the pod). Keys in `docs/API_KEYS.md` (gitignored).

**Realistic ceiling (from the 2026-07-23 assessment):** not a top-venue paper as-is (borrowed Wave-Field-LLM mechanism, needs scale + real baselines); a workshop paper / arXiv report is achievable with honest framing. The PDE/weather-diffusion niche is the direction where the efficiency story, the physics framing, and a real funded need all align.
