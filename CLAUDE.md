# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⇒ CURRENT DIRECTION — read this first

**This repo (`wave-field-pde`) forked from `wave-field-diffusion` on 2026-07-24 and pivoted to a new goal: apply the wave-equation / FFT-convolution diffusion operator to scientific-field generation — PDE surrogates, fluid dynamics, weather/climate.** Everything below the "What this project is" heading describes the *prior* image/audio work — it is proven, reusable infrastructure and lineage, **not the current target**. See `docs/agent-memory/project_pde_pivot.md` (also loaded as auto-memory) for the full pivot rationale, current code state, the first experiment, and hard-won lessons.

**Why the pivot, in one paragraph:** The wave operator is an FFT global convolution — an FNO (Fourier Neural Operator) cousin, and FNO is the dominant ML-for-PDE architecture, so the "wave equation" framing becomes literal physics here. Prior work established (against real baselines) that wave beats softmax on generation *quality* at matched budget, but its efficiency edge is *long-context-only*: vs FlashAttention it's a speed win only beyond ~8k tokens and never a memory win. Scientific fields are the domain where sequences are genuinely long (a hi-res field = 10k–1M+ tokens), a spectral/convolutional prior fits, non-causal is fine, and diffusion-for-forecasting is live and unsaturated (cf. DeepMind GenCast, 2024).

**First experiment (start here):** 2D Navier–Stokes (the standard FNO benchmark dataset) with the existing diffusion backbone, vs a plain FNO baseline; metric = field reconstruction error (relative L2 / spectral), not FID. What to build: a field dataset loader (`datasets/`), a field denoiser (adapt `denoisers/image.py`), and a domain metric (`metrics/`). The `wave_field/` operator and `DDPMDiffusion` machinery transfer directly. **Benchmark against FNO and Mamba, not softmax** — sub-quadratic is table stakes in this domain.

**The #1 lesson from prior work: always sanity-check the baseline before believing a win.** Two overclaims were self-caught, both from unfair comparisons (a training-budget confound; benchmarking against naive softmax instead of FlashAttention). Compare against a real, matched FNO/Mamba.

---

## What this project is

*(Prior-work description — the image/audio project this forked from. Retained as infrastructure reference; see the CURRENT DIRECTION section above for what this repo is actually doing now.)*

Exploratory ML research: replacing softmax self-attention in a diffusion denoiser with a wave-equation-inspired mechanism — per-head damped oscillation kernels `k(r) = exp(-α·r)·cos(ω·r + φ)` applied via FFT convolution (O(n log n)). The central hypothesis is that conditioning the kernel physics (α, ω, φ) on the diffusion timestep is a better inductive bias than generic AdaLN conditioning. Full results and the honest efficiency correction are in README.md.

## Environment & commands

Python 3.11 venv at `.venv/` — use `.venv/bin/python` (deps in `requirements.txt`; PyTorch + clean-fid + torchaudio). No test suite, linter, or build system; the correctness gate is:

```bash
python validate_attention.py       # Phase-0 checks: shapes, gradient flow through α/ω/φ,
                                   # kernel diagnostics → outputs/phase0/
```

Training (one script per dataset; heavy runs are done on rented RunPod GPUs, not locally):

```bash
python train_mnist.py --attn wave --conditioning physics --save_dir outputs/my_run
python train_cifar.py ...          # same flag surface; dim=256 depth=6 defaults
python train_audio.py ...          # SC09 waveforms, 16384 samples / patch 16 → 1024 tokens
```

Key shared flags: `--attn {wave,standard}`, `--conditioning {physics,adaln}`, `--kernel {1d,2d}`, `--dynamic_filter`, `--gating {pointwise,hyena}`, `--aniso_kernel`, `--self_cond` (default on), `--num_classes 10` + `--guidance_scale` for class-conditional CFG, `--sampler {ddim,dpmpp,ddpm}`.

Evaluation (Frechet metrics; modality auto-detected from the checkpoint dir name, override with `--modality`):

```bash
python -m metrics.evaluate outputs/my_run --n_samples 10000
python -m metrics.train_classifier --task {mnist,sc09}   # feature extractors; weights committed in metrics/weights/
python benchmark_attn.py                                  # wave vs softmax wall-clock scaling
cd comparison && python compare_cifar.py                  # cross-run sample/loss/kernel diagnostics
```

Full ablation sweeps (designed for unattended RunPod boxes):

```bash
bash scripts/run_image_ablation.sh    # cumulative CIFAR feature ablation (+optional MNIST)
bash scripts/run_sc09_ablation.sh     # 4-config SC09 sweep
# Env knobs: SMOKE=1 (2-epoch dry run), FAST=1 (big-GPU batch), EPOCHS=N,
# AUTOPUSH=1 (git-push small artifacts after each run), SHUTDOWN=1 (self-terminate pod)
```

## Architecture

- **`wave_field/`** — reusable, task-agnostic core. `attention.py`: `WaveFieldAttention` (1D) and `WaveFieldAttention2D` (radial kernel over the patch grid), including the optional dynamic spectral filter (content-adaptive filter expressed as coefficients over a Hann frequency basis — `hann_freq_basis`), hyena-style gating, and anisotropic kernels. `blocks.py`: `WaveFieldBlock` (transformer block with time-conditioned residual gates), timestep/label embedders, AdaLN modulation. `diffusion.py`: `DDPMDiffusion` (cosine schedule, v-prediction, min-SNR-γ=5 weighting, DDIM/DPM++/DDPM samplers) and `EMA` (decay 0.9999 — sampling always uses EMA weights).
- **`denoisers/`** — task-specific DiT-style backbones that patchify input, stack `WaveFieldBlock`s, and unpatchify (`image.py`, `audio.py`).
- **`datasets/sc09.py`** — Speech Commands digit subset, 16 kHz, exactly 16384 samples/clip.
- **`metrics/`** — sample-quality eval. `evaluate.py` rebuilds a model from a run dir's `config.json` + latest checkpoint, generates samples, computes clean-FID (CIFAR) or classifier-based Frechet distance (MNIST FMD / SC09 FSD), writes `metrics.json` into the run dir.
- **`comparison/`** — analysis scripts + written-up findings (`comparison/README.md` has the metric-disagreement discussion); **`results/`** — committed aggregate tables and sample grids.

### The two conditioning modes (the core experiment)

`physics`: a small MLP maps the timestep embedding to per-head perturbations of (α, ω, φ), so kernels reshape across the reverse process (broad/smooth at high noise → sharp/oscillatory at low noise). `adaln`: kernels are static; conditioning happens via standard AdaLN in the block. Every ablation crosses `--attn` × `--conditioning`, and matched parameter counts between arms matter when comparing.

### Run-directory convention

Each training run writes to `outputs/<name>/` (gitignored): `config.json`, checkpoints (`*.pt`), `training.log`, sample grids, and eventually `metrics.json`. The ablation scripts treat an existing `metrics.json` as "run complete" and skip it — delete it to force a re-run. Run names encode the config (e.g. `cifar_wave_dyn_hyena_sc` = wave attn + dynamic filter + hyena gating + self-conditioning). Only small artifacts (json/log/png) get pushed from pods; checkpoints stay on the pod.

## Gotchas

- Never run a model forward under `torch.no_grad()` inside an autocast region that a later grad-enabled forward shares (the self-conditioning pass in `p_losses` is the canonical case). On older torch (pod images ship ~2.4) the autocast weight cache retains the detached casts, silently zeroing gradients for those weights — and crashing `loss.backward()` in configs with no fp32 parameter path (standard attn + adaln). `p_losses` records grad on its first pass and detaches instead; `train_audio.py` additionally sets `cache_enabled=False`.

- `docs/API_KEYS.md` and `docs/INTERVIEW_NOTES.md` are gitignored on purpose — never commit them.
- Kernels are L1-normalized so different (α, ω) give consistent magnitudes; keep this invariant if touching kernel construction.
- The `--attn standard` softmax baseline (`StandardAttention`) is duplicated in each training script *and* in `metrics/evaluate.py` (used to rebuild checkpoints for eval) — changing it means changing all copies.
- The FAST preset raises batch size at a fixed epoch count, i.e. it *cuts optimizer steps 2–4×*. Never use it for runs whose quality will be compared against non-FAST numbers (the 2026-07-18 FAST retrains scored 20–50 FID worse than their bs-128 July counterparts). Also: softmax attention at L=1024 (SC09) needs a ~4 GB attention matrix per layer at bs=256 and OOMs a 24 GB GPU — `run_sc09_ablation.sh` now forces bs=64 for its ablation runs.
