# Wave Field PDE — Implementation Plan

**Status:** this repo forked from `wave-field-diffusion` on 2026-07-24 and pivoted to
scientific-field generation (PDE surrogates, fluid dynamics, weather/climate). This
document replaces the old image/audio plan. See `docs/agent-memory/project_pde_pivot.md`
for pivot rationale and hard-won lessons.

**This version reflects a deep design review (2026-07-24).** The spine was reordered:
the *deterministic* wave-operator-vs-FNO comparison is the primary first result, and
diffusion is deferred to the tasks where it is actually motivated. See "Why this order"
below.

---

## Thesis

The wave-field operator is an FFT global convolution — a Fourier Neural Operator (FNO)
cousin — so the "wave equation" framing becomes literal physics when the data is a
physical field. Two distinct questions follow, and the review found they must **not** be
conflated:

- **Q1 (operator quality):** is the FFT damped-oscillator operator competitive with FNO as
  a *deterministic* operator on a PDE benchmark? — Well-posed, cheap, honest. **Primary.**
- **Q2 (diffusion payoff):** on a genuinely *stochastic* field task, does wave-**diffusion**
  give calibrated ensembles, and does physics-timestep kernel conditioning beat AdaLN? —
  The actual novelty, but only motivated where the target distribution is non-degenerate.

**The #1 rule (from prior overclaims): sanity-check the baseline before believing a win.**
Every number is reported against a real, matched, budget-matched baseline whose *published*
result we have reproduced *on the exact data we use*. Sub-quadratic is table stakes here.

---

## Why this order (the review's central finding)

The pivot memo named "2D NS with the diffusion backbone vs FNO" as the first experiment.
The review flagged this as **mis-motivated**, and the plan now corrects it:

- The canonical FNO NS benchmark is **near-deterministic**: a specified initial condition
  determines the solution, so `p(u|a)` is ~a delta. Diffusion can at best *tie* regression
  (via its ensemble mean) and structurally tends to *lose* on pointwise relative L2
  (sampling variance + ensemble-mean blurring of a nonlinear field). FNO reaches ~8e-3
  rel-L2 on ν=1e-3 precisely because the map is nearly deterministic.
- Diffusion earns its keep only under genuine aleatoric uncertainty: chaotic long-horizon
  rollout, partial observation, stochastic forcing, or super-resolution/downscaling. That
  is the *weather* regime (why DeepMind's GenCast is a diffusion model), not smooth NS.

So: **Q1 (deterministic) is the honest, cheap Phase-1 headline.** Diffusion on easy-NS is
run only as a *competence check* ("can it at least match regression?"). The diffusion +
physics-conditioning contribution (Q2) lives on a deliberately stochastic task in Phase 3.

**Also corrected:** the FNO NS task is *temporal* — predict vorticity `w(x, t>T₀)` from the
initial segment `w(x, t≤T₀)` on a periodic torus — **not** the static `a(x)→u(x)` map the
first draft described. Time is handled either as stacked channels (predict the whole future
window at once) or autoregressively (GenCast-style, one step per diffusion sample); the
deterministic Phase 1 uses the stacked-channel form for simplicity.

---

## What the code review verified (grounded facts, not assumptions)

- **The wave conv is *circular* (periodic).** `WaveFieldAttention2D` builds the kernel on
  wrapped grids (`attention.py:444`) and does `rfft2/irfft2` with `s=(Gy,Gx)`, no padding
  (`attention.py:522`). Its implicit boundary condition is therefore **periodic — an exact
  match for the torus NS benchmark**, same BC as FNO. (Corollary: wrong BC for non-periodic
  PDEs — Darcy/Dirichlet would need zero-padded linear conv. NS-periodic is the ideal first
  match by construction.)
- **The `[-1,1]` clamp is real but narrower than first stated.** DDPM `p_sample_step` clamps
  x₀ every step (`diffusion.py:271`); DPM++ clamps only the final output (`diffusion.py:460`);
  **DDIM does not clamp at all** (`diffusion.py:392`) → DDIM is already field-safe. Fix:
  add `clamp_range: tuple | None` (default `[-1,1]` for images, `None` for fields) to
  `_predict_xstart` and the DPM++ final clamp.
- **The wave conv runs at the *patch-grid* resolution**, not full resolution: the kernel is
  built on `Gy=ph, Gx=pw` (post-patchify). At `patch_size=4` on 64², the global conv acts at
  **16×16** — 4× coarser than FNO's full-64² spectral conv. This is a silent capacity
  handicap → the field denoiser must use `patch_size ∈ {1,2}`.
- **Physics-timestep conditioning only exists under diffusion.** It modulates the kernel by
  the *diffusion* `t_emb` (`attention.py:437`). A pure regressor has no noise level to
  condition on → the headline conditioning hypothesis is **not testable in the deterministic
  Phase 1** (see the physical-parameter variant below for what *is* testable there).

---

## What transfers vs. what is new

**Transfers directly (proven — do not rewrite):**
- `wave_field/attention.py` — `WaveFieldAttention2D`, radial kernel `k(r)=exp(-αr)·cos(ωr+φ)`,
  physics/adaln conditioning, dynamic spectral filter, hyena gating, anisotropic kernels.
  This is the operator; it already operates on an H×W grid with **periodic** conv.
- `wave_field/blocks.py` — `WaveFieldBlock`, timestep embedder, AdaLN modulation.
- `wave_field/diffusion.py` — `DDPMDiffusion` (cosine, v-pred, min-SNR-γ=5, DDIM/DPM++/DDPM)
  and `EMA`. Used only in the diffusion phases.
- `denoisers/image.py` — the patchify → blocks → unpatchify structure is the field-denoiser
  template. Fields are already `(B, C, H, W)`; C = physical/time channels, not RGB.
- `scripts/` RunPod pattern (test_push / autopush / EMA-only ckpt / shutdown / preflight).

**New code (small, scoped):**
1. `datasets/navier_stokes.py` — load NS data, form temporal `(input_segment, future_segment)`
   pairs `(C,H,W)`, **per-channel standardization with saved stats**.
2. `baselines/fno.py` — clean FNO-2D (SpectralConv2d + pointwise). Reproduce a published
   rel-L2 *on our data* before trusting anything.
3. `models/field_operator.py` — wave backbone usable as a **plain regressor** `u=G_θ(a)`
   (Phase 1) and as a **diffusion denoiser** with conditioning-field concat (Phase 2+).
   `patch_size ∈ {1,2}`; positional embedding **off by default** (see equivariance note).
4. `metrics/field.py` — relative L2 + radially-averaged energy-spectrum error (+ CRPS /
   spread-skill for the stochastic phase).
5. `train_field.py` — training loop cloned from `train_cifar.py` (AdamW/EMA/AMP/self-cond),
   with a `--mode {regression,diffusion}` switch.
6. `baselines/unet.py`, `baselines/dit.py` — see baseline-set note.

---

## Fairness protocol (non-negotiable — from prior overclaims)

1. **Matched params is necessary but NOT sufficient.** The wave op learns *3 scalars/head*
   for a fixed parametric radial spectrum, L1-normalized so filter gain ≤ 1
   (`attention.py:487`); FNO learns a *free dense complex map per retained mode*. They
   allocate capacity completely differently — report the allocation, don't treat a param
   match as a fair fight on its own.
2. **Matched optimizer steps** — same epochs *and* batch size; never the FAST preset for any
   compared number (it cuts steps 2–4×).
3. **Reproduce the baseline's published number on the exact data we use.** If we use PDEBench
   NS, reproduce FNO *on PDEBench* — do not cite the FNO-paper number on FNO-paper data.
4. **Report rel-L2 AND spectrum error** — a model can win L2 by over-smoothing and lose the
   high-frequency spectrum; the spectrum error catches it.
5. Fixed seed, saved normalization stats, published train/val/test split.

**Baseline set (revised):** FNO (must) + a **U-Net** (the diffusion-for-science workhorse)
+ a **DiT/transformer** (a *quadratic* baseline — the only way any efficiency claim is
meaningful; see below). **Mamba is optional and only fair with multi-directional scans** —
a naive 1D raster-scan SSM on a non-causal 2D field is a strawman (rule #1).

---

## Efficiency story — stated honestly

- **Vs FNO there is NO asymptotic win:** both are O(n log n), and the wave op likely has
  *higher* constants (fp32 FFT per channel under AMP, plus the kernel FFT). The ~8k-token
  crossover from prior work was **vs quadratic attention**, not vs FNO.
- **64² = 4096 tokens is below that crossover** → Phases 1–2 are **quality-only** probes;
  no efficiency claim is admissible there.
- Any efficiency claim requires (a) hi-res ≥128² (16k+ tokens) **and** (b) a transformer/DiT
  baseline in the comparison. That is Phase 4, not the headline.

---

## Physics decisions (open questions made explicit)

- **Forward (noising) process is an open question — decided in Phase 2.** The existing
  `DDPMDiffusion` uses white Gaussian noise, which is spectrally flat and destroys a
  turbulent power-law spectrum inefficiently. The thesis-aligned alternative is
  **heat-dissipation / blurring diffusion** (Rissanen 2022; Hoogeboom & Salimans 2022):
  the forward process *is* the heat equation and the reverse is literally wave-like
  sharpening. Phase 2 evaluates Gaussian-DDPM first (reuses proven code) and decides whether
  to invest in blurring diffusion based on whether generated spectra are correct. Flagged
  first-class so Phases 0–1 are not blocked on it.
- **Naming collision to keep straight in code and writing:** "timestep" means both the
  *diffusion* noise level and the *physical* PDE time. The kernel conditions on the **former**.
- **Physical-parameter conditioning (the more novel angle, and testable in Phase 1):** feed
  viscosity ν / horizon through the same `t_emb` port so the kernel adapts to physical
  *regime* rather than noise level. This is arguably a stronger contribution than the
  image-inherited noise-level conditioning, and it works without diffusion.
- **Metric caveat for the stochastic arm:** a single diffusion sample is not meant to match
  ground truth pointwise; only the **ensemble mean** is comparable to FNO's rel-L2, and the
  mean blurs — so it must be read alongside spectrum error + CRPS, never alone.
- **Normalization determines whether the clamp is even a bug:** standardize (zero-mean/unit-
  var) → must fix the clamp; min-max to [-1,1] → no code change but wastes dynamic range on
  turbulent tails. **Recommendation: standardize + fix the clamp.**
- **Equivariance:** the circular conv is translation-equivariant, a good prior for
  homogeneous turbulence — but patchify + absolute positional embeddings break it (FNO keeps
  it for free). Default the field denoiser to `patch_size ∈ {1,2}` and **no pos-embed**.

---

## Phases

### Phase 0 — Data + FNO baseline + metrics + plumbing (~1 wk)
- [ ] Obtain NS data. **PDEBench NS-2D** (stable HDF5) preferred over the fragile original
      FNO Google-drive `.mat` links; document exact source + split in the loader.
- [ ] `datasets/navier_stokes.py`: temporal `(input_segment, future_segment)` pairs `(C,H,W)`,
      per-channel standardization with saved stats, paper-matched split.
- [ ] `baselines/fno.py`: clean FNO-2D. **Reproduce a published rel-L2 on our data first.**
      If it can't hit the ballpark, stop — the baseline is wrong (rule #1).
- [ ] `metrics/field.py`: rel-L2 + radially-averaged spectrum error, with unit checks
      (identical→0, noise→~1).
- [ ] Add `clamp_range` plumbing to `DDPMDiffusion` (only needed for the diffusion phases,
      but land it now so it's not a Phase-2 surprise).

### Phase 1 — Wave operator vs FNO, deterministic (PRIMARY, ~1.5 wk)
- [ ] `models/field_operator.py` as a plain regressor `u=G_θ(a)`: patchify (P∈{1,2}),
      `WaveFieldBlock`s (`use_2d_kernel=True`), unpatchify; pos-embed off.
- [ ] Train vs FNO at **matched params AND matched steps** (never FAST). Report rel-L2 +
      spectrum error; log per-arch capacity allocation.
- [ ] Physical-parameter conditioning experiment: feed ν/horizon through the `t_emb` port;
      does regime-adaptive kernel modulation help vs a static kernel?
- [ ] **Decision gate:** is the wave operator within striking distance of FNO? If it is far
      worse as an operator, diffusion will not rescue it — diagnose before proceeding.

### Phase 2 — Diffusion competence check + forward-process decision (~1.5 wk)
- [ ] `models/field_operator.py` in diffusion mode: conditioning field `a` concatenated at
      input (reuse self-cond channel-concat plumbing); diffuse `u`. `clamp_range=None`.
- [ ] `train_field.py --mode diffusion`. **Competence question:** can conditional diffusion
      match the Phase-1 regressor on easy-NS (ensemble mean rel-L2)? A tie is the expected
      good outcome; a large loss signals a plumbing bug.
- [ ] Physics-vs-AdaLN conditioning ablation (now testable) at matched params.
- [ ] **Forward-process decision:** inspect generated energy spectra under Gaussian-DDPM;
      decide whether to implement heat-dissipation/blurring diffusion.
- [ ] Kernel diagnostics: do (α, ω) specialize across diffusion timesteps (broad/smooth at
      high noise → sharp/oscillatory at low noise) as they did on images?

### Phase 3 — Stochastic task where diffusion is actually motivated (~2 wk)
- [ ] Choose a genuinely stochastic target (one of: turbulent long-horizon rollout with
      IC-uncertainty blow-up; super-resolution/downscaling; partial-observation infilling).
- [ ] Baselines: **diffusion U-Net** (calibration peer) + FNO (deterministic mean) +
      probabilistic-FNO/deep-ensemble if time — otherwise the uncertainty claim is a strawman.
- [ ] Report ensemble-mean rel-L2, **CRPS, spread-skill ratio, rank histogram**, spectrum.
- [ ] This is where the wave-diffusion + physics-conditioning contribution stands or falls.

### Phase 4 — Efficiency, analysis, honest write-up (~1 wk)
- [ ] Hi-res ≥128²/256² (16k–65k tokens) **with a transformer/DiT baseline** — the only
      regime and comparison where an efficiency crossover is meaningful. Vs FNO report
      quality + constants only, explicitly *not* an asymptotic win.
- [ ] Answer Q1 and Q2 plainly. Target: honest arXiv report / workshop paper, not "SOTA".

---

## Environment & commands (target surface)

```bash
# Phase 0
python -m baselines.fno --data data/ns_pdebench --epochs N        # reproduce the reference number
python -m metrics.field --self-test

# Phase 1 (deterministic, PRIMARY)
python train_field.py --mode regression --arch wave --patch_size 2 --save_dir outputs/ns_wave_reg
python train_field.py --mode regression --arch fno                --save_dir outputs/ns_fno

# Phase 2 (diffusion competence + ablation)
python train_field.py --mode diffusion --arch wave --conditioning physics --save_dir outputs/ns_wave_phys
python train_field.py --mode diffusion --arch wave --conditioning adaln   --save_dir outputs/ns_wave_adaln
python -m metrics.field outputs/ns_wave_phys --n_ensemble 16
```

New deps likely: `h5py`/`scipy` for data loading (add to `requirements.txt`).

---

## Timeline (aggressive — treat as ordering, not a contract)

| Phase | Duration | Milestone |
|---|---|---|
| 0 — Data + FNO baseline + metrics + clamp | ~1 wk | Reproduced FNO rel-L2 on our data |
| 1 — Wave operator vs FNO (deterministic) | ~1.5 wk | Well-posed head-to-head; go/no-go gate |
| 2 — Diffusion competence + fwd-process | ~1.5 wk | physics-vs-adaln; Gaussian-vs-blurring decision |
| 3 — Stochastic task (diffusion payoff) | ~2 wk | CRPS/spread-skill vs U-Net + FNO |
| 4 — Efficiency + write-up | ~1 wk | Hi-res vs transformer; honest report |
| **Total** | **~7 wk** | |
