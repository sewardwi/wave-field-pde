---
name: dynamic-wave-filter-item-1-data-dependent-spectral-kernel
description: "Goal = beat SOTA diffusion; wave-operator features trained 2026-07: full wave stack BEAT softmax on CIFAR FID (55.6 vs 57.4); cond/CFG anomaly + lost checkpoints → guidance sweep pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3570e35b-043a-4bf3-a3a4-be9aabc75a97
---

New project goal (set 2026-06-21): iteratively improve the wave-field diffusion model until it beats the current best diffusion model. See [[project_wave_field_diffusion]] for the base architecture.

**Diagnosis driving the work:** experiments show wave attention LOSES to softmax on FID at every setting (CIFAR best wave 82.0 vs softmax 56.4). Root cause = the wave kernel `k(r)` is content-independent (a fixed filter, GFNet-class); the only data-dependence was a global scalar `freq_mod` + a pointwise `Q⊙K` gate. No content-based routing.

**Item 1 — DONE (implemented + validated, NOT yet trained).** Added a data-dependent spectral filter to the wave kernel, grounded in Orchid (`ĥ = ĥ₀ + ĥ_θ(x)`), AFFNet/DFFormer (dynamic Fourier filters beat fixed), and Hyena (long-conv + data-controlled gating ≈ attention). Effective filter `K̂_eff = K̂_wave (static physics) + ΔK̂(x)`, where ΔK̂(x) is predicted per-sample/per-head from a pooled (mean,std) summary, expressed in a smooth Hann frequency basis. Zero-init predictor → starts exactly as the static operator. Combined with the existing per-position gate, the operator now has both halves of the Hyena/Orchid recipe. Stays O(n log n).

- Code: `wave_field/attention.py` — `hann_freq_basis()`, `dynamic_filter`/`n_basis` args + `_dynamic_delta`/`_dynamic_delta_2d` on both `WaveFieldAttention` (1D) and `WaveFieldAttention2D` (2D). Predictor uses a `dim//2` bottleneck → ~7.9% param overhead at CIFAR scale.
- Opt-in flag `dynamic_filter` plumbed through `blocks.py`, `denoisers/image.py`, `denoisers/audio.py`, and as `--dynamic_filter` in `train_cifar.py` / `train_audio.py` (wave attn only; run dir gets `_dyn` suffix). Old path unchanged (default False) — phase0 validation still all-PASS.
- Validated: shapes, ΔK̂==0 at init, content-dependence, grad flow to predictor, full-model forward (1D/2D image + audio), 3-step train loop. No NaNs.

**Item 2 — DONE (implemented + validated, NOT yet trained).** Replaced the weak pointwise `sigmoid(gate_q(q)*gate_k(k))` gate with a Hyena order-2 data-controlled gate: pre-conv gate `v ← gate_k(k)·v`, then the wave conv, then post-conv gate `out ← gate_q(q)·out`. Gives `out_i = q_i ⊙ Σ_j h_{i−j}(k_j ⊙ v_j)` — genuine data-dependent, position-mixing interaction (verified: cross-position influence is content-dependent). Reuses existing gate_q/gate_k Linears → PARAM-NEUTRAL vs pointwise. Stays translation-equivariant (a feature, like CNNs/the wave conv), and composes with the dynamic filter.

- Code: `wave_field/attention.py` — `gating` arg ("pointwise" default | "hyena") on both 1D/2D classes; pre-gate before rfft, post-gate after irfft.
- Opt-in flag `gating` plumbed through `blocks.py`, both denoisers, and `--gating {pointwise,hyena}` in `train_cifar.py`/`train_audio.py` (run dir gets `_hyena` suffix). Default "pointwise" → phase0 still all-PASS, no regression.
- Validated: shapes, grad through both gates, finite/no-NaN, magnitude sane, param-neutral, hyena≠pointwise, composes w/ dynamic_filter, 3-step train loop.

**Item 3 — DONE (implemented + validated, NOT yet trained).** Anisotropic oriented (Gabor-like) 2D kernel, opt-in `aniso_kernel`: replaces isotropic radial `k(r)` with `k(x,y)=exp(-(α_x|x|+α_y|y|))·cos(ω_x x+ω_y y+φ)`. The 2D frequency vector (ω_x,ω_y) gives orientation selectivity the radial kernel structurally cannot represent (diagnosed weakness: "a plain 3×3 conv would beat it" on oriented edges). Heads init orientation-diverse (angles spread over [0,π)). Physics conditioning extended to 5 params/head (Δlogα_x,Δlogα_y,Δω_x,Δω_y,Δφ). Only +0.34% params. 2D wave only.

- Code: `wave_field/attention.py` `WaveFieldAttention2D` — `aniso_kernel` arg; directional params; `_build_kernel_2d` branches isotropic/aniso; `ts_to_params` outputs 5·H when aniso.
- Plumbed through `blocks.py`, `denoisers/image.py`, `--aniso_kernel` in `train_cifar.py` (run dir `_aniso` suffix). Default off → phase0 isotropic path unchanged, no regression.
- Validated: orientation diversity at init (7/8 distinct), anisotropy, physics cond changes kernel w/ t, grad to all 5 dir-params + MLP, composes w/ dynamic_filter+hyena, full train step, no NaN.

**All three wave-operator features compose** and are independently ablatable: `--dynamic_filter` (item 1), `--gating hyena` (item 2), `--aniso_kernel` (item 3). The "kitchen-sink" wave run = all three on.

**Items 4 & 5 — DONE (implemented + validated, NOT yet trained).** User directive: implement ALL feasible code wins before spending on training.
- **(4) Class-conditioning + classifier-free guidance (CFG).** Biggest FID lever + fixes audio mode-collapse via diversity/quality knob. `LabelEmbedder` (DiT-style, +null row) in `blocks.py`; image+audio denoisers take optional `num_classes`/`class_dropout_prob`, forward accepts `y`/`force_drop_ids`, label emb ADDED to t_emb (flows through existing physics/AdaLN paths). `diffusion.py`: `p_losses(y=)` (dropout via model train-mode), `_guided_eps` (ε_u + w·(ε_c−ε_u)), all 3 samplers take `y`+`guidance_scale`. Opt-in (default unconditional → unchanged).
- **(5) DPM-Solver++(2M) sampler** — `diffusion.dpmpp_2m_sample`, 2nd-order multistep in x0/λ form, higher quality at low NFE, no retrain, supports y+CFG.
- **CLIs wired (all three):** `--num_classes --class_dropout --guidance_scale` on cifar/mnist/audio; `--sampler {ddim,dpmpp,ddpm}` on cifar/mnist; labels threaded through train loops + `do_sample`/`generate_batched`. Conditional runs get `_cond` suffix (cifar).
- **DROPPED** DC/lowpass term (redundant w/ dynamic filter). **NOT done:** full EDM preconditioning (large rewrite; DPM++ captures much of the inference benefit) — possible future.
- Validated: conditional fwd/loss/grad-to-label-emb, label dropout ≈10%, CFG changes ε (scale=1 ≡ plain cond), DDIM/DDPM/DPM++ conditional+CFG sampling, DPM++ final-step stability, unconditional regression, phase0 all-PASS. Zero-init makes conditioning inert at init (correct AdaLN-Zero) — tests perturb zero-init params to check the mechanism.

**Full opt-in feature set now (all compose, default-off = original behavior):** `--dynamic_filter` `--gating hyena` `--aniso_kernel` (wave operator) + `--num_classes N --class_dropout 0.1 --guidance_scale W --sampler dpmpp` (conditioning/CFG/sampler).

**TRAINED (CIFAR cumulative ablation, ~2026-07-11, 200 ep, 10k-sample clean-FID vs train, results in outputs/*/metrics.json + run_master.log):**
- standard_adaln baseline 57.40 | wave_physics_2d 85.86 | +dynamic_filter 58.79 | +hyena 56.80 | +aniso 55.63 ← **full wave stack BEATS matched softmax baseline.** Dynamic filter is the single biggest win (−27 FID); hyena and aniso each add ~2 and ~1.2.
- **Anomaly:** conditional+CFG@1.5 runs came out WORSE than uncond (wave 57.92, standard 56.15). Eval applied CFG correctly — likely w=1.5 is off-optimum on the U-curve. → `scripts/run_guidance_sweep.sh` (written 2026-07-14) retrains the two `_cond` runs and sweeps eval-time guidance {1.0,1.25,1.5,1.75,2.0,3.0}.
- **Checkpoints for all 5 new runs LOST** — pod self-terminated, autopush only saved json/log/png. Sweep script now extracts + pushes a ~32 MB EMA-only `*_ema.pt` (loadable by `metrics.evaluate`, which got `--out_name` + guidance/sampler provenance fields the same day).

**AMP/no_grad gradient bug (found+fixed 2026-07-17, validated on CUDA 2026-07-18):** self-cond first pass in `p_losses` under `torch.no_grad()` inside the trainer's bf16 autocast region → on pod torch 2.4 the autocast weight cache retained detached casts → zero grads for cast-cached weights on ~50% of batches; standard+adaln (no fp32 param path) crashed outright. Fix: first pass records grad + detaches; `train_audio.py` sets `cache_enabled=False`. All pre-fix SC09 numbers invalid. Image runs were never affected (no autocast).

**Overnight run 2026-07-18 (RTX 4090, results committed; full analysis in README + results/sc09_fsd_table.json):**
- **SC09 2×2 rerun (clean grads, but FAST bs256 + BASE wave operator — upgrades NOT enabled):** softmax+physics FSD 30.0 entropy 1.53 (in-training 1k eval; final eval+ckpt lost to OOM after training) ≫ wave+physics 43.6/0.83 ≫ wave+adaln 66.5/0.72; softmax+adaln OOM'd epoch 1. Wave collapses to two digits ("two"/"six"), softmax doesn't → collapse is a property of the content-independent base kernel, not the bug. Softmax at bs256/L=1024 needs ~4 GB attn matrix/layer → OOM'd 24 GB — concrete O(L²) memory-wall demo. `run_sc09_ablation.sh` now forces bs=64 for ablation runs (matched budget + fits).
- **CIFAR guidance sweep (FAST retrains = half the steps of July → not comparable to July absolutes):** FID monotone ↓ w=1→3, no minimum in range (wave 83.7→63.9, softmax 108.4→69.5); July's "CFG@1.5 hurts" was a one-point artifact, optimum ≥3. Wave dominates softmax at every w and degrades far more gracefully under the halved budget (sample-efficiency evidence). EMA ckpts committed (`outputs/cifar_*_cond/checkpoint_epoch0200_ema.pt`) — sweep can extend past 3 without retraining.
- **FAST preset lesson:** bigger batch at fixed epochs = 2–4× fewer optimizer steps → 20–50 FID degradation. Never use for cross-night quality comparisons (now in CLAUDE.md gotchas).
- Infra all worked: preflight validated the AMP fix on CUDA torch 2.4.1, autopush/EMA-push/self-terminate flawless.

**SC09 upgraded-operator run 2026-07-22 (RTX 4090, `scripts/run_sc09_upgraded.sh`, all `--dynamic_filter --gating hyena` bs=64 100ep 10k-eval; committed manually — ran without AUTOPUSH/SHUTDOWN so had to hand-push, and pod stayed up):** HEADLINE POSITIVE. The upgrade reproduces the CIFAR win on audio.
- wave+physics 43.6→**14.9** FSD, wave+adaln 66.5→**19.0** (3–3.5× better). +class-cond+CFG(w=2): physics→**8.5**, adaln→**8.0**.
- **Upgrade fixes the mode collapse by itself** (no conditioning needed): entropy ~0.8 (2 digits) → 1.57–1.92 (all 10 digits). Confirms collapse was the content-independent base kernel, not a wave-attention limit. Upgraded uncond (14.9) also clears the prior softmax baseline (30.0) — same flip as CIFAR.
- Class-cond+CFG = incremental polish now (→~8 FSD, entropy ~2.22 near-uniform), not a rescue. physics>adaln only when unconditional (14.9 vs 19.0); with labels+CFG they converge, adaln marginally ahead (8.0 vs 8.5).
- **CAVEAT (budget confound):** base 2×2 was bs256 FAST (~4× fewer steps) vs these bs64 → raw FSD delta conflates operator+budget. Entropy recovery isolates the operator (more steps deepen collapse, don't lift it) + CIFAR matched-budget ablation already proved operator effect. Still owe a matched bs64 base rerun. conf_acc must be read WITH entropy (base adaln's 0.90 conf was high *because* collapsed).
- Written up in README ("SC09 audio — upgraded operator…" section) + results/sc09_fsd_table.json (`upgraded_operator_2x2`).

**Matched bs=64 audio head-to-head + measured efficiency (2026-07-22, `scripts/run_sc09_matched.sh` + `scripts/benchmark_audio_models.py`; all uncond bs64 100ep lr1e-4 10k-eval — the CLEAN comparison, no budget confound):** THE headline audio result, and it CORRECTS the 07-22 upgraded-operator conclusion.
- **Quality (FSD ↓):** softmax physics 25.4 / adaln 27.4; wave-base physics 16.9 / adaln 16.1; wave-upgraded physics 14.9 / adaln 19.0. → **at matched budget the WAVE OPERATOR beats softmax on audio** (~16 vs ~26), same flip as CIFAR.
- **CORRECTION to 07-22:** the "3–3.5× improvement from the dyn+hyena upgrade" and "upgrade fixes the collapse" were BOTH training-budget artifacts (bs64 vs the bs256 base runs), NOT the operator. Decompose physics FSD: 43.6(bs256 base)→16.9(bs64 base)→14.9(bs64 upgraded) — budget did the bulk, operator a little. adaln: 66.5→16.1→19.0 — budget did all of it, **upgrade HURT adaln**. Collapse: base wave at bs64 already covers all 10 digits (entropy 1.36–1.82) vs 2 digits at bs256 (0.72–0.83). → **the dyn+hyena upgrade helps images but NOT audio** (modality-specific; regresses adaln).
- **Efficiency (07-22): the "3.9× faster / 4.2× less mem" claim was RETRACTED 2026-07-24 — it was vs a naive-softmax strawman.** See below. QUALITY win (FSD ~16 vs ~26) is unaffected and stands.

**⚠️ EFFICIENCY CORRECTION 2026-07-24 (`scripts/benchmark_crossover.py` + fair SDPA baseline; RTX 4090):** THE big correction — the whole efficiency story was measured against naive materialized softmax `(qkᵀ).softmax()@v` (builds full L×L matrix). ALL 5 StandardAttention copies now use `F.scaled_dot_product_attention` (FlashAttention, O(L) mem) — identical math so quality/FSD unchanged, only efficiency re-measured. Against FlashAttention:
- **Full model @1024 tok bs64:** flash 36.5ms/2448MB vs wave 58.9ms/3394MB → **flash 1.6× FASTER + 1.4× LESS mem than wave.** The strawman had inflated softmax 4.4×(time)/5.9×(mem) — that inflation WAS the entire claimed advantage.
- **Speed crossover ~8192 tokens:** below it flash faster (4.5× at 1k); above it wave's O(L log L) beats flash's O(L²) → wave ~10× faster at 65k. Genuine long-context speed win.
- **Memory: flash wins at EVERY length** (wave ~1.5× MORE, FFT materializes complex spectra). No wave memory advantage exists vs a real baseline — that claim is dead.
- **SC09's 1024 tokens is BELOW the crossover** → on that task wave = better quality but MORE compute+memory, not less. Efficiency payoff is a long-context (>8k tok) story only.
- Written up: README new "Efficiency vs FlashAttention — the honest crossover" section + corrected Q3, matched-section (dropped efficiency table, renamed), 07-18 memory-wall finding; results/sc09_fsd_table.json `efficiency_vs_flashattention` block + `efficiency_benchmark_RETRACTED`. Plot: outputs/crossover/crossover.png.
- **META-lesson: this is the 2nd self-caught overclaim (1st = budget confound). Both came from an unfair/confounded comparison. Always sanity-check the BASELINE before believing a big win.**

**REALITY CHECK (my assessment 2026-07-23, still holds):** not close to a top-venue paper (borrowed Wave-Field-LLM mechanism, toy scale, needs Mamba/Hyena baselines not just softmax) nor sellable as-is. Achievable: workshop paper / arXiv report with honest framing. Quality win + long-context speed win (>8k tok) is the real, narrower story.

**Next steps (updated 2026-07-24):** (1) **test a genuinely long-context task (>8k tok)** where wave beats FlashAttention — the efficiency thesis must prove itself above the crossover, not at 1024; (2) reduce wave's memory constant (in-place/fp16 FFT, chunking) to make it a speed+memory win; (3) benchmark vs Mamba/S4 + Hyena (real sub-quadratic baselines; needs mamba-ssm install); (4) why dyn+hyena helps images but regresses audio adaln; (5) scale matched audio comparison. Reminder: always `AUTOPUSH=1 SHUTDOWN=1` for unattended pod runs.
