# Plan — Calibrated Probabilistic Renewable-Energy Forecasting (credibility artifact → industry → revenue)

**Pivot date:** 2026-08-02. This repo began as `wave-field-diffusion` (image/audio), forked to
`wave-field-pde` (PDE surrogates), and now pivots again to a concrete, commercially-relevant
target: **calibrated probabilistic day-ahead wind/solar power forecasting on 100% public data.**

## Why this, stated honestly

The goal is **revenue**, but the operator is purely technical with no industry network — the hardest
combination, because industrial ML is sold on trust/relationships, not model quality. So the plan is
**not** "launch a solo product cold." It's to build **one genuinely excellent, public, rigorous
artifact** that does quadruple duty:
1. **Credibility / inbound** that substitutes for a network,
2. **A portfolio piece** that lands a role at a funded player (Jua, Silurian, PhysicsX, an energy
   desk) → the income, domain, and network we currently lack,
3. **Real market learning** (do buyers care? what's the real bar?) without needing intros first,
4. **The seed of a product** if interest shows up.

The honest ladder: **artifact → into the industry → network + domain + capital → found from strength.**
Every rung has standalone value. The model is a commodity component (NVIDIA gives FourCastNet away free);
the durable value is **calibrated uncertainty + rigor + economic framing + the last mile.**

**Non-negotiable ethos (carried from prior work): sanity-check the baseline before believing a win.**
Here that means beating the grid operators' *own published forecast*, not a strawman.

---

## The artifact, precisely scoped

**Task:** Day-ahead (and intraday) **probabilistic** forecasts of aggregate **wind and solar power**
generation for a European **bidding zone**, on fully public data, evaluated against the incumbent
operational forecast and translated into **money**.

- **Probabilistic, not point:** predict full predictive distributions / quantiles, because calibrated
  uncertainty is the actual product value (traders price risk; imbalance is a cost).
- **Region/zone-level first** (aggregate generation per bidding zone), *not* plant-level — no proprietary
  SCADA needed, and it's the quantity markets actually settle on.
- **Europe-first (ENTSO-E)** for one decisive reason: ENTSO-E publishes each TSO's **own day-ahead
  wind/solar forecast**, giving a *real operational baseline to beat* — the single most credible thing
  we can benchmark against. (US via EIA-930 / ERCOT / CAISO is the fallback / expansion.)

**The fail-fast question that governs everything:** *Can a calibrated probabilistic model beat the
TSO's own day-ahead forecast — on CRPS AND on economic (imbalance) cost — for a real bidding zone?*
If a strong model can't, that's critical market intel (the incumbents are already good), learned in
weeks not months.

---

## What transfers vs. what gets parked

**Reuse:** the diffusion machinery (`wave_field/diffusion.py` — DDPM/v-pred/EMA/samplers, for generating
calibrated forecast *ensembles*), the metrics-module *pattern*, the RunPod/scripts pattern, and above all
the experimental discipline (matched baselines, honest eval, decision gates).

**Park (lineage, not deleted):** the wave/FFT operator (`wave_field/attention.py`), the NS generator and
field metrics (Phase-0/1 PDE work). Move under `legacy/` or leave in place but out of the critical path.
The wave/spectral operator only re-enters at the *stretch* stage (regional spatiotemporal fields), and
only if it earns its place — we use the best tool for the job, not a pet architecture.

**New:** energy data loaders, forecasting metrics (CRPS/pinball/reliability/economic), probabilistic
forecast models + baselines, and a public writeup.

---

## Data plan (100% public — verify licensing before any commercial use)

**Targets (what we predict) — ENTSO-E Transparency Platform** (free account + API token; `entsoe-py`):
- Actual aggregate generation per production type (Wind Onshore, Wind Offshore, Solar) per bidding zone.
- **The TSO day-ahead generation forecast** (wind/solar) — our headline baseline to beat.
- Day-ahead prices and **imbalance prices** per zone — for the economic metric.
- Load (optional feature/context).

**Weather features (inputs):**
- **ERA5 reanalysis** (Copernicus CDS, free; `cdsapi`) — hourly, historical — for model *development/training*.
- **Actual NWP forecasts** — NOAA **GFS** (AWS Open Data / NOMADS) and/or **ECMWF open data (IFS)** —
  for **honest evaluation**: a real day-ahead forecast uses forecast weather, not reanalysis.
- **HRRR** (NOAA, US, 3 km hourly, AWS Open Data) if/when we do the US expansion.
- `Open-Meteo` free API as a quick-start convenience (note: free tier is non-commercial — fine for the
  artifact, flag for any product).

**The rigor detail that makes or breaks credibility (the train/serve gap):** training on ERA5 (perfect
hindcast weather) then serving on GFS forecasts is leakage — the model never saw forecast error. So:
develop on ERA5 to prove weather→power skill, but **report headline numbers using real NWP forecasts
(GFS/ECMWF-open) as inputs**, and always compare to the TSO forecast on the same footing. Document this
explicitly in the writeup — it's exactly the honesty that signals competence.

**Physical modeling helpers:** `pvlib` (clear-sky / PV physics for solar features + a physics baseline);
NREL NSRDB / WIND Toolkit for US expansion.

**First slice:** one wind-heavy zone (e.g. Denmark DK1 or a German zone) for wind, and one solar-heavy
zone (e.g. Spain) for solar — enough signal, clean data, TSO forecast available.

---

## Metrics (what "good" means)

- **CRPS** (continuous ranked probability score) — the standard probabilistic-forecast score. Primary.
- **Pinball / quantile loss** at the quantiles a trader uses (e.g. P10/P50/P90).
- **Calibration/reliability:** PIT histograms, reliability diagrams, coverage of nominal intervals,
  and a scalar calibration error.
- **Sharpness** (interval width) — calibrated *and* sharp, not just wide.
- **Point-forecast sanity:** RMSE/MAE vs the TSO forecast (so we're comparable to how they self-report).
- **THE money metric:** translate forecast error → **imbalance cost** using ENTSO-E day-ahead vs
  imbalance prices (a simple market-settlement model). "€X/MW/year saved vs the TSO forecast" is the
  sentence a buyer reacts to. This is the differentiator of the whole artifact.

Land these in `metrics/forecasting.py` with self-tests (perfect forecast → CRPS 0; calibrated noise →
flat PIT), same discipline as the existing `metrics/field.py`.

---

## Baselines (the rigor foundation — build these FIRST)

1. **Persistence** (today→tomorrow; and "same hour yesterday").
2. **Climatology / diurnal-seasonal** (calibrated marginal by hour/month).
3. **TSO day-ahead forecast** (ENTSO-E) — *the* bar. Point forecast; we wrap it with an empirical error
   distribution to make it probabilistic for a fair CRPS comparison.
4. **Physical:** `pvlib` clear-sky → PV for solar; simple power-curve for wind (NWP wind speed → power).
5. **Gradient-boosted quantile regression** (LightGBM/XGBoost, quantile objective) — the *industry
   workhorse* ML baseline. If our fancy model can't beat well-tuned GBM quantile, we don't have a result.

Establishing that #3 and #5 are strong is the whole point — beating a strawman proves nothing.

---

## Modeling stages

**Stage A — Baselines + honest harness (the foundation).** All five baselines, the metrics, the
economic model, the train/serve-honest evaluation, one zone. Deliver a table: everyone vs the TSO
forecast on CRPS + imbalance cost.

**Stage B — Strong probabilistic ML (the core result).**
- Distributional/quantile neural net (multi-quantile head or a parametric/mixture density) on
  NWP + calendar + recent-generation features.
- **Conditional diffusion ensemble** (reuse `wave_field/diffusion.py`): condition on weather+context,
  sample an *ensemble* of generation trajectories → a calibrated joint predictive distribution over the
  24h horizon (captures temporal correlation, which quantile regression misses — this matters for the
  economic metric). This is where the repo's machinery genuinely helps.
- Gate: beat TSO forecast + GBM-quantile on CRPS *and* imbalance cost, with good calibration.

**Stage C — Differentiation stretch (only if A/B win).** Regional/spatiotemporal: forecast the *field*
of generation across many zones / a spatial grid with a spatiotemporal generative model — the one place
the parked **wave/FFT + diffusion** machinery is genuinely apt (spatial multiscale + calibrated
ensembles = "GenCast-lite for renewables"). Optional downscaling angle. This is the "novel method"
upside, not the credibility foundation.

---

## Repo / engineering structure

```
data/                     # gitignored — cached ENTSO-E / ERA5 / GFS pulls
datasets/
  entsoe.py               # generation actuals, TSO forecasts, prices, load
  weather.py              # ERA5 (cdsapi) + GFS/ECMWF-open loaders; feature alignment
  build_dataset.py        # join weather+generation → aligned (features, target) tables; CLI
metrics/
  forecasting.py          # CRPS, pinball, reliability/PIT, coverage, sharpness + self-tests
  economic.py             # imbalance-cost settlement model → €/MW
baselines/
  persistence.py, climatology.py, tso_forecast.py, physical.py, gbm_quantile.py
models/
  quantile_net.py         # distributional / multi-quantile NN
  diffusion_forecast.py   # conditional diffusion ensemble (reuses wave_field/diffusion.py)
train_forecast.py         # unified trainer (arch/baseline switch), matched protocol
evaluate.py               # one-command leaderboard: all models vs TSO on CRPS + € for a zone
reports/                  # the public writeup + figures (the artifact)
legacy/                   # parked PDE + wave-operator code (lineage)
```
Add deps: `entsoe-py`, `cdsapi`, `xarray`, `netCDF4`/`cfgrib`, `lightgbm`, `pvlib`, `pandas`, `scikit-learn`.
Eventually rename the repo (e.g. `calibrated-energy-forecasting`); not urgent.

---

## Phased steps with decision gates

### Phase 0 — Data spine + honest harness (~1 week)
- [ ] ENTSO-E account + token; `datasets/entsoe.py` pulling actual gen, TSO day-ahead forecast, prices,
      load for 1 wind zone + 1 solar zone, ~3 years history. Cache locally.
- [ ] `datasets/weather.py`: ERA5 pull (cdsapi) for the zone's bounding box; align to hourly generation.
- [ ] `datasets/build_dataset.py`: join → aligned feature/target tables with a clean **temporal**
      train/val/test split (no shuffling across time; forecast-origin discipline).
- [ ] `metrics/forecasting.py` + `metrics/economic.py` with self-tests.
- [ ] **Gate:** can we reproduce the TSO forecast's error against actuals from raw ENTSO-E? If our
      numbers don't line up with reality, the data pipeline is wrong — fix before modeling (rule #1).

### Phase 1 — Baselines + the fail-fast (~1 week)
- [ ] All five baselines; the leaderboard harness (`evaluate.py`).
- [ ] **THE fail-fast gate:** GBM-quantile vs the TSO forecast on CRPS **and** imbalance cost, using
      real NWP-forecast inputs (not ERA5) for headline numbers.
      - Beats TSO → real signal; proceed to Stage B with confidence.
      - Ties/loses → *important finding.* Diagnose (is it the weather input? the zone?); the incumbents
        being hard to beat is itself market intel worth writing up honestly. Decide before scaling.

### Phase 2 — Strong probabilistic model (~2 weeks)
- [ ] Quantile NN + conditional diffusion ensemble; matched training protocol.
- [ ] Full leaderboard: CRPS, pinball, calibration, sharpness, RMSE, **€ saved vs TSO** — per zone.
- [ ] Ablate what matters (weather source, features, temporal-correlation modeling).
- [ ] **Gate:** does the generative model's *joint/temporal* calibration buy real economic value over
      GBM-quantile? If not, the simpler model is the honest headline.

### Phase 3 — The artifact + go-to-market (parallel; ongoing)
- [ ] Public repo (clean, reproducible, one-command eval) + a sharp writeup: method, the train/serve
      honesty, the leaderboard, and **€/MW/year vs the TSO forecast**. This IS the credibility artifact.
- [ ] Publish (arXiv/blog/GitHub) and use it as the outreach hook.
- [ ] **Customer-discovery track (yours — the real bottleneck):** talk to 5–10 potential buyers (energy
      traders, small utilities, wind/solar operators, forecasting desks). Is the pain real? What do they
      pay now? What accuracy/calibration changes their decision? The artifact is the door-opener; the
      conversations decide if there's a business. *No model substitutes for this.*

### Phase 4 — Expand or productize (only if signal is real)
- [ ] Second region / US (EIA-930/ERCOT + HRRR) to show generality.
- [ ] Stage C spatiotemporal/regional generative model (the differentiation + method-paper upside).
- [ ] If buyer interest appears: a minimal live API/dashboard for one design partner.

---

## Honest risks & kill criteria

- **TSOs are already good.** Beating their operational forecast (esp. day-ahead wind) is genuinely hard.
  If we can't beat it economically after Phase 2, the *product* thesis weakens — but a rigorous "here's
  how close public-data models get, and where they don't" is still a strong artifact. Kill the *product*
  ambition early if the economics don't clear; keep the *credibility* artifact regardless.
- **Distribution, not tech, is the bottleneck.** The model can be great and still sell nothing. The
  customer-discovery track is the real gate on "revenue"; treat it as first-class.
- **Data licensing for commercial use.** ENTSO-E/ERA5/GFS are fine for a public artifact; a *product*
  needs a licensing review (esp. Open-Meteo non-commercial tier). Flag, don't ignore.
- **Don't force the wave operator.** It re-enters only at Stage C and only if it wins. Best tool for the job.

## Timeline (aggressive; all-in)

| Phase | Duration | Milestone |
|---|---|---|
| 0 — Data spine + metrics | ~1 wk | Reproduce TSO-forecast error from raw data; honest harness |
| 1 — Baselines + fail-fast | ~1 wk | GBM-quantile vs TSO on CRPS + € — go/no-go |
| 2 — Probabilistic model | ~2 wk | Diffusion/quantile beats TSO + GBM; calibrated; €-quantified |
| 3 — Artifact + discovery | parallel | Public repo + writeup; 5–10 buyer conversations |
| 4 — Expand / productize | open | 2nd region, Stage C, or a design-partner demo |
