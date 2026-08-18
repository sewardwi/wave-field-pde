# Findings & Assessment — Calibrated Probabilistic Energy Forecasting

**Written 2026-08-17, after Phase 0 and Phase 1 completed.** Read this before
`docs/final-plan.md`: the plan is still a good description of *intent*, but Phase 1
produced evidence that changes several of its assumptions. Where they disagree, this
document is newer.

Relevant commits: `697df0f` (Phase 0), `3268a27` `cab853d` `39b88f3` `98d29a7`
(Phase 1), `7bfba93` (the generalization test).

---

## TL;DR

The fail-fast question — *can a calibrated model on public data beat a TSO's own
day-ahead forecast?* — is **answered, and the answer is conditional**:

- **Replacing** the incumbent with public weather alone works only where the
  incumbent is weak. It does **not** generalize.
- **Post-processing** the incumbent (using their forecast as a feature) generalizes
  across both solar slices, and fails on wind.
- **Nobody produces calibrated day-ahead solar uncertainty — including the TSOs.**
  That is the one clearly open lane found, and it is unsolved here too.

Strategic read: **this is worth finishing as a credibility artifact and portfolio
piece, not as a product.** The technical findings argue against the product thesis
fairly directly (no moat, small conditional edge, relationship-driven market,
non-commercial data licence). They argue *for* the artifact thesis, which is roughly
80% done and needs writing, not more modelling.

---

## What was built

```
phase0_gate.py            data-spine gate + per-zone scope check (doubles as a
                          zone-screening tool — see "Strategy" below)
evaluate.py               the leaderboard: every model vs the TSO, one command
datasets/
  energy_charts.py        token-free ENTSO-E-derived actuals, TSO forecast, prices
  entsoe.py               ENTSO-E client (needs a token; only imbalance prices
                          actually require it)
  weather.py              Open-Meteo: ERA5 / best-run / genuine forecast-lead /
                          live, + fleet-averaged zone aggregation
  build_dataset.py        calendar + forecast-origin features, splits, leakage guards
  slices.py               slice name -> modelling-ready table
  _cache.py               shared on-disk cache
baselines/
  base.py                 shared interface, empirical-residual wrapper,
                          ConformalRecalibrator, CapacityNormalized
  persistence.py climatology.py physical.py tso_forecast.py gbm_quantile.py
metrics/
  forecasting.py          CRPS / pinball / PIT / coverage / calibration (self-tested)
  economic.py             imbalance settlement, EUR/MW/yr (self-tested)
reports/                  committed result JSON for every run below
```

### Reproducing

```bash
python phase0_gate.py --all --years 3
python evaluate.py --slice de_wind  --lead-days 2 --val-split blocked --end 2026-08-08
python evaluate.py --slice es_solar --lead-days 2 --val-split blocked --end 2026-08-08
python evaluate.py --slice de_solar --lead-days 2 --val-split blocked --end 2026-08-08

python -m metrics.forecasting --self-test
python -m metrics.economic --self-test
python -m datasets.build_dataset --self-test      # includes the leakage tests
```

Weather pulls are chunked and cached per chunk, so an interrupted run resumes.
A cold multi-year, multi-point pull takes 20–30 minutes and Open-Meteo will
intermittently time out; the retry logic handles it.

---

## Phase 0 — the data spine

`datasets/energy_charts.py` uses **Fraunhofer ISE's Energy-Charts API** (no key,
CC BY 4.0), which republishes ENTSO-E data *including each TSO's own day-ahead
forecast*. The ENTSO-E token was assumed to be a blocker for Phase 0; it is not.
It is only needed for **imbalance prices** (and for true bidding-zone granularity —
the power endpoints are country-level, though ES is a single zone so ES is clean).

Gate result, 3 years, TSO day-ahead vs actuals, nRMSE as % of observed peak:

| slice | nRMSE | bias | scope slope | verdict |
|---|---|---|---|---|
| DE solar | 2.29% | +3.8% | 0.968 | PASS |
| DE wind | 3.66% | +0.6% | 0.970 | PASS |
| ES solar | 4.53% | +0.2% | 0.992 | PASS |
| DK wind | 10.18% | −12.3% | **1.256** | **FAIL — scope** |
| FR wind | 5.71% | +6.5% | **0.909** | **FAIL — scope** |

ES solar landing inside the published 3–5% band for day-ahead solar is the evidence
the pipeline is correct.

---

## Phase 1 — the leaderboards

All: genuine 48h NWP lead, 2024-03 → 2026-08, blocked val split, test is the final
contiguous ~15%. CRPS in MW, lower is better. `skill` = 1 − CRPS/CRPS_tso.

### DE wind (mean generation 11,959 MW)

| model | CRPS | skill | RMSE | cov80 | calib |
|---|---|---|---|---|---|
| **tso** | **993.5** | — | 2205.9 | 77.0% | 0.010 |
| tso_debias | 993.5 | +0.0% | 2205.6 | 76.8% | 0.011 |
| gbm_plus_tso | 1022.7 | −2.9% | **2158.7** | 65.6% | 0.058 |
| gbm_plus_tso_cf | 1026.2 | −3.3% | 2188.1 | 67.4% | 0.050 |
| gbm_quantile | 1830.9 | −84.3% | 3550.9 | 69.1% | 0.067 |
| physical | 1888.1 | −90.0% | 3865.0 | 81.5% | 0.042 |
| climatology | 4758.3 | −378.9% | 8820.8 | 73.6% | 0.030 |
| persistence | 4860.5 | −389.2% | 8771.7 | 81.7% | 0.059 |

Verdict: **not beaten.** Stable across two split designs and with/without capacity
normalisation. Note the TSO baseline's calibration (0.010) is excellent — wrapping a
good point forecast in level-conditional empirical residuals is a genuinely strong
probabilistic model, and it is hard to beat.

### ES solar (daylight hours only, mean 16,867 MW)

| model | CRPS | skill | RMSE | cov80 | calib |
|---|---|---|---|---|---|
| **gbm_plus_tso_cf** | **1213.3** | **+20.0%** | **2005.1** | 47.9% | 0.133 |
| gbm_quantile_cf | 1248.0 | **+17.7%** | 2130.8 | 53.6% | 0.137 |
| gbm_plus_tso_cal | 1450.7 | +4.3% | 2360.0 | 52.2% | 0.197 |
| gbm_plus_tso | 1468.3 | +3.2% | 2352.8 | 40.4% | 0.217 |
| tso | 1516.2 | — | 2726.7 | 64.4% | 0.116 |
| gbm_quantile | 1861.5 | −22.8% | 2880.9 | 38.4% | 0.292 |
| persistence | 1877.0 | −23.8% | 3343.5 | 69.4% | 0.086 |

Verdict: **beaten, including on public weather alone (+17.7%).**

### DE solar (daylight hours only, mean 26,079 MW) — the generalization test

| model | CRPS | skill | RMSE | cov80 | calib |
|---|---|---|---|---|---|
| **gbm_plus_tso_cf** | **1273.7** | **+7.4%** | **2348.5** | 58.2% | 0.208 |
| tso | 1376.1 | — | 2552.2 | 67.8% | 0.218 |
| tso_debias | 1424.1 | −3.5% | 2640.5 | 66.8% | 0.230 |
| gbm_quantile_cf | 1561.4 | **−13.5%** | 2904.0 | 66.1% | 0.096 |
| gbm_plus_tso | 2043.9 | −48.5% | 3503.2 | 37.5% | 0.328 |
| physical | 5239.5 | −280.7% | 9055.2 | 53.9% | 0.207 |

Verdict: **post-processing wins, replacement loses.**

### The pattern

Ordered by incumbent strength:

| slice | incumbent nRMSE | replace | post-process |
|---|---|---|---|
| DE solar | 2.29% (strongest) | −13.5% ✗ | **+7.4% ✓** |
| DE wind | 3.66% | −84% ✗ | −2.9% ✗ |
| ES solar | 4.53% (weakest) | **+17.7% ✓** | **+20.0% ✓** |

Skill tracks incumbent weakness. The ES "+17.7% on public weather" result is
**Spain-specific and must not be stated as a general claim** — DE solar was run
specifically to try to break it, and did.

### The unsolved problem

Calibration, everywhere on solar, **including the incumbents**: TSO coverage is
64.4% (ES) and 67.8% (DE) against a nominal 80%, with calibration errors of 0.116
and 0.218. Our winners are no better (47.9%, 58.2%). Every win above is a
*point-accuracy* win. The uncertainty quality the project set out to sell is
unsolved by anyone in these slices — the clearest open lane found, and the right
target for any future modelling work.

---

## The traps (the most reusable output)

These cost real debugging time and each would have corrupted a headline number.
Anyone resuming this, or doing similar work elsewhere, should read this section
even if they skip the rest.

1. **The zone-scope trap.** Some zones publish "actual generation" and a
   "day-ahead forecast" covering *different fleets*, showing up as
   `actual ≈ slope · forecast` with slope far from 1. DK is 1.256, FR is 0.909 —
   opposite directions. On such a zone you can "beat the TSO" by >10% **purely by
   rescaling their forecast**. `phase0_gate.py` auto-fails any zone with
   |slope−1| > 0.05. Always run it before adopting a new zone.

2. **Capacity extrapolation.** A gradient-boosted tree predicts by averaging
   training targets in a leaf, so it can *never* output a value above the largest
   target it saw. Spanish solar added 64% of capacity across the window
   (monthly peak 17.9 → 29.3 GW), putting **6.7% of test hours above the entire
   training range**. Fix: fit on capacity factor (`CapacityNormalized` +
   leakage-safe `capacity_gate`). On DE solar this moved a model from −48.5% to
   +7.4%; it is **inert on wind**, where 0% of test rows exceeded the train range —
   which is the control showing it addresses the real mechanism rather than acting
   as a generic accuracy hack. **Always check `(test.y > train.y.max()).mean()`.**

3. **Forecast-lead leakage.** Open-Meteo's `previous_dayN` is a *constant* N×24h
   lead. A real day-ahead forecast is not: every hour of delivery day D is bid at
   gate closure on D−1, so leads run 24h (hour 00) to 47h (hour 23). `lead_days=1`
   therefore **leaks** — for hour 23 it is the run issued 12h *after* gate closure.
   Use `lead_days=2` for headline numbers (48h > 47h always) and treat lead-1 as an
   optimistic bracket.

4. **`lag24h` leakage.** Same root cause on the generation side: a fixed 24h lag
   leaks for 11 of 24 hours when gate closure is 11:00 UTC. Replaced with
   gate-frozen forecast-origin features. `build_dataset.py --self-test` poisons
   post-gate data and asserts the features don't move (and that a naive lag24 does).

5. **Calibration is far more shift-sensitive than point forecasts.** A contiguous
   validation block landed on a high-wind winter (mean 20 GW) while test was a calm
   summer (12 GW). Recalibrating on it *worsened* CRPS by over-widening intervals
   (coverage overshot to 85.6%, sharpness ×2.8). Early stopping on the same split
   barely noticed. Use `blocked_val_split` (test still strictly last; train/val
   interleaved by whole days) whenever fitting calibration.

6. **CRPS rewards sharpness as well as calibration.** On DE wind, correctly
   calibrating an over-confident model made CRPS *worse* (−3.1% → −3.8%): the
   over-confident version's narrow intervals plus an accurate median bought back
   more than its miscalibration cost. Do not assume better calibration ⇒ better CRPS.

7. **Fleet averaging has two subtleties.** Wind direction is circular (a naive mean
   of 350° and 10° gives 180°, pointing the wrong way) — average sin/cos. And wind
   power goes as v³, so the fleet aggregate is the **mean of cubes, not the cube of
   the mean**; the gap was 1.35× on real data.

8. **Spatial resolution mattered more than the model.** A 7-point fleet average left
   the GBM ~1.7× worse than the TSO on RMSE while the same GBM *given the TSO
   forecast as a feature* matched it — diagnosing that as feature poverty rather
   than model failure is what led to 27 points.

9. **Engineering gotchas.** Cache keys must include the point set (or editing
   geometry silently reuses stale weather). Retry **network exceptions**, not just
   HTTP status codes — a bare `ReadTimeout` propagates out of `requests.get()` and
   killed a pull that was 40% done. Don't pipe a background run through `tail`; it
   buffers until the stream ends and you see nothing.

10. **Metric hygiene.** The 13-level quantile-integral CRPS is 0.97% low in absolute
    terms versus the closed-form Gaussian, but *identically* so across predictive
    spreads, so rankings and skill ratios are unaffected. Don't compare these CRPS
    values against CRPS computed elsewhere at a different level density.

---

## Strategic assessment

### Why the product thesis is weak

- **No moat.** The result came from public data plus careful engineering. Replicable
  in a few weeks by anyone competent.
- **The edge is small and conditional.** +7.4% CRPS is not a business, and +20% on
  ES solar reflects *that incumbent's weakness* rather than our capability.
- **The market is relationship-driven and occupied.** Meteologica, Enfor, Vaisala,
  DNV, Solargis already sell probabilistic renewable forecasts with track records
  and procurement relationships. The original diagnosis — distribution is the
  bottleneck, not tech — is exactly what this market punishes, and no model result
  changes it.
- **Licensing erases the cost advantage.** Open-Meteo's free tier is
  non-commercial; a product needs paid NWP.

### Why the artifact thesis is strong

What accumulated is more valuable than the model:

- Four self-caught leakage/correctness bugs, each with a test that bites
- A **falsification test that killed its own headline claim** (DE solar was chosen
  because it was the most likely to break the ES result)
- A mechanism — incumbent strength — that explains both the wins and the losses
- Paired negative and positive results, reported plainly

Most portfolio projects show a win. Almost none show someone dismantling their own
win on purpose. That contrast is the asset, and it is what funded players
(Jua, Silurian, PhysicsX, an energy desk) are trying to determine about a candidate
and usually cannot.

`phase0_gate.py` is also quietly a **business/screening tool**: it measures any
European bidding zone's incumbent forecast quality from public data. "Which of 30+
zones have an exploitable forecast gap" is a more defensible capability than any
single model.

### If resumed, in this order

1. **Write it up.** Highest value, lowest risk. Frame as benchmark + methodology:
   *where European TSO day-ahead forecasts can be beaten, where they can't, and the
   traps in measuring it*, with the traps above as the contributions.
   Calibrate venue expectations honestly: three slices and one test window is an
   arXiv preprint plus a strong blog post, or a Climate Change AI / ICLR workshop
   paper. It is not a top-tier conference paper.
2. **Robustness check before publishing anything.** Every number rests on a single
   ~4-month test window per slice. Three slices, one window each, with a headline
   that flipped between them, invites the cherry-picking objection. Re-run ES and DE
   solar over two or three different test windows. If +7.4% holds, the paper is
   solid; if it moves, you need to know first.
3. **One bounded calibration attempt.** Nobody in these slices produces calibrated
   day-ahead solar uncertainty, TSOs included. If a cheap fix closes it, that is the
   paper's centrepiece and a genuinely novel claim. One day of work, not a project.
4. **Only then consider Phase 2** (conditional diffusion ensemble), scoped to
   **solar only, calibration as the objective, post-processing as the framing**.
   Its advantage — calibrated *joint* samples across the 24h horizon — is real and
   also feeds the € metric, since imbalance cost depends on the whole day's error
   pattern rather than hourly marginals. Do not escalate to it on a maybe.

### What to drop

- **Wind, from the headline entirely.** German day-ahead wind is effectively solved
  and cannot be out-resourced on public data.
- **The € metric as a sales instrument.** Keep it as a rigor exercise if the token
  is easy; it will not close a sale by itself.
- **The solo-product ambition via this wedge.**

The ladder in `final-plan.md` — artifact → industry → network + capital → found from
strength — still holds. Phase 1 validated the ladder and invalidated the shortcut,
in about two weeks rather than a year. That is the plan working as designed.

---

## Open items and blockers

| item | state |
|---|---|
| € / imbalance-cost metric | **Blocked.** Energy-Charts has no imbalance prices. Needs ENTSO-E, or for Spain **ESIOS** (the win is Spanish, so ESIOS is the relevant one). Currently a proxy spread, clearly marked as indicative. |
| Multi-window robustness | Not done. Do before publishing. |
| Calibration fix | Not done. The one bounded bet worth taking. |
| lead-1 optimistic bracket | Not run. Cheap; brackets the honest range. |
| Bidding-zone granularity | Energy-Charts power endpoints are country-level. ES is a single zone so ES is exact; DE is the DE-LU zone. Finer granularity needs ENTSO-E. |
| Data licensing | Open-Meteo free tier is **non-commercial**. Fine for a public artifact; blocks a product. |
| Weather archive floor | Open-Meteo previous-runs starts **2024-03**, which bounds the honest evaluation window. |
| Fleet-point weighting | Zone points are equally weighted; no per-point installed capacity. A known, documented refinement. |
