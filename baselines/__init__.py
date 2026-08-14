"""
baselines — the references any claimed result must beat.

Rule #1 (from prior overclaims): a comparison is only meaningful against a real,
matched baseline whose published number we have reproduced on our own data.

Current direction — probabilistic energy forecasting. All expose the same
`fit(train)` / `predict_quantiles(df, levels)` interface (see `base.py`):
  persistence.py   — the floor (same hour, last fully observed day)
  climatology.py   — calibrated no-skill reference, uses no weather
  tso_forecast.py  — THE bar: the grid operator's own day-ahead forecast
  physical.py      — power curve / irradiance physics with one fitted scale
  gbm_quantile.py  — LightGBM pinball regression, the industry workhorse

Parked lineage: `fno.py` (FNO-2D, Li et al. 2020) from the PDE direction.
"""
