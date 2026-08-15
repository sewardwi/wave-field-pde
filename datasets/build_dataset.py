"""
Assemble the modelling table for day-ahead probabilistic forecasting.

Joins weather features + calendar features + (leakage-safe) generation lags into
an aligned hourly table, with a strict **temporal** train/val/test split. The two
disciplines that keep the whole artifact honest live here:

  1. Forecast-origin / no-leakage: a day-ahead forecast issued at gate-closure for
     the whole next day may only use information available *at issue time*. So
     generation autoregressive features must be lagged by at least the forecast
     lead (>= 24h for day-ahead); weather is the NWP *forecast* valid at target
     time (provided upstream), never reanalysis of the target hour.
  2. Temporal split: never shuffle across time — train is strictly before val,
     val strictly before test, so we never train on the future.

The ENTSO-E / weather I/O lives in `datasets/entsoe.py` and `datasets/weather.py`;
this module is pure dataframe logic and is unit-tested (`--self-test`).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar features (hour-of-day, day-of-year) + raw parts."""
    idx = pd.DatetimeIndex(index)
    hod = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hod / 24),
        "hour_cos": np.cos(2 * np.pi * hod / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow": idx.dayofweek.astype(float),
    }, index=idx)


def add_lags(series: pd.Series, lags_hours: list[int], min_lead: int = 24,
             name: str | None = None) -> pd.DataFrame:
    """
    Lagged copies of `series`, refusing any lag shorter than `min_lead` (the
    forecast lead time) so no post-issue information leaks into a day-ahead model.

    Args:
        series:     hourly series (e.g. actual generation).
        lags_hours: lags to build, in hours.
        min_lead:   minimum admissible lag (day-ahead ⇒ 24). Smaller lags raise.
    """
    bad = [h for h in lags_hours if h < min_lead]
    if bad:
        raise ValueError(f"lags {bad} < min_lead {min_lead}h would leak future info "
                         f"into a day-ahead forecast")
    nm = name or (series.name or "y")
    return pd.DataFrame({f"{nm}_lag{h}h": series.shift(h) for h in lags_hours})


def forecast_origin_features(y: pd.Series, gate_hour: int = 12) -> pd.DataFrame:
    """
    Autoregressive features that respect **forecast origin**, i.e. gate closure.

    Why a plain `lag24h` is not safe. In a day-ahead market every hour of delivery
    day D is bid before gate closure on D-1 (12:00 CET for EPEX/OMIE). So the whole
    day shares one information cutoff. A fixed 24h lag silently violates that for
    later hours: for target 23:00 on D, `y_lag24h` is 23:00 on D-1 — eleven hours
    *after* the cutoff. That's future information, and it flatters exactly the
    late-day hours where a real system is least certain.

    The honest construction is features frozen at the cutoff and therefore constant
    across the delivery day:
      y_at_gate     — last observed generation before cutoff
      y_mean24_gate — mean generation over the 24h ending at cutoff
      y_std24_gate  — its variability (recent regime)
      y_sameh_d2    — same hour of day D-2, the most recent *fully observed* day

    `add_lags` remains for intraday work, where a short lag is legitimate.
    """
    y = y.sort_index()
    full = pd.date_range(y.index.min().floor("D"), y.index.max().ceil("D"),
                         freq="h", tz=y.index.tz)
    ys = y.reindex(full)

    roll_mean = ys.rolling("24h", min_periods=6).mean()
    roll_std = ys.rolling("24h", min_periods=6).std()

    days = pd.DatetimeIndex(sorted(set(y.index.floor("D"))))
    cutoff = days - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)

    def at_cutoff(series: pd.Series) -> np.ndarray:
        # Last value at or strictly before the cutoff instant.
        pos = series.index.searchsorted(cutoff, side="right") - 1
        vals = series.to_numpy()
        out = np.full(len(pos), np.nan)
        ok = pos >= 0
        out[ok] = vals[pos[ok]]
        return out

    # Installed-capacity proxy: the largest output actually observed in the
    # trailing 60 days, frozen at the cutoff like everything else here.
    #
    # This exists because fleets grow. Spanish solar added 64% of capacity over
    # 2024-03..2026-08 (monthly peak 17.9 -> 29.3 GW), which puts 6.7% of test
    # hours above anything in the training range — and a tree model cannot
    # predict above the largest target it was trained on, so absolute-MW models
    # break down structurally rather than merely losing accuracy. Dividing by
    # this proxy turns the target into a capacity factor, which is stationary.
    roll_peak = ys.rolling("60D", min_periods=48).max()

    daily = pd.DataFrame({
        "y_at_gate": at_cutoff(ys),
        "y_mean24_gate": at_cutoff(roll_mean),
        "y_std24_gate": at_cutoff(roll_std),
        "capacity_gate": at_cutoff(roll_peak),
    }, index=days)

    out = daily.reindex(y.index.floor("D"))
    out.index = y.index
    # Same hour two days back is always before the cutoff, for every hour of D.
    out["y_sameh_d2"] = ys.shift(48).reindex(y.index).to_numpy()
    return out


def assemble(target: pd.Series, weather: pd.DataFrame,
             tso_forecast: pd.Series | None = None,
             prices: pd.DataFrame | None = None,
             gen_lags_hours: list[int] | None = None,
             min_lead: int = 24,
             origin_features: bool = False,
             gate_hour: int = 12) -> pd.DataFrame:
    """
    Build the aligned modelling table. All inputs are hourly, tz-aware (UTC).

    Returns a dataframe indexed by target time with columns:
      y                 — target (e.g. actual generation, MW)
      <weather...>      — NWP-forecast weather features valid at target time
      <calendar...>     — cyclical calendar features
      y_lag{h}h         — leakage-safe generation lags (optional)
      tso_forecast      — the incumbent baseline aligned to target time (optional)
      da_price/imb_*    — prices for the economic metric (optional)
    Rows with any NaN (from lag warm-up / missing data) are dropped.
    """
    target = target.rename("y")
    parts = [target, weather, calendar_features(target.index)]
    if origin_features:
        parts.append(forecast_origin_features(target, gate_hour=gate_hour))
    if gen_lags_hours:
        parts.append(add_lags(target, gen_lags_hours, min_lead=min_lead, name="y"))
    if tso_forecast is not None:
        parts.append(tso_forecast.rename("tso_forecast"))
    if prices is not None:
        parts.append(prices)
    df = pd.concat(parts, axis=1)
    return df.dropna()


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15
                   ) -> dict[str, pd.DataFrame]:
    """
    Chronological split — train | val | test in time order, no shuffling.
    (Fractions are of the row count; boundaries fall on row order, which for an
    hourly index is time order.)
    """
    assert df.index.is_monotonic_increasing, "index must be time-sorted"
    n = len(df)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test
    assert n_train > 0, "not enough rows for the requested split"
    return {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train + n_val],
        "test": df.iloc[n_train + n_val:],
    }


def blocked_val_split(df: pd.DataFrame, val_frac: float = 0.15,
                      test_frac: float = 0.15, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Split with a *seasonally representative* validation set.

    `temporal_split` puts val in one contiguous block between train and test.
    That is the right shape for measuring generalisation, but it makes val a poor
    **calibration** set on a short history: on DE wind the val block landed on a
    high-wind winter (mean 20 GW) while test was a calm summer (12 GW), so a
    coverage curve fitted there over-widened the intervals and made CRPS worse.
    (Point-forecast early stopping barely noticed — calibration is much more
    sensitive to distribution shift than a conditional median is.)

    Here **test is still strictly the final contiguous period** — the guarantee
    that actually matters, since nothing from the test window informs fitting or
    calibration. Only train/val are interleaved, by whole **days** so no day is
    split across the two, and val is drawn across all seasons.
    """
    assert df.index.is_monotonic_increasing, "index must be time-sorted"
    n_test = int(round(test_frac * len(df)))
    head, test = df.iloc[:len(df) - n_test], df.iloc[len(df) - n_test:]

    days = pd.DatetimeIndex(sorted(set(head.index.floor("D"))))
    n_val_days = max(1, int(round(val_frac / (1 - test_frac) * len(days))))
    val_days = set(np.random.default_rng(seed).choice(days, size=n_val_days, replace=False))
    is_val = head.index.floor("D").isin(val_days)
    return {"train": head[~is_val], "val": head[is_val], "test": test}


def xy(df: pd.DataFrame, target_col: str = "y",
       drop: tuple[str, ...] = ("tso_forecast",)) -> tuple[pd.DataFrame, pd.Series]:
    """Split a table into feature matrix X and target y (excluding price/baseline cols)."""
    price_cols = [c for c in df.columns if c.startswith(("da_", "imb_", "price"))]
    feat_cols = [c for c in df.columns if c != target_col and c not in drop and c not in price_cols]
    return df[feat_cols], df[target_col]


# ---------------------------------------------------------------------------
# Self-test (pure logic — no credentials needed)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    idx = pd.date_range("2021-01-01", periods=24 * 400, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    gen = pd.Series(np.abs(np.sin(np.arange(len(idx)) * 2 * np.pi / 24) * 100)
                    + rng.normal(0, 5, len(idx)), index=idx, name="gen")
    weather = pd.DataFrame({"wind100": rng.uniform(0, 25, len(idx)),
                            "ssrd": np.clip(rng.normal(200, 100, len(idx)), 0, None)}, index=idx)
    tso = gen + rng.normal(0, 8, len(idx))

    # Calendar features are bounded and cyclical.
    cal = calendar_features(idx)
    assert np.allclose(cal["hour_sin"] ** 2 + cal["hour_cos"] ** 2, 1.0)
    print(f"calendar features: {list(cal.columns)}  (unit-circle ✓)")

    # Leakage guard: a <24h lag must be refused for a day-ahead model.
    try:
        add_lags(gen, [1], min_lead=24)
        raise AssertionError("expected leakage guard to fire")
    except ValueError:
        print("leakage guard: 1h lag correctly refused for day-ahead ✓")

    # Assembly + admissible lags.
    df = assemble(gen, weather, tso_forecast=tso, gen_lags_hours=[24, 48], min_lead=24)
    assert "y_lag24h" in df.columns and "tso_forecast" in df.columns
    # lag24 really is the target 24h earlier.
    common = df.index[100]
    assert abs(df.loc[common, "y_lag24h"] - gen.loc[common - pd.Timedelta("24h")]) < 1e-9
    print(f"assembled table: {df.shape[0]} rows, cols={list(df.columns)}")

    # Forecast-origin features must not react to anything after gate closure.
    # Spike one delivery day's post-cutoff hours and assert the features for that
    # day are unchanged — while a naive lag24 *does* move (proving the test bites).
    gate = 12
    spike_day = pd.Timestamp("2021-06-10", tz="UTC")
    cut = spike_day - pd.Timedelta(days=1) + pd.Timedelta(hours=gate)
    poisoned = gen.copy()
    poisoned[(poisoned.index > cut) & (poisoned.index < spike_day + pd.Timedelta(days=1))] += 1e4

    base = forecast_origin_features(gen, gate_hour=gate)
    after = forecast_origin_features(poisoned, gate_hour=gate)
    day = slice(spike_day, spike_day + pd.Timedelta(hours=23))
    assert np.allclose(base.loc[day].to_numpy(), after.loc[day].to_numpy(), equal_nan=True), \
        "forecast-origin features leaked post-gate information"
    naive = add_lags(gen, [24], min_lead=24, name="y")["y_lag24h"]
    naive_p = add_lags(poisoned, [24], min_lead=24, name="y")["y_lag24h"]
    n_moved = int((~np.isclose(naive.loc[day].to_numpy(), naive_p.loc[day].to_numpy(),
                               equal_nan=True)).sum())
    assert n_moved > 0, "control failed: naive lag24 should have leaked"
    print(f"forecast-origin features: immune to a post-gate spike ✓  "
          f"(naive lag24h leaked on {n_moved}/24 hours of that day)")

    # Temporal split is ordered and non-overlapping.
    sp = temporal_split(df)
    assert sp["train"].index.max() < sp["val"].index.min() < sp["val"].index.max() < sp["test"].index.min()
    print(f"temporal split: train {len(sp['train'])} < val {len(sp['val'])} < test {len(sp['test'])}  "
          f"(train ends {sp['train'].index.max().date()}, test starts {sp['test'].index.min().date()})")

    X, y = xy(sp["train"])
    assert "tso_forecast" not in X.columns and "y" not in X.columns
    print(f"X/y split: {X.shape[1]} features, target 'y'  (baseline/prices excluded from X)")

    print("\ndatasets/build_dataset.py self-test PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        _self_test()
    else:
        print("pass --self-test")
