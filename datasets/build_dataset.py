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


def assemble(target: pd.Series, weather: pd.DataFrame,
             tso_forecast: pd.Series | None = None,
             prices: pd.DataFrame | None = None,
             gen_lags_hours: list[int] | None = None,
             min_lead: int = 24) -> pd.DataFrame:
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
