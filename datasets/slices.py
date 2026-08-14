"""
Slice assembly — one call from a slice name to a modelling-ready table.

A "slice" is a (country, technology) pair: `de_wind`, `es_solar`. This module is
the only place that knows how the pieces fit together:

    energy_charts (target + TSO forecast + price)
  + weather.zone_weather (fleet-averaged NWP at a genuine D-1 lead)
  + build_dataset (calendar + forecast-origin features, temporal split)
  = a table ready for baselines/ and models/.

Only slices that passed phase0_gate.py belong here as headline candidates: DE wind,
ES solar, DE solar. DK/FR are excluded because their published actuals and forecasts
cover different fleets (see phase0_gate.py).

**Solar and the night-hours trap.** Roughly half of a solar year is dark, and every
model — including a constant zero — is exactly right then. Scoring over all hours
therefore mostly measures how much night is in the sample and compresses the real
differences between forecasts. `daylight_mask()` uses pvlib solar geometry, which
is deterministic and known months ahead, so masking on it leaks nothing. Solar
slices are evaluated daylight-only by default and all-hours as a secondary number.

    python -m datasets.slices --slice es_solar --days 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from datasets import build_dataset as bd
from datasets import energy_charts as ec
from datasets import weather as W

SLICES: dict[str, dict] = {
    "de_wind": dict(country="de", kind="wind", bzn="DE-LU", weather="de_wind",
                    label="DE wind", solar=False),
    "es_solar": dict(country="es", kind="solar", bzn="ES", weather="es_solar",
                     label="ES solar", solar=True),
    "de_solar": dict(country="de", kind="solar", bzn="DE-LU", weather="de_solar",
                     label="DE solar", solar=True),
}

# Day-ahead gate closure: 12:00 CET for EPEX/OMIE ⇒ 11:00 UTC in winter, 10:00 in
# summer. We use 11:00 UTC year-round — off by an hour for half the year, and in
# the conservative direction (a slightly earlier cutoff never invents information).
GATE_HOUR_UTC = 11


def daylight_mask(index: pd.DatetimeIndex, slice_key: str,
                  min_elevation_deg: float = 5.0) -> pd.Series:
    """True where the sun is meaningfully up, averaged over the slice's points.

    Purely astronomical, so it is knowable at any lead — no leakage.
    """
    from pvlib.solarposition import get_solarposition

    pts = W.ZONE_POINTS[SLICES[slice_key]["weather"]]
    elev = np.zeros(len(index))
    for lat, lon in pts:
        elev += get_solarposition(index, lat, lon)["apparent_elevation"].to_numpy()
    return pd.Series(elev / len(pts) > min_elevation_deg, index=index, name="daylight")


def build(slice_key: str, start: str | None = None, end: str | None = None,
          source: str = "day_ahead", gate_hour: int = GATE_HOUR_UTC,
          lead_days: int = 2) -> pd.DataFrame:
    """Assemble the modelling table for one slice.

    `source`: 'day_ahead' (genuine D-1 NWP lead — honest, the default), 'hist_fc'
    (best-available run, optimistic), or 'archive' (ERA5, leaky; development only).
    The honest window is bounded below by the previous-runs archive (2024-03).
    """
    if slice_key not in SLICES:
        raise KeyError(f"unknown slice '{slice_key}'; known: {sorted(SLICES)}")
    spec = SLICES[slice_key]

    end = end or (pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    if start is None:
        start = W.PREV_RUNS_START if source == "day_ahead" else "2021-01-01"
    if source == "day_ahead" and pd.Timestamp(start) < pd.Timestamp(W.PREV_RUNS_START):
        raise ValueError(f"previous-runs archive starts {W.PREV_RUNS_START}; "
                         f"got start={start}. Use source='hist_fc' for earlier data "
                         f"(and label it as optimistic).")

    if spec["kind"] == "wind":
        target = ec.wind_actual(spec["country"], start, end)
        tso = ec.wind_forecast(spec["country"], start, end)
    else:
        target = ec.generation_actual(spec["country"], start, end, spec["kind"])
        tso = ec.tso_forecast(spec["country"], start, end, spec["kind"])

    wx = W.zone_weather(spec["weather"], start, end, source=source, lead_days=lead_days)

    prices = None
    try:
        prices = ec.day_ahead_prices(spec["bzn"], start, end).to_frame()
    except Exception as e:                        # price is optional for CRPS
        print(f"  (no day-ahead price for {spec['bzn']}: {type(e).__name__})")

    df = bd.assemble(target, wx, tso_forecast=tso, prices=prices,
                     origin_features=True, gate_hour=gate_hour)
    df["daylight"] = daylight_mask(df.index, slice_key).to_numpy()
    df.attrs["slice"] = slice_key
    df.attrs["label"] = spec["label"]
    df.attrs["solar"] = spec["solar"]
    df.attrs["source"] = source
    df.attrs["lead_days"] = lead_days if source == "day_ahead" else 0
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except target, baseline, prices, and the mask."""
    drop = {"y", "tso_forecast", "daylight"}
    return [c for c in df.columns
            if c not in drop and not c.startswith(("da_", "imb_", "price"))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="es_solar", choices=sorted(SLICES))
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--source", default="day_ahead",
                    choices=["day_ahead", "hist_fc", "archive"])
    args = ap.parse_args()

    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=2)
    start = end - pd.Timedelta(days=args.days)
    df = build(args.slice, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
               source=args.source)

    print(f"\n{df.attrs['label']} ({args.source}): {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  {df.index[0]} → {df.index[-1]}")
    print(f"  features: {feature_columns(df)}")
    print(f"  daylight hours: {int(df.daylight.sum())} / {len(df)} ({df.daylight.mean():.0%})")
    err = df.tso_forecast - df.y
    print(f"  TSO all-hours   MAE={err.abs().mean():7.1f} MW  bias={err.mean():+7.1f}")
    if df.attrs["solar"]:
        d = df[df.daylight]
        e2 = d.tso_forecast - d.y
        print(f"  TSO daylight    MAE={e2.abs().mean():7.1f} MW  bias={e2.mean():+7.1f}"
              f"   (mean gen {d.y.mean():.0f} MW)")
