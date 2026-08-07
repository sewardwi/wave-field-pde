"""
ENTSO-E Transparency Platform loader — the forecasting *targets* and the baseline.

Wraps `entsoe-py`'s pandas client to pull, per European bidding zone:
  - actual aggregate generation by type (wind onshore/offshore, solar)  → the target
  - the TSO's own day-ahead wind/solar forecast                         → the baseline to beat
  - day-ahead prices + imbalance prices                                 → the economic metric
  - load (optional context feature)

Results are cached locally (data/entsoe/) so we hit the API once. Requires a free
ENTSO-E API token (register at transparency.entsoe.eu, then request API access;
historically you email transparency@entsoe.eu to enable it). Set it in the
environment as ENTSOE_TOKEN.

This module does network I/O and needs the token, so it is not exercised by a
self-test; `python -m datasets.entsoe --check` validates config without pulling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# ENTSO-E PSR (production-source) codes.
PSR = {"solar": "B16", "wind_offshore": "B18", "wind_onshore": "B19"}

CACHE = Path("data/entsoe")


def _cache_path(zone: str, what: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{zone}_{what}_{start.date()}_{end.date()}"
    return CACHE / f"{tag}.parquet"


def _read_cache(p: Path):
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_pickle(p.with_suffix(".pkl")) if p.with_suffix(".pkl").exists() else None


def _write_cache(obj, p: Path) -> None:
    try:
        (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_parquet(p)
    except Exception:
        obj.to_pickle(p.with_suffix(".pkl"))     # fallback if no parquet engine


class ENTSOE:
    """Thin cached wrapper over entsoe.EntsoePandasClient for one workflow."""

    def __init__(self, token: str | None = None, tz: str = "Europe/Brussels"):
        self.token = token or os.environ.get("ENTSOE_TOKEN")
        if not self.token:
            raise RuntimeError("no ENTSO-E token — set ENTSOE_TOKEN (see module docstring)")
        from entsoe import EntsoePandasClient          # imported lazily
        self.client = EntsoePandasClient(api_key=self.token)
        self.tz = tz

    def _ts(self, d) -> pd.Timestamp:
        t = pd.Timestamp(d)
        return t.tz_localize(self.tz) if t.tzinfo is None else t.tz_convert(self.tz)

    def _cached(self, what, fn, zone, start, end):
        s, e = self._ts(start), self._ts(end)
        p = _cache_path(zone, what, s, e)
        hit = _read_cache(p)
        if hit is not None:
            return hit.squeeze("columns") if hit.shape[1] == 1 else hit
        print(f"  ENTSO-E pull: {what} {zone} {s.date()}→{e.date()}", flush=True)
        obj = fn(zone, start=s, end=e)
        _write_cache(obj, p)
        return obj

    # ---- individual series -------------------------------------------------
    def generation_actual(self, zone, start, end, kind="wind_onshore") -> pd.Series:
        """Actual generation (MW, hourly-resampled) for one production type."""
        df = self._cached(f"gen_{kind}", lambda z, start, end: self.client.query_generation(
            z, start=start, end=end, psr_type=PSR[kind]), zone, start, end)
        s = df.squeeze() if hasattr(df, "squeeze") else df
        if isinstance(s, pd.DataFrame):                # (Actual Aggregated / Consumption) cols
            s = s.filter(like="Actual").iloc[:, 0]
        return s.resample("h").mean().tz_convert("UTC").rename(f"gen_{kind}")

    def tso_forecast(self, zone, start, end, kind="wind_onshore") -> pd.Series:
        """The TSO's own day-ahead forecast (MW) — the baseline to beat."""
        df = self._cached(f"fc_{kind}", lambda z, start, end: self.client.query_wind_and_solar_forecast(
            z, start=start, end=end, psr_type=PSR[kind]), zone, start, end)
        s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        return s.resample("h").mean().tz_convert("UTC").rename(f"tso_forecast_{kind}")

    def day_ahead_prices(self, zone, start, end) -> pd.Series:
        s = self._cached("da_price", lambda z, start, end: self.client.query_day_ahead_prices(
            z, start=start, end=end), zone, start, end)
        return s.resample("h").mean().tz_convert("UTC").rename("da_price")

    def imbalance_prices(self, zone, start, end) -> pd.DataFrame:
        df = self._cached("imb_price", lambda z, start, end: self.client.query_imbalance_prices(
            z, start=start, end=end), zone, start, end)
        return df.resample("h").mean().tz_convert("UTC")

    def load(self, zone, start, end) -> pd.Series:
        s = self._cached("load", lambda z, start, end: self.client.query_load(
            z, start=start, end=end), zone, start, end)
        s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        return s.resample("h").mean().tz_convert("UTC").rename("load")

    # ---- one call for a zone ----------------------------------------------
    def bundle(self, zone, start, end, kind="wind_onshore") -> dict:
        """Pull everything needed for one zone+type into a dict of aligned series."""
        return {
            "target": self.generation_actual(zone, start, end, kind),
            "tso_forecast": self.tso_forecast(zone, start, end, kind),
            "da_price": self.day_ahead_prices(zone, start, end),
            "imbalance": self.imbalance_prices(zone, start, end),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate config without pulling")
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2022-01-08")
    ap.add_argument("--kind", default="wind_onshore", choices=list(PSR))
    args = ap.parse_args()

    if args.check:
        tok = os.environ.get("ENTSOE_TOKEN")
        print(f"ENTSOE_TOKEN set: {bool(tok)}")
        print(f"would pull: zone={args.zone} kind={args.kind} {args.start}→{args.end}")
        print("PSR codes:", PSR)
        print("run without --check (and with a token) to actually pull + cache.")
    else:
        e = ENTSOE()
        b = e.bundle(args.zone, args.start, args.end, args.kind)
        for k, v in b.items():
            print(f"  {k}: {type(v).__name__} shape={getattr(v, 'shape', None)}")
