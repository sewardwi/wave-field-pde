"""
Energy-Charts loader — the forecasting *targets* and the TSO baseline, token-free.

Fraunhofer ISE's Energy-Charts API (https://api.energy-charts.info, CC BY 4.0,
no API key) republishes ENTSO-E Transparency data, including the one series that
makes this project credible: **the TSO's own day-ahead generation forecast**.
That makes it a drop-in substitute for datasets/entsoe.py for all of Phase 0,
without waiting on an ENTSO-E API token.

Endpoints used:
  /public_power           actual generation by production type + load  → the target
  /public_power_forecast  TSO day-ahead / intraday forecast            → the baseline to beat
  /price                  day-ahead price                              → the economic metric

What it does NOT have, and what still needs an ENTSO-E token:
  - **imbalance prices** — so the full €-settlement metric (metrics/economic.py
    dual-price path) can't be run from here; day-ahead price only.
  - **bidding-zone granularity for generation** — /price takes `bzn`, but the
    power endpoints are country-level only. Spain is a single bidding zone
    (country == zone, so ES is the clean solar slice); Denmark is DK1+DK2.

Two API quirks this module absorbs:
  - Resolution changes *within* a series (ES is hourly early on, 15-min later),
    so everything is resampled to hourly rather than assumed fixed-step.
  - Missing series return the plain text "no content available", not JSON, and
    not an HTTP error — that becomes a clear exception here.

`python -m datasets.energy_charts --demo` does a small live pull to verify.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from datasets._cache import cache_path, read_cache, write_cache

BASE = "https://api.energy-charts.info"
CACHE = Path("data/energy_charts")

# `production_type` codes for /public_power_forecast (underscored; note that a
# bare "wind" is NOT valid and returns "no content available").
FORECAST_TYPES = ("day-ahead", "intraday", "current")
KINDS = ("solar", "wind_onshore", "wind_offshore", "load")

# Display names of the same quantities in the /public_power response.
ACTUAL_NAMES = {
    "solar": "Solar",
    "wind_onshore": "Wind onshore",
    "wind_offshore": "Wind offshore",
    "load": "Load",
}

# Country codes worth knowing for the first slices. ES is the clean solar case
# (single bidding zone); DE/DK are the wind cases.
COUNTRIES = ("es", "de", "dk", "fr", "nl", "be", "pt", "it", "pl", "ie")


class NoContent(RuntimeError):
    """The API answered 200 with 'no content available' — series doesn't exist."""


def _get(path: str, timeout: int = 120, retries: int = 4, **params) -> dict:
    """GET with backoff. The API rate-limits (429) under the multi-year pulls the
    Phase-0 gate does, so a bare request drops slices silently-ish."""
    delay = 5.0
    for attempt in range(retries + 1):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=timeout)
        if r.status_code in (429, 502, 503, 504) and attempt < retries:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"    {r.status_code} from {path}; retrying in {wait:.0f}s "
                  f"({attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            raise NoContent(f"{path} {params}: {r.text.strip()[:120]}") from None
    raise RuntimeError(f"{path}: exhausted {retries} retries")


def _to_hourly(unix_seconds, values, name: str) -> pd.Series:
    """unix seconds + values → hourly-mean UTC Series (handles mixed resolution)."""
    idx = pd.to_datetime(pd.Series(unix_seconds), unit="s", utc=True)
    s = pd.Series(values, index=idx, dtype="float64", name=name)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.resample("h").mean()


def _cached(what: str, zone: str, start, end, fn):
    p = cache_path(CACHE, zone, what, start, end)
    hit = read_cache(p)
    if hit is not None:
        return hit.squeeze("columns") if hit.shape[1] == 1 else hit
    print(f"  energy-charts pull: {what} {zone} {start}→{end}", flush=True)
    obj = fn()
    write_cache(obj, p)
    return obj


# ---- individual series -----------------------------------------------------
def generation_actual(country: str, start, end, kind: str = "wind_onshore") -> pd.Series:
    """Actual generation (MW, hourly) for one production type."""
    name = f"gen_{kind}"

    def pull():
        d = _get("public_power", country=country, start=str(start), end=str(end))
        want = ACTUAL_NAMES[kind]
        for p in d["production_types"]:
            if p["name"] == want:
                return _to_hourly(d["unix_seconds"], p["data"], name)
        avail = [p["name"] for p in d["production_types"]]
        raise NoContent(f"{country}: no '{want}' in actuals; available={avail}")

    return _cached(name, country, start, end, pull)


def tso_forecast(country: str, start, end, kind: str = "wind_onshore",
                 forecast_type: str = "day-ahead") -> pd.Series:
    """The TSO's own forecast (MW, hourly) — the baseline to beat."""
    if forecast_type not in FORECAST_TYPES:
        raise ValueError(f"forecast_type must be one of {FORECAST_TYPES}")
    name = f"tso_{forecast_type}_{kind}"

    def pull():
        d = _get("public_power_forecast", country=country, production_type=kind,
                 forecast_type=forecast_type, start=str(start), end=str(end))
        return _to_hourly(d["unix_seconds"], d["forecast_values"], name)

    return _cached(name, country, start, end, pull)


def day_ahead_prices(bzn: str, start, end) -> pd.Series:
    """Day-ahead price (EUR/MWh, hourly). Takes a *bidding zone* (e.g. 'ES', 'DK1')."""
    def pull():
        d = _get("price", bzn=bzn, start=str(start), end=str(end))
        return _to_hourly(d["unix_seconds"], d["price"], "da_price")

    return _cached("da_price", bzn, start, end, pull)


def load(country: str, start, end) -> pd.Series:
    return generation_actual(country, start, end, "load").rename("load")


# ---- composite: total wind -------------------------------------------------
def _combine_strict(parts: dict[str, pd.Series], name: str) -> pd.Series:
    """Sum components, but yield NaN wherever *any* component is missing.

    This is deliberately strict. Summing an onshore+offshore *actual* against a
    forecast that only has onshore silently manufactures a large negative bias —
    exactly the artefact that showed up on DK in the first probe. Better a NaN
    we can see than a bias we can't.
    """
    df = pd.concat(parts.values(), axis=1)
    return df.sum(axis=1, skipna=False).rename(name)


def wind_actual(country: str, start, end, offshore: bool = True) -> pd.Series:
    parts = {"on": generation_actual(country, start, end, "wind_onshore")}
    if offshore:
        try:
            parts["off"] = generation_actual(country, start, end, "wind_offshore")
        except NoContent:
            pass                                  # landlocked / no offshore fleet
    return _combine_strict(parts, "gen_wind")


def wind_forecast(country: str, start, end, forecast_type: str = "day-ahead",
                  offshore: bool = True) -> pd.Series:
    parts = {"on": tso_forecast(country, start, end, "wind_onshore", forecast_type)}
    if offshore:
        try:
            parts["off"] = tso_forecast(country, start, end, "wind_offshore", forecast_type)
        except NoContent:
            pass
    return _combine_strict(parts, f"tso_{forecast_type}_wind")


# ---- one call for a country ------------------------------------------------
def bundle(country: str, start, end, kind: str = "wind", bzn: str | None = None) -> dict:
    """Aligned target + TSO forecast + day-ahead price for one country.

    `kind` is 'wind' (onshore+offshore) or any single entry of KINDS.
    Returns an inner-joined DataFrame under 'df' plus the raw series.
    """
    if kind == "wind":
        target, fc = (wind_actual(country, start, end),
                      wind_forecast(country, start, end))
    else:
        target, fc = (generation_actual(country, start, end, kind),
                      tso_forecast(country, start, end, kind))
    out = {"target": target, "tso_forecast": fc}
    try:
        out["da_price"] = day_ahead_prices(bzn or country.upper(), start, end)
    except (NoContent, requests.HTTPError) as e:
        print(f"  (no day-ahead price for bzn={bzn or country.upper()}: {e})")
    # Canonical column names ('target'/'tso_forecast'/'da_price'), not the raw
    # per-series names, so downstream (build_dataset.py) sees a stable schema.
    out["df"] = pd.concat({k: v for k, v in out.items()}, axis=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="small live pull (no key needed)")
    ap.add_argument("--country", default="es")
    ap.add_argument("--kind", default="solar")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-01-07")
    args = ap.parse_args()

    if not args.demo:
        print("pass --demo (hits the free Energy-Charts API) or import the loaders")
        print("kinds:", KINDS, "| forecast types:", FORECAST_TYPES)
        raise SystemExit(0)

    b = bundle(args.country, args.start, args.end, args.kind)
    df = b["df"]
    print(f"Energy-Charts {args.country} {args.kind} {args.start}→{args.end}")
    print(f"  shape={df.shape}  cols={list(df.columns)}")
    print(f"  index {df.index[0]} → {df.index[-1]} (hourly, UTC)")
    both = df[["target", "tso_forecast"]].dropna()
    print(f"  overlapping target/forecast rows: {len(both)} of {len(df)}")
    if len(both):
        err = both["tso_forecast"] - both["target"]
        print(f"  mean actual={both['target'].mean():.0f} MW  "
              f"TSO MAE={err.abs().mean():.0f} MW  bias={err.mean():+.0f} MW")
