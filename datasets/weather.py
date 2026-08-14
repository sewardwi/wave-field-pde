"""
Weather feature loader — the forecasting *inputs*.

Primary path is Open-Meteo (free, no API key, simple JSON), which exposes four
endpoints that map onto the honesty discipline in docs/final-plan.md:

  - archive   (ERA5 reanalysis)         → *not* usable for headline numbers (see below)
  - hist_fc   (archived past forecasts) → best-available run per valid hour
  - prev_runs (archived *previous runs*) → **genuine day-ahead lead** — the honest input
  - forecast  (live forecast)            → operation

**The leakage subtlety that decides the whole comparison.** A day-ahead forecast
issued at gate closure only knows the NWP run from the day before. ERA5 reanalysis
is fitted to what actually happened, so training and scoring on it inflates skill.
Less obviously, the `hist_fc` archive stores the *best available* run for each valid
hour — much shorter lead than a real D-1 forecast, so it is also optimistic.

Only `prev_runs` gives a genuine forecast lead, via Open-Meteo's `<var>_previous_dayN`
variables (the forecast for a valid time as issued N days earlier). Measured on
DE, June 2024: `previous_day1` differs from the best-available run by 1.17 m/s MAE
(23% of the 5.16 m/s mean) and `previous_day2` by 1.34 — errors grow with lead,
exactly as a real forecast does, so this is genuine lead-time information.

**Choosing N, which is not as obvious as it looks.** `previous_dayN` is a *constant*
N×24h lead. A real day-ahead forecast is not: every hour of delivery day D is bid at
gate closure on D-1 (11:00 UTC), using the newest run available then — the D-1 00Z
run — so leads run from 24h (for hour 00) to 47h (for hour 23).

  lead_days=1  constant 24h. Correct only for hour 00, and **leaks** for the rest:
               for hour 23 it is the run issued 23:00 on D-1, twelve hours after
               gate closure. Optimistic — treat as an upper bound, never headline.
  lead_days=2  constant 48h. Never leaks (48h > 47h for every hour) and is at worst
               one hour pessimistic at the end of the day. The honest default.

So the headline uses lead 2 and the true operational number sits between the two.
Reporting both brackets it rather than quietly picking the flattering one.

That archive **starts ~2024-03** (2024-02 returns nulls), which is why the honest
evaluation window is 2024-03 onward rather than the full generation history. Using
a shorter, consistent window beats a longer, leaky one.

Renewable-relevant hourly variables: 100 m wind speed/direction (wind power),
shortwave/direct radiation + cloud cover (solar), 2 m temperature (air density).

`era5_cdsapi()` is scaffolding for the heavier Copernicus CDS path (finer control,
needed at scale). Note: Open-Meteo's free tier is non-commercial — fine for the
public artifact, must be revisited for a product (flagged in the plan).

`python -m datasets.weather --demo` does a small live pull to verify.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from datasets._cache import cache_path, read_cache, write_cache

WIND_SOLAR_VARS = [
    "wind_speed_100m", "wind_direction_100m",
    "shortwave_radiation", "cloud_cover", "temperature_2m", "wind_speed_10m",
    "direct_radiation",
]

CACHE = Path("data/weather")

_ENDPOINTS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "hist_fc": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "prev_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}

# First date the previous-runs archive has data (bisected 2026-08-09).
PREV_RUNS_START = "2024-03-01"


def _request(endpoint: str, params: dict, timeout: int = 300, retries: int = 4):
    """GET with backoff.

    The timeout is generous on purpose: these requests are latency-bound rather
    than size-bound (27 points x 30 days took 84s, x 60 days took 89s), so a tight
    timeout kills a request that was about to succeed and then retries it from
    scratch. Open-Meteo also rate-limits multi-year, multi-point pulls.
    """
    delay = 5.0
    for attempt in range(retries + 1):
        r = requests.get(_ENDPOINTS[endpoint], params=params, timeout=timeout)
        if r.status_code in (429, 502, 503, 504) and attempt < retries:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"    {r.status_code} from open-meteo; retrying in {wait:.0f}s "
                  f"({attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"open-meteo {endpoint}: exhausted {retries} retries")


def _open_meteo(endpoint: str, lat, lon,
                start: str | None = None, end: str | None = None,
                variables: list[str] | None = None, timeout: int = 300,
                lead_days: int = 0, **extra) -> list[pd.DataFrame]:
    """One request for one or many points; returns a DataFrame per point.

    `lead_days > 0` requests the `<var>_previous_day{N}` variants (genuine forecast
    lead) and renames them back to the plain variable names, so downstream feature
    code is identical whichever weather source is used.
    """
    variables = variables or WIND_SOLAR_VARS
    lats = [lat] if np.isscalar(lat) else list(lat)
    lons = [lon] if np.isscalar(lon) else list(lon)
    asked = [f"{v}_previous_day{lead_days}" for v in variables] if lead_days else list(variables)

    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": ",".join(asked),
        "timezone": "UTC", "wind_speed_unit": "ms",
        **extra,
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end

    payload = _request(endpoint, params, timeout=timeout)
    blocks = payload if isinstance(payload, list) else [payload]

    out = []
    for blk in blocks:
        hourly = blk["hourly"]
        idx = pd.to_datetime(hourly["time"], utc=True)
        df = pd.DataFrame({v: hourly[a] for v, a in zip(variables, asked)}, index=idx)
        df.index.name = "time"
        out.append(df.astype("float64"))
    return out


def archive(lat, lon, start, end, variables=None) -> pd.DataFrame:
    """ERA5 reanalysis. Development only — leaky as a day-ahead model input."""
    return _open_meteo("archive", lat, lon, start, end, variables)[0]


def historical_forecast(lat, lon, start, end, variables=None) -> pd.DataFrame:
    """Archived past forecasts, best-available run per valid hour (optimistic lead)."""
    return _open_meteo("hist_fc", lat, lon, start, end, variables)[0]


def day_ahead_forecast(lat, lon, start, end, variables=None, lead_days: int = 2) -> pd.DataFrame:
    """Archived forecast at a genuine D-`lead_days` lead — the honest model input."""
    return _open_meteo("prev_runs", lat, lon, start, end, variables, lead_days=lead_days)[0]


def forecast(lat, lon, variables=None, forecast_days: int = 2) -> pd.DataFrame:
    """Live forecast (operation)."""
    return _open_meteo("forecast", lat, lon, variables=variables,
                       forecast_days=forecast_days)[0]


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap physics-motivated transforms: wind³ (∝ available wind power) and a
    clear-sky-ish solar proxy. Real physical baselines (pvlib / power curves) live
    in baselines/physical.py; these are just features."""
    out = df.copy()
    if "wind_speed_100m" in out:
        out["wind100_cubed"] = out["wind_speed_100m"] ** 3
    return out


# Approx bidding-zone representative points (centroid-ish; refine per zone later).
ZONE_LATLON = {
    "DK_1": (56.15, 9.0),     # Denmark West (very wind-heavy)
    "DE_LU": (51.0, 10.0),    # Germany-Luxembourg
    "ES": (40.4, -3.7),       # Spain (solar-heavy)
    "FR": (46.6, 2.5),
}

# ---------------------------------------------------------------------------
# Zone aggregation
# ---------------------------------------------------------------------------
# A single centroid is a poor stand-in for a whole country's fleet, so each slice
# gets a handful of points sited near where that technology actually sits (German
# wind is northern; Spanish solar is southern/central). Points are equally
# weighted — we don't have per-point installed capacity, and inventing weights
# would be less honest than a flat mean. Documented as a known refinement.
ZONE_POINTS: dict[str, list[tuple[float, float]]] = {
    # Germany, wind: ~27 points covering the country with a northern bias plus
    # three offshore sites, because that is where the fleet is. A first attempt
    # with 7 points left a GBM at ~1.7x the TSO's RMSE while the same GBM given
    # the TSO forecast as a feature matched it — i.e. the shortfall was spatial
    # resolution, not the model.
    "de_wind": [(54.4, 6.6), (54.0, 7.5), (54.6, 13.9),
                (54.8, 9.4), (54.3, 8.8), (53.9, 9.9), (53.6, 8.1),
                (53.5, 11.5), (53.7, 13.3), (53.2, 10.4),
                (52.9, 8.7), (52.7, 12.4), (52.5, 9.8), (52.4, 14.0), (52.2, 11.6),
                (51.9, 8.3), (51.7, 10.8), (51.5, 13.0), (51.3, 7.5), (51.1, 11.9),
                (50.7, 9.5), (50.5, 12.2), (50.3, 7.8),
                (49.5, 10.5), (49.0, 8.5), (48.5, 11.5), (48.2, 9.0)],
    "de_solar": [(47.8, 11.0), (48.2, 9.0), (48.5, 12.5), (49.0, 8.4), (49.3, 11.0),
                 (49.8, 9.5), (50.3, 7.8), (50.9, 11.5), (51.3, 9.5), (51.5, 13.0),
                 (52.0, 8.5), (52.5, 13.4), (53.0, 8.8), (53.5, 11.0), (54.0, 9.5)],
    "es_solar": [(37.0, -6.0), (37.4, -4.8), (37.2, -3.3), (38.0, -5.5), (38.3, -3.5),
                 (38.7, -1.5), (39.0, -4.5), (39.5, -6.0), (39.4, -0.5), (40.0, -3.0),
                 (40.4, -5.0), (41.0, -1.5), (41.5, -4.5), (41.6, -0.9), (42.5, -3.5)],
    "es_wind": [(42.6, -5.6), (41.6, -0.9), (40.0, -3.0), (37.4, -5.98), (43.2, -8.0)],
    "dk_wind": [(56.15, 9.0), (55.5, 8.5), (57.0, 10.0), (55.3, 11.5)],
    "fr_wind": [(49.5, 2.5), (48.5, -1.5), (43.5, 4.5), (47.0, 5.0), (46.0, -0.5)],
}


def zone_weather(slice_key: str, start: str, end: str, source: str = "day_ahead",
                 variables: list[str] | None = None, lead_days: int = 2,
                 use_cache: bool = True, chunk_days: int = 90) -> pd.DataFrame:
    """Fleet-averaged weather for a slice (e.g. 'de_wind'), one row per hour.

    `source` is 'day_ahead' (genuine D-1 lead — the honest default), 'hist_fc'
    (best-available run) or 'archive' (ERA5; development only).

    Returns the zone mean of each variable plus `<var>_spread`, the across-point
    standard deviation, for the two variables where spatial heterogeneity carries
    real signal (a windy-in-the-north/calm-in-the-south day behaves differently
    from a uniformly moderate one at the same mean).

    Pulls are chunked in time (a multi-year, multi-point, multi-variable request
    times out server-side) and each chunk is cached, so an interrupted pull
    resumes instead of starting over.
    """
    bounds = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                           freq=f"{chunk_days}D").tolist()
    if bounds[-1] <= pd.Timestamp(end):
        bounds.append(pd.Timestamp(end) + pd.Timedelta(days=1))
    if len(bounds) > 2:
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            parts.append(_zone_weather_one(
                slice_key, a.strftime("%Y-%m-%d"),
                (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                source, variables, lead_days, use_cache))
        out = pd.concat(parts).sort_index()
        return out[~out.index.duplicated(keep="first")]
    return _zone_weather_one(slice_key, start, end, source, variables,
                             lead_days, use_cache)


def _zone_weather_one(slice_key: str, start: str, end: str, source: str,
                      variables: list[str] | None, lead_days: int,
                      use_cache: bool) -> pd.DataFrame:
    if slice_key not in ZONE_POINTS:
        raise KeyError(f"unknown slice '{slice_key}'; known: {sorted(ZONE_POINTS)}")
    variables = variables or WIND_SOLAR_VARS
    endpoint = {"day_ahead": "prev_runs", "hist_fc": "hist_fc", "archive": "archive"}[source]
    lead = lead_days if source == "day_ahead" else 0

    # The point set is part of the cache identity: editing ZONE_POINTS must
    # invalidate old files, or a re-run silently reuses the previous geometry.
    pts_id = hashlib.sha1(repr(ZONE_POINTS[slice_key]).encode()).hexdigest()[:8]
    tag = f"{source}{lead}_{'-'.join(sorted(variables))[:40]}_{pts_id}"
    p = cache_path(CACHE, slice_key, tag, start, end)
    if use_cache:
        hit = read_cache(p)
        if hit is not None:
            return hit

    pts = ZONE_POINTS[slice_key]
    print(f"  open-meteo pull: {slice_key} {source}(lead={lead}) "
          f"{len(pts)} pts {start}→{end}", flush=True)
    frames = _open_meteo(endpoint, [a for a, _ in pts], [b for _, b in pts],
                         start, end, variables, lead_days=lead)

    # Per-point derived features BEFORE averaging. Wind power goes as v³, so the
    # fleet aggregate is the mean of cubes, not the cube of the mean (Jensen —
    # they differ a lot on a spatially heterogeneous day). Direction is circular,
    # so it averages as sin/cos; a plain mean of 350° and 10° gives 180°, which
    # points the wrong way entirely.
    prepared = []
    for f in frames:
        g = add_power_features(f)
        if "wind_direction_100m" in g:
            rad = np.deg2rad(g["wind_direction_100m"])
            g = g.drop(columns=["wind_direction_100m"])
            g["wind_dir_sin"], g["wind_dir_cos"] = np.sin(rad), np.cos(rad)
        prepared.append(g)

    stacked = pd.concat(prepared, keys=range(len(prepared)))
    mean = stacked.groupby(level=1).mean()
    spread = stacked.groupby(level=1).std()
    for v in ("wind_speed_100m", "shortwave_radiation"):
        if v in spread:
            mean[f"{v}_spread"] = spread[v]
    mean = mean.sort_index()
    mean.index.name = "time"

    write_cache(mean, p)
    return mean


def era5_cdsapi(*_args, **_kwargs):
    """Scaffold for the Copernicus CDS / cdsapi ERA5 path (needs ~/.cdsapirc).
    Retrieve 'reanalysis-era5-single-levels' with 100m u/v wind, ssrd, t2m over a
    bbox, open with xarray, |wind|=sqrt(u²+v²), average over the zone. Implement
    when Open-Meteo resolution/limits become the bottleneck."""
    raise NotImplementedError("cdsapi ERA5 path is a scaffold — use archive() for now")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="small live Open-Meteo archive pull")
    ap.add_argument("--zone", default="DK_1")
    args = ap.parse_args()
    if args.demo:
        lat, lon = ZONE_LATLON[args.zone]
        df = add_power_features(archive(lat, lon, "2022-06-01", "2022-06-03"))
        print(f"Open-Meteo archive {args.zone} ({lat},{lon}) 2022-06-01→03:")
        print(f"  shape={df.shape}  cols={list(df.columns)}")
        print(f"  index {df.index[0]} → {df.index[-1]} (hourly, UTC)")
        print(f"  wind_speed_100m: mean {df['wind_speed_100m'].mean():.1f} m/s, "
              f"max {df['wind_speed_100m'].max():.1f}")
        print(f"  finite: {df.notna().all().all()}")
    else:
        print("pass --demo (hits the free Open-Meteo API) or import the loaders")
