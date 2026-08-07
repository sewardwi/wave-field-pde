"""
Weather feature loader — the forecasting *inputs*.

Primary path is Open-Meteo (free, no API key, simple JSON), which exposes three
endpoints that map exactly onto the honesty discipline in docs/final-plan.md:

  - archive  (ERA5 reanalysis)        → model *development / training*
  - hist-fc  (archived past forecasts) → honest *evaluation* (what was actually
                                         knowable at issue time — no leakage)
  - forecast (live forecast)           → operation

Renewable-relevant hourly variables: 100 m wind speed/direction (wind power),
shortwave radiation + cloud cover (solar), 2 m temperature.

`era5_cdsapi()` is scaffolding for the heavier Copernicus CDS path (finer control,
needed at scale). Note: Open-Meteo's free tier is non-commercial — fine for the
public artifact, must be revisited for a product (flagged in the plan).

`python -m datasets.weather --demo` does a small live archive pull to verify.
"""

from __future__ import annotations

import argparse

import pandas as pd
import requests

WIND_SOLAR_VARS = [
    "wind_speed_100m", "wind_direction_100m",
    "shortwave_radiation", "cloud_cover", "temperature_2m", "wind_speed_10m",
]

_ENDPOINTS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "hist_fc": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}


def _open_meteo(endpoint: str, lat: float, lon: float,
                start: str | None = None, end: str | None = None,
                variables: list[str] | None = None, timeout: int = 60,
                **extra) -> pd.DataFrame:
    variables = variables or WIND_SOLAR_VARS
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(variables),
        "timezone": "UTC", "wind_speed_unit": "ms",
        **extra,
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    r = requests.get(_ENDPOINTS[endpoint], params=params, timeout=timeout)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    idx = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({k: hourly[k] for k in variables}, index=idx)
    df.index.name = "time"
    return df


def archive(lat, lon, start, end, variables=None) -> pd.DataFrame:
    """ERA5 reanalysis (training). start/end as 'YYYY-MM-DD'."""
    return _open_meteo("archive", lat, lon, start, end, variables)


def historical_forecast(lat, lon, start, end, variables=None) -> pd.DataFrame:
    """Archived past forecasts (honest eval — what was knowable at issue time)."""
    return _open_meteo("hist_fc", lat, lon, start, end, variables)


def forecast(lat, lon, variables=None, forecast_days: int = 2) -> pd.DataFrame:
    """Live forecast (operation)."""
    return _open_meteo("forecast", lat, lon, variables=variables, forecast_days=forecast_days)


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
