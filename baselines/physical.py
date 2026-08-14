"""
Physical baselines — weather → power through physics, not statistics.

Wind: a generic aggregate power curve on hub-height wind speed. A single turbine
has a sharp cut-in/rated/cut-out shape, but a whole country's fleet is spread over
hundreds of km and many models, which smooths the curve considerably — so the
aggregate curve here is deliberately soft rather than a textbook single-turbine
step.

Solar: irradiance → power with a temperature derate, the standard first-order PV
model (efficiency falls as cells heat, which is why a 40 °C Spanish afternoon
yields less than the irradiance alone suggests).

Both have one free scale parameter (effective installed capacity) fitted by
least squares on train; everything else is physics. That keeps them honest
*physical* baselines rather than regressions in disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.base import PointBaseline

CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0


def fleet_power_curve(v: np.ndarray) -> np.ndarray:
    """Normalised (0–1) aggregate wind power curve, smoothed for fleet spread."""
    v = np.asarray(v, float)
    p = np.clip((v ** 3 - CUT_IN ** 3) / (RATED ** 3 - CUT_IN ** 3), 0.0, 1.0)
    # Soft high-wind shutdown instead of a cliff: turbines cut out at different
    # speeds across a fleet, so aggregate output tapers rather than drops.
    taper = 1.0 - np.clip((v - CUT_OUT) / 5.0, 0.0, 1.0)
    return p * taper


class WindPhysical(PointBaseline):
    name = "physical"

    def _prepare(self, train: pd.DataFrame) -> None:
        x = fleet_power_curve(train["wind_speed_100m"].to_numpy())
        y = train["y"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        # Scale-only fit through the origin: zero wind must mean zero power.
        self.cap_ = float(np.dot(x[ok], y[ok]) / max(np.dot(x[ok], x[ok]), 1e-9))

    def point(self, df: pd.DataFrame) -> np.ndarray:
        return self.cap_ * fleet_power_curve(df["wind_speed_100m"].to_numpy())


class SolarPhysical(PointBaseline):
    name = "physical"

    GAMMA = -0.004        # per-°C power temperature coefficient (typical c-Si)
    NOCT_RISE = 0.03      # cell heating per W/m² of irradiance

    def _driver(self, df: pd.DataFrame) -> np.ndarray:
        ghi = df["shortwave_radiation"].to_numpy(dtype=float)
        t_air = df["temperature_2m"].to_numpy(dtype=float)
        t_cell = t_air + self.NOCT_RISE * ghi
        return np.clip(ghi, 0, None) * (1.0 + self.GAMMA * (t_cell - 25.0))

    def _prepare(self, train: pd.DataFrame) -> None:
        x = self._driver(train)
        y = train["y"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        self.scale_ = float(np.dot(x[ok], y[ok]) / max(np.dot(x[ok], x[ok]), 1e-9))

    def point(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.scale_ * self._driver(df), 0.0, None)


def for_slice(is_solar: bool) -> PointBaseline:
    return SolarPhysical() if is_solar else WindPhysical()
