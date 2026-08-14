"""
The TSO's own day-ahead forecast — **the bar**.

This is the incumbent operational forecast, published by the grid operator and
already used to run the market. Beating it is the fail-fast question for the
whole project (docs/final-plan.md).

It is a point forecast, so scoring it on CRPS means giving it a predictive
distribution. We give it the *same* level-conditional empirical-residual
treatment as our own point models — and deliberately no worse. Two variants:

  `tso`        — the published forecast, wrapped. The honest incumbent.
  `tso_debias` — additionally corrected for any systematic bias/scale error
                 learned on train. Not the headline bar, but a diagnostic: if we
                 only beat `tso` and not `tso_debias`, our "skill" is just bias
                 correction, which the TSO could apply themselves tomorrow. Say
                 so rather than banking it as a win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.base import PointBaseline


class TSOForecast(PointBaseline):
    name = "tso"

    def point(self, df: pd.DataFrame) -> np.ndarray:
        return df["tso_forecast"].to_numpy(dtype=float)


class TSODebiased(PointBaseline):
    """TSO forecast with a linear (scale + offset) correction fitted on train."""

    name = "tso_debias"

    def _prepare(self, train: pd.DataFrame) -> None:
        f = train["tso_forecast"].to_numpy(dtype=float)
        y = train["y"].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(y)
        self.slope_, self.intercept_ = np.polyfit(f[ok], y[ok], 1)

    def point(self, df: pd.DataFrame) -> np.ndarray:
        f = df["tso_forecast"].to_numpy(dtype=float)
        return self.slope_ * f + self.intercept_
