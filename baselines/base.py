"""
Shared machinery for the probabilistic baselines.

Every baseline exposes the same interface — `fit(train_df)` then
`predict_quantiles(df, levels) -> (N, Q)` — so `evaluate.py` can score them
identically and the leaderboard is a fair comparison rather than an assortment.

The important piece here is `EmpiricalResidualWrapper`, which turns a *point*
forecast (persistence, the TSO forecast, a physical model) into a predictive
distribution. This is what makes the TSO comparison fair: the TSO publishes a
single number, so scoring it against a probabilistic model on CRPS requires
giving it a distribution too, and giving it the *best* one we reasonably can.
A strawman wrapper would hand us a fake win.

Residuals are binned by predicted level, because wind and solar errors are
strongly heteroscedastic — a forecast of 2 GW has a very different error
distribution from a forecast of 40 GW, and a single pooled residual spread would
be far too wide at the bottom and too narrow at the top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# The levels the leaderboard scores on.
#
# Checked against the closed-form Gaussian CRPS on 20k samples: this 13-level
# trapezoid integral *underestimates* absolute CRPS by 0.97% (99 levels: 0.11%).
# It is safe anyway, because the bias is the same 0.97% at 0.9x, 1.0x and 1.1x
# predictive spread — it shifts every model equally, so rankings and the
# skill-vs-TSO ratios are unaffected. Quote CRPS values as ~1% low in absolute
# terms; do not compare them to CRPS computed elsewhere at other level densities.
DEFAULT_LEVELS = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


class QuantileForecaster(ABC):
    """Common interface: fit on a train table, emit quantiles for any table."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "QuantileForecaster":
        ...

    @abstractmethod
    def predict_quantiles(self, df: pd.DataFrame,
                          levels: np.ndarray = DEFAULT_LEVELS) -> np.ndarray:
        ...

    def predict_median(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(df, np.array([0.5]))[:, 0]


def _sort_and_clip(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Enforce the two things a quantile function must satisfy: monotone, in range.

    Binned empirical residuals can cross near bin edges; sorting repairs that
    (a standard, distribution-free fix) and costs nothing when they don't cross.
    """
    return np.sort(np.clip(q, lo, hi), axis=1)


class EmpiricalResidualWrapper:
    """Point forecast → quantiles, via residual quantiles conditioned on level.

    Fitted purely on training rows; the test set never informs the spread.
    """

    def __init__(self, n_bins: int = 10, min_per_bin: int = 100):
        self.n_bins = n_bins
        self.min_per_bin = min_per_bin

    def fit(self, point: np.ndarray, y: np.ndarray, cap: float | None = None
            ) -> "EmpiricalResidualWrapper":
        point, y = np.asarray(point, float), np.asarray(y, float)
        ok = np.isfinite(point) & np.isfinite(y)
        point, y = point[ok], y[ok]
        resid = y - point

        self.lo_ = 0.0
        self.hi_ = float(cap if cap is not None else y.max() * 1.05)
        self.global_resid_ = resid

        # Quantile bin edges on the predicted level (unique-ified: a solar
        # forecast is zero for ~half the rows, which collapses several edges).
        edges = np.quantile(point, np.linspace(0, 1, self.n_bins + 1))
        self.edges_ = np.unique(edges)
        idx = np.clip(np.searchsorted(self.edges_, point, side="right") - 1,
                      0, len(self.edges_) - 2)
        self.bin_resid_ = []
        for b in range(max(len(self.edges_) - 1, 1)):
            r = resid[idx == b]
            self.bin_resid_.append(r if len(r) >= self.min_per_bin else resid)
        return self

    def _bin_of(self, point: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges_, point, side="right") - 1,
                       0, len(self.bin_resid_) - 1)

    def predict_quantiles(self, point: np.ndarray, levels: np.ndarray) -> np.ndarray:
        point = np.asarray(point, float)
        levels = np.asarray(levels, float)
        table = np.stack([np.quantile(r, levels) for r in self.bin_resid_])  # (B, Q)
        q = point[:, None] + table[self._bin_of(point)]
        return _sort_and_clip(q, self.lo_, self.hi_)


class PointBaseline(QuantileForecaster):
    """A point forecast + the residual wrapper. Subclasses supply `point()`."""

    def __init__(self, n_bins: int = 10):
        self.wrapper = EmpiricalResidualWrapper(n_bins=n_bins)

    @abstractmethod
    def point(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def fit(self, train: pd.DataFrame) -> "PointBaseline":
        self._prepare(train)
        self.wrapper.fit(self.point(train), train["y"].to_numpy(),
                         cap=float(train["y"].max() * 1.05))
        return self

    def _prepare(self, train: pd.DataFrame) -> None:
        """Hook for baselines that need to learn something before `point()` works."""

    def predict_quantiles(self, df: pd.DataFrame,
                          levels: np.ndarray = DEFAULT_LEVELS) -> np.ndarray:
        return self.wrapper.predict_quantiles(self.point(df), levels)
