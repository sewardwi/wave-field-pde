"""
Climatology — the calibrated no-skill reference.

Natively probabilistic: the empirical distribution of generation for this
hour-of-day and month, straight from the training years. It uses no weather at
all, so it is the score to beat before claiming any weather skill. It is also
usually *well calibrated* while being very unsharp, which is exactly why CRPS
(not calibration alone) has to be the headline metric.

Bins are (month, hour). With ~2 years of training data that is ~60 samples per
bin, so bins fall back to hour-only when thin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.base import DEFAULT_LEVELS, QuantileForecaster, _sort_and_clip


class Climatology(QuantileForecaster):
    name = "climatology"

    def __init__(self, min_per_bin: int = 30):
        self.min_per_bin = min_per_bin

    def fit(self, train: pd.DataFrame) -> "Climatology":
        y = train["y"].astype(float)
        self.lo_, self.hi_ = 0.0, float(y.max() * 1.05)
        self.by_mh_ = {k: v.to_numpy() for k, v in
                       y.groupby([train.index.month, train.index.hour])}
        self.by_h_ = {k: v.to_numpy() for k, v in y.groupby(train.index.hour)}
        self.all_ = y.to_numpy()
        return self

    def predict_quantiles(self, df: pd.DataFrame,
                          levels: np.ndarray = DEFAULT_LEVELS) -> np.ndarray:
        levels = np.asarray(levels, float)
        cache: dict[tuple[int, int], np.ndarray] = {}
        out = np.empty((len(df), len(levels)))
        for i, ts in enumerate(df.index):
            key = (ts.month, ts.hour)
            if key not in cache:
                sample = self.by_mh_.get(key)
                if sample is None or len(sample) < self.min_per_bin:
                    sample = self.by_h_.get(ts.hour, self.all_)
                cache[key] = np.quantile(sample, levels)
            out[i] = cache[key]
        return _sort_and_clip(out, self.lo_, self.hi_)
