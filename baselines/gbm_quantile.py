"""
LightGBM quantile regression — the industry workhorse, and the real test.

This is the baseline that decides whether the project has a result. Gradient
boosting with a pinball objective is what a competent forecasting desk actually
runs; if a fancy generative model can't beat a well-tuned GBM, there is no story.
Equally, if the GBM already beats the TSO forecast, that is the headline finding
and the fancy model is upside rather than the point.

One model per quantile level (LightGBM's `objective='quantile'` optimises a single
level at a time). Independently-fitted quantiles can cross; `_sort_and_clip`
repairs that by sorting, which is the standard distribution-free fix.

Hyperparameters are deliberately modest and shared across levels — tuned enough
to be a fair fight, not so much that the baseline becomes a research project.
Early stopping uses a validation slice, so `fit` accepts one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from baselines.base import DEFAULT_LEVELS, QuantileForecaster, _sort_and_clip

PARAMS = dict(
    objective="quantile",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


class GBMQuantile(QuantileForecaster):
    name = "gbm_quantile"

    def __init__(self, features: list[str], levels: np.ndarray = DEFAULT_LEVELS,
                 params: dict | None = None, n_estimators: int | None = None):
        self.features = list(features)
        self.levels = np.asarray(levels, float)
        self.params = {**PARAMS, **(params or {})}
        if n_estimators:
            self.params["n_estimators"] = n_estimators

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "GBMQuantile":
        import lightgbm as lgb

        X, y = train[self.features], train["y"].astype(float)
        self.lo_, self.hi_ = 0.0, float(y.max() * 1.05)
        eval_set = [(val[self.features], val["y"].astype(float))] if val is not None else None

        self.models_ = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tau in self.levels:
                m = lgb.LGBMRegressor(alpha=float(tau), **self.params)
                m.fit(X, y, eval_set=eval_set,
                      callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None)
                self.models_[float(tau)] = m
        return self

    def predict_quantiles(self, df: pd.DataFrame,
                          levels: np.ndarray = DEFAULT_LEVELS) -> np.ndarray:
        levels = np.asarray(levels, float)
        missing = [t for t in levels if float(t) not in self.models_]
        if missing:
            raise ValueError(f"model was not fitted for levels {missing}; "
                             f"construct GBMQuantile(levels=...) to match")
        X = df[self.features]
        q = np.column_stack([self.models_[float(t)].predict(X) for t in levels])
        return _sort_and_clip(q, self.lo_, self.hi_)
