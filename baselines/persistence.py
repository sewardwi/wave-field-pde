"""
Persistence — the floor every forecast must clear.

For a day-ahead forecast the naive answer is "tomorrow looks like the last day I
fully observed." At gate closure on D-1 that is **day D-2**, not D-1: at 11:00
UTC on D-1 only part of that day has happened, so a same-hour-yesterday
persistence would use unobserved data for most hours. `y_sameh_d2` (built in
datasets/build_dataset.py) is the honest version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.base import PointBaseline


class Persistence(PointBaseline):
    name = "persistence"

    def point(self, df: pd.DataFrame) -> np.ndarray:
        return df["y_sameh_d2"].to_numpy(dtype=float)
