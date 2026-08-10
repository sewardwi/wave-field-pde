"""
Tiny on-disk cache shared by the data loaders (parquet, pickle fallback).

Factored out of datasets/entsoe.py so the ENTSO-E and Energy-Charts loaders
can't drift apart — the repo has been bitten before by the same helper living
in three places (see the StandardAttention note in CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def cache_path(root: Path, zone: str, what: str, start, end) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{zone}_{what}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}"
    return root / f"{tag}.parquet"


def read_cache(p: Path):
    """Return the cached object as a DataFrame, or None on a miss.

    Must check the .pkl fallback even when the .parquet is absent: without
    pyarrow installed every write lands as .pkl, and an early `return None`
    here turns the whole cache into a permanent miss (each 3-year pull is ~30s).
    """
    for path, reader in ((p, pd.read_parquet), (p.with_suffix(".pkl"), pd.read_pickle)):
        if path.exists():
            try:
                obj = reader(path)
            except Exception:
                continue
            return obj.to_frame() if isinstance(obj, pd.Series) else obj
    return None


def write_cache(obj, p: Path) -> None:
    """Store as a DataFrame so read_cache can always return one (callers squeeze)."""
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    try:
        df.to_parquet(p)
    except Exception:
        df.to_pickle(p.with_suffix(".pkl"))      # fallback if no parquet engine
