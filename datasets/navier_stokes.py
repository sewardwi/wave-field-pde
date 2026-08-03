"""
Navier–Stokes dataset loader for the deterministic operator-learning task.

Consumes the trajectory bundle written by `datasets.ns_generator` (or any
`.pt` with the same schema) and forms the FNO benchmark's temporal map:

    input   x = vorticity over the first  t_in  recorded times   → (t_in,  H, W)
    target  y = vorticity over the next   t_out recorded times    → (t_out, H, W)

Time slices are stacked as channels (the FNO-3D / "predict the whole future
window at once" form), so a single forward pass maps x → y. This is what the
deterministic Phase-1 wave-vs-FNO comparison and the FNO baseline both train on.

Standardization is **per physical variable, fit on the training split only**:
NS vorticity is a single variable, so one global (mean, std) is applied to both
x and y. We deliberately do NOT standardize per time-slice — that would erase
the physical decay of energy over time and put input/target on inconsistent
scales. Stats are saved with the split so evaluation and later phases reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class Standardizer:
    """Single-variable standardization: (x - mean) / std, with saved stats."""
    mean: float
    std: float

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardizer":
        return cls(mean=float(x.mean()), std=float(x.std().clamp(min=1e-8)))


class NavierStokesDataset(Dataset):
    """
    One split of the NS operator-learning task.

    Args:
        data:  (N, T, H, W) vorticity trajectories for this split.
        t_in:  number of leading time slices used as input.
        t_out: number of following time slices to predict (default: all remaining).
        stats: a fitted Standardizer (fit on train, shared to val/test). If None,
               fit on `data` — only do this for the training split.
    Each item: (x, y) with x=(t_in, H, W), y=(t_out, H, W), standardized.
    """

    def __init__(self, data: torch.Tensor, t_in: int = 10,
                 t_out: int | None = None, stats: Standardizer | None = None):
        assert data.dim() == 4, f"expected (N,T,H,W), got {tuple(data.shape)}"
        N, T, H, W = data.shape
        if t_out is None:
            t_out = T - t_in
        assert t_in + t_out <= T, f"t_in+t_out={t_in + t_out} exceeds T={T}"

        self.data = data
        self.t_in, self.t_out = t_in, t_out
        self.stats = stats if stats is not None else Standardizer.fit(data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        traj = self.data[idx]                                   # (T, H, W)
        x = self.stats.normalize(traj[:self.t_in])
        y = self.stats.normalize(traj[self.t_in:self.t_in + self.t_out])
        return x, y

    @property
    def in_channels(self) -> int:
        return self.t_in

    @property
    def out_channels(self) -> int:
        return self.t_out


def load_ns_splits(path: str, t_in: int = 10, t_out: int | None = None,
                   split: tuple[float, float, float] = (0.8, 0.1, 0.1),
                   seed: int = 0) -> dict:
    """
    Load a trajectory bundle and deterministically split into train/val/test,
    fitting standardization on the training split only.

    Returns a dict: {"train","val","test": NavierStokesDataset, "stats","meta"}.

    Note: for an exact match to a published FNO number, use that paper's split
    *sizes* (e.g. 1000 train / 200 test) rather than these fractions — set them
    via `split` accordingly. Whatever split is used, the baseline must be
    reproduced on this same data (rule #1).
    """
    bundle = torch.load(path, weights_only=False)
    data = bundle["data"] if isinstance(bundle, dict) else bundle    # (N,T,H,W)
    N = data.shape[0]

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_tr = int(round(split[0] * N))
    n_va = int(round(split[1] * N))
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    train = NavierStokesDataset(data[idx_tr], t_in, t_out)          # fits stats
    stats = train.stats
    val = NavierStokesDataset(data[idx_va], t_in, t_out, stats=stats)
    test = NavierStokesDataset(data[idx_te], t_in, t_out, stats=stats)

    return {
        "train": train, "val": val, "test": test, "stats": stats,
        "meta": bundle.get("meta", {}) if isinstance(bundle, dict) else {},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--t-in", type=int, default=10)
    args = ap.parse_args()

    s = load_ns_splits(args.path, t_in=args.t_in)
    tr, va, te = s["train"], s["val"], s["test"]
    x, y = tr[0]
    print(f"splits: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"item: x={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"stats (train): mean={s['stats'].mean:.4e} std={s['stats'].std:.4e}")
    print(f"normalized x: mean={x.mean():.3f} std={x.std():.3f}  (≈0, ≈1 expected)")
    print(f"meta: {s['meta']}")
