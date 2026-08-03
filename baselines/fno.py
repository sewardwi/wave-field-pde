"""
FNO-2D baseline (Li et al. 2020) — the primary reference for the wave operator.

Standard architecture: lift input channels (+ appended x,y grid) to a width-D
feature field, apply N blocks of (SpectralConv2d + pointwise 1×1 Conv) with GELU,
then project to the output channels. SpectralConv2d keeps the lowest `modes`
Fourier modes and learns a *free dense complex channel-mixing map per mode* — the
expressive spectral operator the wave op's 3-scalar parametric kernel is measured
against (see the fairness note in docs/final-plan.md: matched params is necessary
but not sufficient — the two allocate capacity very differently).

The `train` CLI doubles as the Phase-0 "reproduce the baseline" tool: on the full
ν=1e-3 dataset it should land near the literature ~1e-2 relative L2. On a tiny set
it is an overfit / wiring sanity check.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Keep the lowest `modes1×modes2` Fourier modes; learn a dense complex
    (in→out) channel map per retained mode. Two weight blocks cover the low
    positive and low negative vertical wavenumbers (rfft2 keeps the full first
    axis, half the second)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in, x, y) × (in, out, x, y) → (B, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(H, W))


# ---------------------------------------------------------------------------
# FNO-2D
# ---------------------------------------------------------------------------

class FNO2d(nn.Module):
    """
    Args:
        in_channels:  input time slices (t_in).
        out_channels: predicted time slices (t_out).
        modes:        retained Fourier modes per axis (12 is standard for 64²).
        width:        feature width.
        n_layers:     spectral blocks.
        use_grid:     append normalized (x,y) coordinate channels (standard FNO).
    Forward: (B, in_channels, H, W) → (B, out_channels, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int = 12,
                 width: int = 32, n_layers: int = 4, use_grid: bool = True):
        super().__init__()
        self.use_grid = use_grid
        self.width = width
        lift_in = in_channels + (2 if use_grid else 0)
        self.fc0 = nn.Linear(lift_in, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    @staticmethod
    def _grid(shape, device) -> torch.Tensor:
        B, _, H, W = shape
        gx = torch.linspace(0, 1, W, device=device).reshape(1, 1, 1, W).expand(B, 1, H, W)
        gy = torch.linspace(0, 1, H, device=device).reshape(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self._grid(x.shape, x.device)], dim=1)
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # (B, width, H, W)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = conv(x) + w(x)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
        x = F.gelu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).permute(0, 3, 1, 2)                      # (B, out, H, W)

    def param_count(self) -> int:
        # complex params count as 2 reals.
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())


# ---------------------------------------------------------------------------
# Training / reproduce-the-baseline entry
# ---------------------------------------------------------------------------

def train(args) -> dict:
    from torch.utils.data import DataLoader
    from datasets.navier_stokes import load_ns_splits
    from metrics.field import relative_l2, field_metrics

    device = torch.device(args.device)
    splits = load_ns_splits(args.data, t_in=args.t_in, t_out=args.t_out)
    tr, te = splits["train"], splits["test"]
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = FNO2d(tr.in_channels, tr.out_channels, modes=args.modes,
                  width=args.width, n_layers=args.layers).to(device)
    print(f"FNO2d params: {model.param_count():,}  "
          f"(in={tr.in_channels} out={tr.out_channels} modes={args.modes} width={args.width})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = relative_l2(model(x), y)                 # FNO trains with rel-L2
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        sched.step()
        if ep % args.log_every == 0 or ep == args.epochs:
            print(f"  epoch {ep:4d}  train rel_l2 {tot / n:.4f}")

    # Evaluation on the held-out test split.
    model.eval()
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for x, y in DataLoader(te, batch_size=args.batch_size):
            xs.append(x); ys.append(y); ps.append(model(x.to(device)).cpu())
    if xs:
        pred, targ = torch.cat(ps), torch.cat(ys)
        m = field_metrics(pred, targ)
        print(f"TEST  rel_l2={m['rel_l2']:.4f}  spec_err={m['spectrum_err']:.4f}  "
              f"corr={m['correlation']:.4f}")
        return m
    print("TEST  (empty test split — too few trajectories)")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-in", dest="t_in", type=int, default=10)
    ap.add_argument("--t-out", dest="t_out", type=int, default=None)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", dest="log_every", type=int, default=10)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    train(ap.parse_args())


if __name__ == "__main__":
    main()
