"""
Field-reconstruction metrics for PDE / scientific-field experiments.

Replaces the image/audio Frechet metrics for the PDE pivot. The two headline
numbers, following the FNO literature (Li et al. 2020):

  - relative L2 error   — the standard operator-learning metric. Per-sample
                          ||pred - target|| / ||target||, averaged over the batch.
  - spectrum error      — relative error between the radially-averaged (isotropic)
                          energy spectra of pred vs target. Catches the failure
                          mode where a model wins L2 by over-smoothing but loses
                          the high-frequency content (rule: a good L2 with a bad
                          spectrum is a warning, not a win).

Both operate on fields shaped (B, C, H, W) — C is physical/time channels, not RGB.

Run `python -m metrics.field --self-test` for the sanity checks:
  identical fields → 0;  predict-zero → rel-L2 = 1;  independent noise → ≈√2.
"""

from __future__ import annotations

import argparse

import torch


# ---------------------------------------------------------------------------
# Relative L2
# ---------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                reduction: str = "mean", eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample relative L2 error, the standard FNO metric.

        e_i = ||pred_i - target_i||_2 / (||target_i||_2 + eps)

    where the norm is over all non-batch dims (C, H, W).

    Args:
        pred, target: (B, ...) tensors of matching shape.
        reduction:    'mean' | 'sum' | 'none' over the batch dimension.
        eps:          guard against a zero-norm target.
    Returns:
        scalar (reduction='mean'/'sum') or (B,) tensor (reduction='none').
    """
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    B = pred.shape[0]
    num = (pred - target).reshape(B, -1).norm(dim=1)
    den = target.reshape(B, -1).norm(dim=1).clamp(min=eps)
    err = num / den
    if reduction == "mean":
        return err.mean()
    if reduction == "sum":
        return err.sum()
    if reduction == "none":
        return err
    raise ValueError(f"unknown reduction {reduction!r}")


# ---------------------------------------------------------------------------
# Radially-averaged (isotropic) energy spectrum
# ---------------------------------------------------------------------------

def radial_spectrum(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Isotropic (shell-averaged) power spectrum of a 2D field.

    For each leading-index sample, take the 2D FFT, form the power |F(k)|², and
    average power over each integer-|k| shell. Shell *averaging* (mean power per
    mode at radius k), not shell integration, so the metric is resolution-stable
    and directly comparable between fields of the same grid.

    Args:
        field: (..., H, W) real tensor.
    Returns:
        spectrum: (..., n_bins) shell-averaged power, n_bins = floor(|k|_max) + 1.
        k:        (n_bins,) integer shell wavenumbers 0..n_bins-1.
    """
    *lead, H, W = field.shape
    N = int(torch.tensor(lead).prod().item()) if lead else 1
    x = field.reshape(N, H, W)

    psd = torch.fft.fft2(x).abs() ** 2                      # (N, H, W)

    # Integer wavenumber magnitude on the FFT grid.
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=field.device)   # cycles over the grid
    kx = torch.fft.fftfreq(W, d=1.0 / W, device=field.device)
    kmag = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)      # (H, W)
    kbin = kmag.round().long().reshape(-1)                      # (H*W,)
    n_bins = int(kbin.max().item()) + 1

    psd_flat = psd.reshape(N, H * W)
    spec_sum = torch.zeros(N, n_bins, device=field.device, dtype=psd.dtype)
    spec_sum.index_add_(1, kbin, psd_flat)
    counts = torch.zeros(n_bins, device=field.device, dtype=psd.dtype)
    counts.index_add_(0, kbin, torch.ones_like(psd_flat[0]))
    spec = spec_sum / counts.clamp(min=1)                       # (N, n_bins)

    spec = spec.reshape(*lead, n_bins) if lead else spec.reshape(n_bins)
    k = torch.arange(n_bins, device=field.device)
    return spec, k


def spectrum_error(pred: torch.Tensor, target: torch.Tensor,
                   eps: float = 1e-12, log: bool = False) -> torch.Tensor:
    """
    Relative L2 error between the batch-mean isotropic energy spectra.

    Fields are averaged over batch (and channels) into a single mean spectrum for
    pred and for target, then compared:

        err = || E_pred - E_target ||_2 / || E_target ||_2                 (log=False)
        err = || logE_pred - logE_target ||_2 / || logE_target ||_2        (log=True)

    log=True weights all scales more evenly (spectra span many orders of
    magnitude); log=False is dominated by the energetic low-|k| modes.

    Args:
        pred, target: (B, C, H, W) or (B, H, W).
    Returns:
        scalar tensor.
    """
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    # Collapse everything but (H, W) so the spectrum is averaged over B and C.
    ep, _ = radial_spectrum(pred)
    et, _ = radial_spectrum(target)
    ep = ep.reshape(-1, ep.shape[-1]).mean(0)
    et = et.reshape(-1, et.shape[-1]).mean(0)
    if log:
        ep = torch.log(ep.clamp(min=eps))
        et = torch.log(et.clamp(min=eps))
    return (ep - et).norm() / et.norm().clamp(min=eps)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def field_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Standard bundle: relative L2, spectrum error (linear + log), correlation."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    pc = p - p.mean(1, keepdim=True)
    tc = t - t.mean(1, keepdim=True)
    corr = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1)).clamp(min=1e-8)
    return {
        "rel_l2": relative_l2(pred, target).item(),
        "spectrum_err": spectrum_error(pred, target).item(),
        "spectrum_err_log": spectrum_error(pred, target, log=True).item(),
        "correlation": corr.mean().item(),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    B, C, H, W = 8, 2, 64, 64
    target = torch.randn(B, C, H, W)

    # Identical → everything perfect.
    m = field_metrics(target, target)
    assert m["rel_l2"] < 1e-6, m
    assert m["spectrum_err"] < 1e-6, m
    assert abs(m["correlation"] - 1.0) < 1e-5, m
    print(f"identical:        rel_l2={m['rel_l2']:.2e}  spec={m['spectrum_err']:.2e}  corr={m['correlation']:.4f}")

    # Predict zero → rel-L2 = 1 exactly, correlation ~0.
    zeros = torch.zeros_like(target)
    e = relative_l2(zeros, target).item()
    assert abs(e - 1.0) < 1e-5, e
    print(f"predict-zero:     rel_l2={e:.4f}  (expected 1.0000)")

    # Independent noise → rel-L2 ≈ √2, correlation ~0.
    other = torch.randn_like(target)
    e = relative_l2(other, target).item()
    corr = field_metrics(other, target)["correlation"]
    assert 1.3 < e < 1.5, e
    assert abs(corr) < 0.05, corr
    print(f"independent noise: rel_l2={e:.4f}  (expected ≈{2**0.5:.4f})  corr={corr:+.4f}")

    # Spectrum error catches over-smoothing: low-pass the target, rel-L2 may be
    # modest but the spectrum error is clearly non-zero.
    F = torch.fft.fft2(target)
    ky = torch.fft.fftfreq(H, d=1.0 / H)[:, None]
    kx = torch.fft.fftfreq(W, d=1.0 / W)[None, :]
    mask = (ky ** 2 + kx ** 2) <= (H / 6) ** 2
    smooth = torch.fft.ifft2(F * mask).real
    m = field_metrics(smooth, target)
    assert m["spectrum_err"] > 0.05, m
    print(f"over-smoothed:    rel_l2={m['rel_l2']:.4f}  spec={m['spectrum_err']:.4f}  spec_log={m['spectrum_err_log']:.4f}")

    print("\nmetrics/field.py self-test PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        print("nothing to do; pass --self-test")
