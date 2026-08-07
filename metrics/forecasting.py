"""
Probabilistic-forecast metrics for the energy-forecasting pivot.

The scoring foundation for the whole artifact — the numbers a buyer (and a
reviewer) will trust only if they're correct and honestly calibrated. Everything
here is numpy (the common denominator across LightGBM baselines, torch models,
and the TSO reference forecast).

Primary metric: CRPS (continuous ranked probability score) — the standard
probabilistic-forecast score, in the same units as the target (MW). Implemented
two independent ways (sample/ensemble estimator + closed-form Gaussian) which the
self-test cross-checks, because CRPS is easy to get subtly wrong.

Also: pinball/quantile loss, PIT/reliability, interval coverage, sharpness.

Run `python -m metrics.forecasting --self-test`.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.special import ndtr   # standard-normal CDF


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def crps_ensemble(forecasts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample-based CRPS via the *fair* (unbiased) estimator (Zamo & Naveau 2018):

        CRPS ≈ mean_i |x_i - y|  -  1/(2 m (m-1)) Σ_i Σ_j |x_i - x_j|

    The double sum is computed in O(m log m) from the sorted samples using
        Σ_i Σ_j |x_i - x_j| = 2 Σ_k x_(k) (2k - m + 1)    (0-indexed, ascending).

    Args:
        forecasts: (N, m) — m ensemble members per observation.
        y:         (N,)  — observations.
    Returns:
        (N,) per-observation CRPS (same units as y). Take .mean() for the score.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    y = np.asarray(y, dtype=float)
    N, m = forecasts.shape
    if m < 2:
        raise ValueError("crps_ensemble needs >= 2 ensemble members")

    term1 = np.abs(forecasts - y[:, None]).mean(axis=1)

    xs = np.sort(forecasts, axis=1)
    k = np.arange(m)
    # Σ_{i<j}(x_j - x_i) = Σ_k x_(k) (2k - m + 1); fair 2nd term = that / (m(m-1)).
    pairwise = (xs * (2 * k - m + 1)).sum(axis=1)
    term2 = pairwise / (m * (m - 1))
    return term1 - term2


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian predictive distribution (Gneiting & Raftery 2007):

        CRPS(N(mu, sigma), y) = sigma [ z(2Φ(z) - 1) + 2φ(z) - 1/√π ],   z = (y-mu)/sigma
    """
    mu, sigma, y = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(y, float)
    sigma = np.clip(sigma, 1e-9, None)
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def crps_from_quantiles(pred_q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    CRPS approximated from a set of predictive quantiles via the identity
    CRPS = 2 ∫₀¹ QL_τ dτ, trapezoid-integrated over the provided `levels`.

    Args:
        pred_q: (N, Q) predicted quantile values (must be sorted along Q by level).
        y:      (N,)
        levels: (Q,) quantile levels in (0,1), ascending.
    Returns:
        (N,) approximate CRPS. (Approximation improves with denser levels.)
    """
    pred_q, y, levels = np.asarray(pred_q, float), np.asarray(y, float), np.asarray(levels, float)
    ql = pinball_loss(pred_q, y, levels)                 # (N, Q)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # np>=2.0 renamed trapz
    return 2.0 * trapezoid(ql, levels, axis=1)


# ---------------------------------------------------------------------------
# Pinball / quantile loss
# ---------------------------------------------------------------------------

def pinball_loss(pred_q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Pinball (quantile) loss. Broadcasts over Q quantile levels.

        L_τ(q, y) = max(τ(y - q), (τ - 1)(y - q))

    Args:
        pred_q: (N, Q) or (N,) predicted quantiles.
        y:      (N,)
        levels: (Q,) or scalar τ.
    Returns:
        (N, Q) per-obs per-level loss (or (N,) if scalar level).
    """
    pred_q, y = np.asarray(pred_q, float), np.asarray(y, float)
    levels = np.asarray(levels, float)
    if pred_q.ndim == 1:
        diff = y - pred_q
        return np.maximum(levels * diff, (levels - 1) * diff)
    diff = y[:, None] - pred_q                            # (N, Q)
    return np.maximum(levels[None, :] * diff, (levels[None, :] - 1) * diff)


# ---------------------------------------------------------------------------
# Calibration / reliability
# ---------------------------------------------------------------------------

def pit_ensemble(forecasts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform values from an ensemble: fraction of members
    ≤ y (with a random tie-break for the discrete ensemble, so a calibrated
    forecast gives PIT ~ Uniform(0,1)). A flat PIT histogram = calibrated;
    ∪-shape = under-dispersed (overconfident); ∩-shape = over-dispersed.
    """
    forecasts, y = np.asarray(forecasts, float), np.asarray(y, float)
    below = (forecasts < y[:, None]).sum(axis=1)
    equal = (forecasts == y[:, None]).sum(axis=1)
    u = np.random.uniform(size=y.shape[0])
    return (below + u * (equal + 1)) / (forecasts.shape[1] + 1)


def quantile_coverage(pred_q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Empirical coverage P(y ≤ pred_τ) per level; should ≈ `levels` if calibrated."""
    pred_q, y = np.asarray(pred_q, float), np.asarray(y, float)
    return (y[:, None] <= pred_q).mean(axis=0)            # (Q,)


def interval_coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """Empirical coverage of the central interval [lo, hi]."""
    lo, hi, y = np.asarray(lo, float), np.asarray(hi, float), np.asarray(y, float)
    return float(((y >= lo) & (y <= hi)).mean())


def sharpness(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean interval width (calibrated AND sharp is the goal, not just wide)."""
    return float((np.asarray(hi, float) - np.asarray(lo, float)).mean())


def calibration_error(pred_q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    """Scalar reliability: RMS gap between empirical coverage and nominal levels."""
    cov = quantile_coverage(pred_q, y, levels)
    return float(np.sqrt(np.mean((cov - np.asarray(levels, float)) ** 2)))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.default_rng(0)
    N = 20000

    # 1) Perfect deterministic forecast → CRPS 0.
    y = rng.normal(size=N)
    perfect = np.repeat(y[:, None], 4, axis=1)
    c = crps_ensemble(perfect, y).mean()
    assert c < 1e-9, c
    print(f"perfect ensemble:      CRPS={c:.2e}  (expect 0)")

    # 2) Ensemble CRPS ≈ closed-form Gaussian CRPS.
    mu, sigma = 3.0, 2.0
    y = rng.normal(mu, sigma, size=N)
    ens = rng.normal(mu, sigma, size=(N, 400))
    c_ens = crps_ensemble(ens, y).mean()
    c_gauss = crps_gaussian(np.full(N, mu), np.full(N, sigma), y).mean()
    assert abs(c_ens - c_gauss) < 0.02 * sigma, (c_ens, c_gauss)
    print(f"ensemble vs gaussian:  {c_ens:.4f} vs {c_gauss:.4f}  (analytic σ/√π={sigma/np.sqrt(np.pi):.4f})")

    # 3) CRPS-from-quantiles ≈ closed form.
    levels = np.linspace(0.01, 0.99, 99)
    from scipy.special import ndtri
    pred_q = mu + sigma * ndtri(levels)[None, :].repeat(N, axis=0)
    c_q = crps_from_quantiles(pred_q, y, levels).mean()
    assert abs(c_q - c_gauss) < 0.02 * sigma, (c_q, c_gauss)
    print(f"quantile-CRPS:         {c_q:.4f}  (vs gaussian {c_gauss:.4f})")

    # 4) Calibration: a correctly-specified Gaussian is calibrated; an
    #    overconfident (too-narrow) one under-covers.
    cov = quantile_coverage(pred_q, y, levels)
    ce = calibration_error(pred_q, y, levels)
    assert ce < 0.02, ce
    narrow = mu + 0.3 * sigma * ndtri(levels)[None, :].repeat(N, axis=0)
    lo, hi = narrow[:, 4], narrow[:, 94]                  # ~90% nominal interval
    cov90_narrow = interval_coverage(lo, hi, y)
    assert cov90_narrow < 0.9, cov90_narrow
    print(f"calibration error:     {ce:.4f} (calibrated)   overconfident 90% cov={cov90_narrow:.2f} (<0.90)")

    # 5) PIT of a calibrated ensemble is ~uniform (mean ~0.5, flat-ish).
    ens = rng.normal(y[:, None], 1.0, size=(N, 50))       # calibrated around y+N(0,1)...
    ens = rng.normal(mu, sigma, size=(N, 50))
    pit = pit_ensemble(ens, y)
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    rel = hist / hist.mean()
    assert rel.min() > 0.85 and rel.max() < 1.15, rel
    print(f"PIT uniformity:        bin counts within ±15% of flat  (min {rel.min():.2f}, max {rel.max():.2f})")

    # 6) Pinball loss is minimized at the true quantile.
    y1 = rng.normal(size=200000)
    tau = 0.9
    true_q = ndtri(tau)
    losses = {q: pinball_loss(np.full_like(y1, q), y1, tau).mean()
              for q in [true_q - 0.5, true_q, true_q + 0.5]}
    assert losses[true_q] < losses[true_q - 0.5] and losses[true_q] < losses[true_q + 0.5]
    print(f"pinball @τ=0.9 min at true q={true_q:.3f}: "
          f"{losses[true_q-0.5]:.4f} > {losses[true_q]:.4f} < {losses[true_q+0.5]:.4f}")

    print("\nmetrics/forecasting.py self-test PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        _self_test()
    else:
        print("pass --self-test")
