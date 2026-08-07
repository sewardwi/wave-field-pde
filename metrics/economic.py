"""
Economic value of a forecast — the metric a buyer actually feels.

Statistical scores (CRPS, pinball) tell you a forecast is *better*; the money
metric tells you *how much it's worth*, which is the sentence that opens a sales
conversation: "€X/MW/year saved versus the grid operator's own forecast."

Model: single-imbalance-price settlement (the standard simplification). A
generator schedules `forecast` MW in the day-ahead market at the day-ahead price,
then the deviation Δ = actual − forecast is settled at the imbalance price. The
*cost of the forecast error* relative to perfect foresight is:

    cost_t = (actual_t − forecast_t) · (da_price_t − imbalance_price_t)

Derivation: revenue = forecast·da + Δ·imb;  perfect = actual·da;
cost = perfect − revenue = Δ·(da − imb). Positive in expectation because
imbalance pricing penalizes deviation (imb > da when the system is short, imb < da
when long). A dual-price variant selects the imbalance price by deviation sign.

Everything is per-timestep (hourly MW ⇒ MWh); costs sum over the horizon. The
headline number is a *difference vs a reference forecast*, which is robust to the
exact settlement assumptions. Run `python -m metrics.economic --self-test`.
"""

from __future__ import annotations

import argparse

import numpy as np


def imbalance_cost(forecast: np.ndarray, actual: np.ndarray,
                   da_price: np.ndarray, imbalance_price: np.ndarray | None = None,
                   up_price: np.ndarray | None = None,
                   down_price: np.ndarray | None = None) -> np.ndarray:
    """
    Per-timestep imbalance cost relative to perfect foresight (currency units).

    Single price:  pass `imbalance_price`  → cost = Δ·(da − imb).
    Dual price:    pass `up_price` (short, actual<forecast) and `down_price`
                   (long, actual>forecast); the applicable price is chosen by the
                   sign of Δ = actual − forecast.

    Args:
        forecast, actual, da_price: (N,) MW / (MW) / (currency/MWh).
        imbalance_price / up_price / down_price: (N,) currency/MWh.
    Returns:
        (N,) per-step cost (can be negative on a favourable error; the *sum* is
        the net balancing cost).
    """
    forecast, actual, da_price = map(lambda a: np.asarray(a, float), (forecast, actual, da_price))
    delta = actual - forecast
    if imbalance_price is not None:
        imb = np.asarray(imbalance_price, float)
    elif up_price is not None and down_price is not None:
        up, down = np.asarray(up_price, float), np.asarray(down_price, float)
        imb = np.where(delta < 0, up, down)              # short → up-reg price, long → down-reg
    else:
        raise ValueError("provide imbalance_price, or both up_price and down_price")
    return delta * (da_price - imb)


def cost_summary(forecast, actual, da_price, imbalance_price=None,
                 up_price=None, down_price=None, capacity_mw: float | None = None,
                 hours_per_step: float = 1.0) -> dict:
    """
    Aggregate balancing cost + normalized headline figures.

    Returns dict with total cost, mean cost/step, cost per MWh of |error|, and —
    if `capacity_mw` given — the annualized €/MW/yr (the comparable figure).
    """
    cost = imbalance_cost(forecast, actual, da_price, imbalance_price, up_price, down_price)
    forecast, actual = np.asarray(forecast, float), np.asarray(actual, float)
    n = cost.shape[0]
    total = float(cost.sum())
    abs_err_mwh = float(np.abs(actual - forecast).sum() * hours_per_step)
    out = {
        "total_cost": total,
        "mean_cost_per_step": total / n,
        "cost_per_mwh_error": total / abs_err_mwh if abs_err_mwh > 0 else 0.0,
        "n_steps": n,
    }
    if capacity_mw:
        hours = n * hours_per_step
        out["cost_per_mw_year"] = total / capacity_mw * (8760.0 / hours)
    return out


def savings_vs_reference(model_fc, reference_fc, actual, da_price,
                         imbalance_price=None, up_price=None, down_price=None,
                         capacity_mw: float | None = None,
                         hours_per_step: float = 1.0) -> dict:
    """
    The headline: balancing cost of `model_fc` vs a `reference_fc` (e.g. the TSO's
    own day-ahead forecast). Positive `savings` = the model is cheaper to balance.
    """
    m = cost_summary(model_fc, actual, da_price, imbalance_price, up_price, down_price,
                     capacity_mw, hours_per_step)
    r = cost_summary(reference_fc, actual, da_price, imbalance_price, up_price, down_price,
                     capacity_mw, hours_per_step)
    out = {
        "model_total_cost": m["total_cost"],
        "reference_total_cost": r["total_cost"],
        "savings_total": r["total_cost"] - m["total_cost"],
        "savings_pct": (r["total_cost"] - m["total_cost"]) / abs(r["total_cost"]) * 100
                       if r["total_cost"] != 0 else float("nan"),
    }
    if capacity_mw:
        out["savings_per_mw_year"] = r["cost_per_mw_year"] - m["cost_per_mw_year"]
    return out


def _self_test() -> None:
    # Hand-checkable single step: short by 10 MWh, must buy at 80 vs would-sell 50.
    cost = imbalance_cost([100.0], [90.0], [50.0], [80.0])
    assert abs(cost[0] - 300.0) < 1e-9, cost      # Δ=-10, (50-80)=-30, -10*-30=300
    print(f"hand-check (short 10 @ imb 80 vs da 50): cost={cost[0]:.1f}  (expect 300.0)")

    # Long by 10 MWh, surplus sold at 30 vs would-sell 50 → lose 200.
    cost = imbalance_cost([100.0], [110.0], [50.0], [30.0])
    assert abs(cost[0] - 200.0) < 1e-9, cost      # Δ=+10, (50-30)=20, 10*20=200
    print(f"hand-check (long 10 @ imb 30 vs da 50):  cost={cost[0]:.1f}  (expect 200.0)")

    # Perfect forecast → zero cost regardless of prices.
    rng = np.random.default_rng(0)
    n = 8760
    actual = np.abs(rng.normal(50, 20, n))
    da = rng.uniform(20, 80, n)
    imb = da + rng.normal(0, 15, n)
    assert abs(imbalance_cost(actual, actual, da, imb).sum()) < 1e-6
    print("perfect forecast: total cost ≈ 0  ✓")

    # A worse forecast costs more — but ONLY under a genuinely penalizing price.
    # A single imbalance price symmetric around `da` does not penalize deviation
    # on average (a bigger error can net a lucky profit); real settlement makes
    # imbalance costly, which dual pricing (up > da > down) encodes: then
    # per-step cost = |Δ|·spread ≥ 0 and is monotone in error. This subtlety is
    # itself a modelling point to document in the writeup.
    good = actual + rng.normal(0, 3, n)
    bad = actual + rng.normal(0, 10, n)
    up, down = da + 20.0, da - 20.0
    s = savings_vs_reference(good, bad, actual, da, up_price=up, down_price=down,
                             capacity_mw=100.0)
    assert s["savings_total"] > 0, s
    print(f"good vs bad (dual-price): savings_total={s['savings_total']:.0f}  "
          f"({s['savings_pct']:.0f}%)  €/MW/yr={s['savings_per_mw_year']:.0f}")

    # Dual-price path runs and penalizes deviations.
    c = imbalance_cost([100.0, 100.0], [90.0, 110.0], [50.0, 50.0],
                       up_price=[80.0, 80.0], down_price=[30.0, 30.0])
    assert c[0] > 0 and c[1] > 0, c
    print(f"dual-price: short cost={c[0]:.0f}, long cost={c[1]:.0f}  (both > 0)")

    print("\nmetrics/economic.py self-test PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        _self_test()
    else:
        print("pass --self-test")
