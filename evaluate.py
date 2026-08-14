"""
The leaderboard — every model against the TSO forecast, one command.

    python evaluate.py --slice de_wind
    python evaluate.py --slice es_solar --all-hours
    python evaluate.py --slice de_wind --source hist_fc     # optimistic-weather ablation

This is the Phase-1 fail-fast (docs/final-plan.md): can a calibrated model beat
the grid operator's own day-ahead forecast on CRPS? Everything is scored on the
same rows, from the same table, with the same quantile levels.

Protocol notes that keep it honest:
  - **Temporal split.** Train strictly before val, val strictly before test. No
    shuffling, so no training on the future.
  - **Honest weather.** Inputs default to a 48h NWP lead, which is the shortest
    lead that cannot leak past day-ahead gate closure for *any* hour of the
    delivery day — not reanalysis, and not the best-available run. `--lead-days 1`
    gives the optimistic bracket; the operational truth lies between the two.
    See the "Choosing N" note in datasets/weather.py.
  - **The TSO gets a fair distribution.** Its point forecast is wrapped with the
    same level-conditional empirical residuals our own point models get.
  - **Skill scores are relative to the TSO**, not to persistence, because beating
    persistence is not a claim anyone should be impressed by.

On the economic column: Energy-Charts publishes day-ahead prices but not
imbalance prices, so the €-settlement metric in metrics/economic.py can only be
run with a *proxy* imbalance spread. It is reported as an indication and clearly
marked; the real number needs ENTSO-E imbalance prices. CRPS is the headline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.base import DEFAULT_LEVELS
from baselines.climatology import Climatology
from baselines.gbm_quantile import GBMQuantile
from baselines.persistence import Persistence
from baselines.physical import for_slice as physical_for_slice
from baselines.tso_forecast import TSODebiased, TSOForecast
from datasets import build_dataset as bd
from datasets import slices as sl
from metrics import forecasting as mf

# Proxy imbalance spread as a fraction of the day-ahead price, used only for the
# indicative € column. Real settlement needs published imbalance prices.
PROXY_IMBALANCE_SPREAD = 0.30


def score(pred_q: np.ndarray, y: np.ndarray, levels: np.ndarray,
          da_price: np.ndarray | None = None) -> dict:
    """All the metrics for one model on one set of rows."""
    crps = mf.crps_from_quantiles(pred_q, y, levels).mean()
    median = pred_q[:, np.argmin(np.abs(levels - 0.5))]
    lo = pred_q[:, np.argmin(np.abs(levels - 0.1))]
    hi = pred_q[:, np.argmin(np.abs(levels - 0.9))]
    err = median - y

    out = {
        "crps": float(crps),
        "pinball": float(mf.pinball_loss(pred_q, y, levels).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "coverage_80": float(mf.interval_coverage(lo, hi, y)),
        "sharpness_80": float(mf.sharpness(lo, hi)),
        "calibration_error": float(mf.calibration_error(pred_q, y, levels)),
    }
    if da_price is not None:
        # Indicative only — proxy imbalance price, not the published one.
        spread = np.abs(da_price) * PROXY_IMBALANCE_SPREAD
        out["imbalance_cost_proxy"] = float((np.abs(err) * spread).sum())
    return out


def build_models(train: pd.DataFrame, val: pd.DataFrame, df: pd.DataFrame,
                 levels: np.ndarray) -> dict:
    """Fit every baseline on train (+val for early stopping where useful)."""
    feats = sl.feature_columns(df)
    models = {
        "persistence": Persistence(),
        "climatology": Climatology(),
        "physical": physical_for_slice(df.attrs["solar"]),
        "tso": TSOForecast(),
        "tso_debias": TSODebiased(),
    }
    fitted = {}
    for name, m in models.items():
        print(f"  fitting {name}…", flush=True)
        fitted[name] = m.fit(train)

    print("  fitting gbm_quantile (one model per level)…", flush=True)
    gbm = GBMQuantile(features=feats, levels=levels)
    gbm.fit(train, val=val)
    fitted["gbm_quantile"] = gbm

    # Two different commercial claims, so both get scored:
    #   gbm_quantile  — replace the TSO forecast using only public weather.
    #   gbm_plus_tso  — post-process it, i.e. add calibrated uncertainty and a
    #                   correction on top of the incumbent. Weaker as a
    #                   scientific claim, but it is the one a trading desk can
    #                   actually buy, since the TSO forecast is free and public.
    print("  fitting gbm_plus_tso (TSO forecast as a feature)…", flush=True)
    gbm2 = GBMQuantile(features=feats + ["tso_forecast"], levels=levels)
    gbm2.fit(train, val=val)
    fitted["gbm_plus_tso"] = gbm2
    return fitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="de_wind", choices=sorted(sl.SLICES))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--source", default="day_ahead",
                    choices=["day_ahead", "hist_fc", "archive"])
    ap.add_argument("--lead-days", type=int, default=2,
                    help="NWP forecast lead: 2 = no leakage (headline), "
                         "1 = optimistic upper bound (leaks for late hours)")
    ap.add_argument("--all-hours", action="store_true",
                    help="score solar over night hours too (default: daylight only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = DEFAULT_LEVELS
    df = sl.build(args.slice, args.start, args.end, source=args.source,
                  lead_days=args.lead_days)
    split = bd.temporal_split(df)
    train, val, test = split["train"], split["val"], split["test"]

    daylight_only = df.attrs["solar"] and not args.all_hours
    test_rows = test[test["daylight"]] if daylight_only else test

    print(f"\n{df.attrs['label']} — weather source: {args.source} "
          f"(lead {df.attrs['lead_days']}d)")
    print(f"  train {len(train)} ({train.index[0].date()}→{train.index[-1].date()})  "
          f"val {len(val)}  test {len(test)} "
          f"({test.index[0].date()}→{test.index[-1].date()})")
    print(f"  scoring on {len(test_rows)} test rows"
          f"{' (daylight only)' if daylight_only else ''}\n")

    fitted = build_models(train, val, df, levels)

    y = test_rows["y"].to_numpy(dtype=float)
    price = test_rows["da_price"].to_numpy(float) if "da_price" in test_rows else None

    results = {}
    for name, m in fitted.items():
        results[name] = score(m.predict_quantiles(test_rows, levels), y, levels, price)

    ref = results["tso"]["crps"]
    for r in results.values():
        r["crps_skill_vs_tso"] = float(1.0 - r["crps"] / ref)

    order = sorted(results, key=lambda k: results[k]["crps"])
    hdr = (f"\n{'model':14s} {'CRPS':>8s} {'skill':>7s} {'RMSE':>8s} {'MAE':>8s} "
           f"{'bias':>8s} {'cov80':>6s} {'sharp80':>8s} {'calib':>6s}")
    print(hdr); print("-" * len(hdr))
    for name in order:
        r = results[name]
        star = " ←TSO" if name == "tso" else ""
        print(f"{name:14s} {r['crps']:8.1f} {r['crps_skill_vs_tso']:+6.1%} "
              f"{r['rmse']:8.1f} {r['mae']:8.1f} {r['bias']:+8.1f} "
              f"{r['coverage_80']:5.1%} {r['sharpness_80']:8.1f} "
              f"{r['calibration_error']:6.3f}{star}")

    best = order[0]
    gbm_skill = results["gbm_quantile"]["crps_skill_vs_tso"]
    plus_skill = results["gbm_plus_tso"]["crps_skill_vs_tso"]
    print(f"\nmean generation on scored rows: {y.mean():.0f} MW")
    print(f"CRPS is in MW; 'skill' is 1 - CRPS/CRPS_tso (positive = better than the TSO).")
    print(f"coverage_80 should be ≈80%; calib is RMS deviation of coverage from nominal.")

    def verdict(s):
        return "BEATS" if s > 0 else "does NOT beat"
    print(f"\nFAIL-FAST (replace the TSO, public weather only): "
          f"gbm_quantile {verdict(gbm_skill)} it on CRPS ({gbm_skill:+.1%}).")
    print(f"FAIL-FAST (post-process the TSO): "
          f"gbm_plus_tso {verdict(plus_skill)} it on CRPS ({plus_skill:+.1%}).")
    print(f"Best overall: {best}.")

    out = args.out or (f"reports/leaderboard_{args.slice}_{args.source}"
                       f"_lead{df.attrs['lead_days']}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "slice": args.slice, "label": df.attrs["label"], "source": args.source,
        "lead_days": df.attrs["lead_days"],
        "daylight_only": bool(daylight_only),
        "window": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "n_train": len(train), "n_val": len(val), "n_test_scored": len(test_rows),
        "mean_generation_mw": float(y.mean()),
        "levels": levels.tolist(),
        "economic_note": "imbalance_cost_proxy uses a proxy spread, not published "
                         "imbalance prices — indicative only",
        "results": results,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
