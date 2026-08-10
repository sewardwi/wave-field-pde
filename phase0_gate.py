"""
Phase-0 gate — can we reproduce the TSO's own forecast error from raw public data?

This is the decision gate from docs/final-plan.md: *before* any modeling, prove the
data spine is right by recovering the incumbent forecast's error against actuals and
checking it against what day-ahead forecasting is publicly known to achieve
(solar ≈ 3–5% of peak, wind ≈ 5–10%). If our numbers don't line up with reality,
the pipeline is wrong — fix it before modeling. That's rule #1 of this repo.

It also runs a **scope check** that turned out to matter more than the error itself.
Some zones publish an "actual generation" aggregate and a "day-ahead forecast" that
cover *different fleets*, which shows up as a multiplicative mismatch:

    actual ≈ slope · forecast,   slope far from 1

DK (slope 1.25) and FR (slope 0.87) are both mis-scoped, in opposite directions;
DE and ES are clean. This matters commercially, not just cosmetically: on a
mis-scoped zone you can "beat the TSO forecast" by 10%+ purely by rescaling it,
which would be exactly the kind of fake win this project exists not to publish.
Zones that fail the scope check are unusable as a headline benchmark.

    python phase0_gate.py                 # the two clean slices, 3 years
    python phase0_gate.py --all           # + the known mis-scoped control zones
    python phase0_gate.py --years 1

Writes reports/phase0_gate.json. Exit status is nonzero if a headline slice fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import energy_charts as ec

# Headline slices are the clean ones; the control slices are kept visible on
# purpose so the scope finding is reproducible rather than folklore.
SLICES = [
    dict(label="ES solar", country="es", kind="solar", bzn="ES",
         band=(0.02, 0.08), headline=True),
    dict(label="DE wind", country="de", kind="wind", bzn="DE-LU",
         band=(0.03, 0.12), headline=True),
    dict(label="DK wind", country="dk", kind="wind", bzn="DK1",
         band=(0.03, 0.12), headline=False),
    dict(label="FR wind", country="fr", kind="wind", bzn="FR",
         band=(0.03, 0.12), headline=False),
    dict(label="DE solar", country="de", kind="solar", bzn="DE-LU",
         band=(0.02, 0.08), headline=False),
]

MAX_SCOPE_SLOPE_DEV = 0.05      # |slope - 1| beyond this ⇒ fleets don't match
MAX_ABS_BIAS_FRAC = 0.05        # |bias| / mean actual
MIN_COVERAGE = 0.95


def evaluate_slice(country: str, kind: str, start: str, end: str) -> dict:
    """Point-forecast error + scope diagnostics for one country/technology."""
    if kind == "wind":
        actual, fc = (ec.wind_actual(country, start, end),
                      ec.wind_forecast(country, start, end))
    else:
        actual, fc = (ec.generation_actual(country, start, end, kind),
                      ec.tso_forecast(country, start, end, kind))

    joined = pd.concat({"actual": actual, "forecast": fc}, axis=1)
    n_rows = len(joined)
    d = joined.dropna()
    if len(d) < 24 * 30:
        return {"error": f"only {len(d)} aligned rows", "n_aligned": len(d)}

    a, f = d["actual"].to_numpy(), d["forecast"].to_numpy()
    err = f - a
    peak = float(a.max())
    slope, intercept = np.polyfit(f, a, 1)

    return {
        "n_aligned": int(len(d)),
        "coverage": float(len(d) / n_rows),
        "start": str(d.index[0]), "end": str(d.index[-1]),
        "mean_actual_mw": float(a.mean()),
        "peak_actual_mw": peak,
        "mae_mw": float(np.abs(err).mean()),
        "rmse_mw": float(np.sqrt((err ** 2).mean())),
        # Normalised by peak *observed* generation (not installed capacity —
        # we don't pull capacity here, so don't call it capacity).
        "nrmse_pct_peak": float(np.sqrt((err ** 2).mean()) / peak),
        "nmae_pct_peak": float(np.abs(err).mean() / peak),
        "bias_mw": float(err.mean()),
        "bias_frac_mean": float(err.mean() / a.mean()),
        "scope_slope": float(slope),
        "scope_intercept_mw": float(intercept),
        "corr": float(np.corrcoef(f, a)[0, 1]),
    }


def judge(res: dict, band: tuple[float, float]) -> dict:
    """Apply the gate's pass/fail criteria to one slice's stats."""
    if "error" in res:
        return {"pass": False, "reasons": [res["error"]]}
    reasons = []
    if res["coverage"] < MIN_COVERAGE:
        reasons.append(f"coverage {res['coverage']:.1%} < {MIN_COVERAGE:.0%}")
    if abs(res["scope_slope"] - 1) > MAX_SCOPE_SLOPE_DEV:
        reasons.append(f"scope mismatch: actual ≈ {res['scope_slope']:.3f}·forecast")
    if abs(res["bias_frac_mean"]) > MAX_ABS_BIAS_FRAC:
        reasons.append(f"bias {res['bias_frac_mean']:+.1%} of mean")
    lo, hi = band
    if not (lo <= res["nrmse_pct_peak"] <= hi):
        reasons.append(f"nRMSE {res['nrmse_pct_peak']:.1%} outside plausible {lo:.0%}–{hi:.0%}")
    return {"pass": not reasons, "reasons": reasons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: last full year end)")
    ap.add_argument("--all", action="store_true", help="include known mis-scoped control zones")
    ap.add_argument("--out", default="reports/phase0_gate.json")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=args.years)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    chosen = SLICES if args.all else [x for x in SLICES if x["headline"]]
    print(f"Phase-0 gate — TSO day-ahead forecast vs actuals, {s} → {e}\n")

    out, failures = {}, []
    for spec in chosen:
        print(f"[{spec['label']}] pulling…", flush=True)
        try:
            res = evaluate_slice(spec["country"], spec["kind"], s, e)
        except Exception as ex:                       # a dead series shouldn't kill the run
            res = {"error": f"{type(ex).__name__}: {ex}"}
        verdict = judge(res, spec["band"])
        out[spec["label"]] = {**res, **verdict, "headline": spec["headline"]}
        if spec["headline"] and not verdict["pass"]:
            failures.append(spec["label"])

    hdr = (f"\n{'slice':10s} {'n':>6s} {'mean':>8s} {'nRMSE':>7s} {'nMAE':>7s} "
           f"{'bias':>8s} {'slope':>6s} {'r':>6s}  verdict")
    print(hdr); print("-" * len(hdr))
    for label, r in out.items():
        if "error" in r:
            print(f"{label:10s} {'—':>6s}  ERROR: {r['error'][:60]}")
            continue
        flag = "PASS" if r["pass"] else "FAIL: " + "; ".join(r["reasons"])
        print(f"{label:10s} {r['n_aligned']:6d} {r['mean_actual_mw']:8.0f} "
              f"{r['nrmse_pct_peak']:6.2%} {r['nmae_pct_peak']:6.2%} "
              f"{r['bias_frac_mean']:+7.1%} {r['scope_slope']:6.3f} {r['corr']:6.3f}  {flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"window": {"start": s, "end": e}, "slices": out}, indent=2))
    print(f"\nwrote {args.out}")

    if failures:
        print(f"\nGATE FAILED on headline slice(s): {', '.join(failures)}")
        return 1
    print("\nGATE PASSED — headline slices reproduce plausible TSO day-ahead error; "
          "data spine is trustworthy for Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
