"""
Unified field-operator trainer — the Phase-1 wave-vs-FNO head-to-head.

Both architectures run through the *identical* loop (same rel-L2 loss, AdamW,
cosine LR, epochs, batch size) so the only differences are the operator and the
parameter count. That is the fairness protocol from docs/final-plan.md: matched
params AND matched steps, reproduce-able, no FAST preset.

    python train_field.py --data data/ns_V1e-3.pt --arch fno  --save_dir outputs/ns_fno
    python train_field.py --data data/ns_V1e-3.pt --arch wave --save_dir outputs/ns_wave

Reports relative L2 + spectrum error + correlation on the held-out test split,
writes config.json / metrics.json / pred.png into the run dir.

--mode is 'regression' for Phase 1 (deterministic). Diffusion mode is Phase 2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.navier_stokes import load_ns_splits
from metrics.field import relative_l2, field_metrics


def build_model(args, in_ch: int, out_ch: int):
    if args.arch == "fno":
        from baselines.fno import FNO2d
        return FNO2d(in_ch, out_ch, modes=args.modes, width=args.width,
                     n_layers=args.layers)
    if args.arch == "wave":
        from models.field_operator import FieldOperator
        return FieldOperator(in_ch, out_ch, image_size=args.res,
                             patch_size=args.patch_size, dim=args.dim,
                             depth=args.depth, num_heads=args.num_heads,
                             cond_mode=args.cond, dynamic_filter=args.dynamic_filter,
                             gating=args.gating)
    raise ValueError(args.arch)


@torch.no_grad()
def evaluate(model, ds, device, batch_size: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targs = [], []
    for x, y in DataLoader(ds, batch_size=batch_size):
        preds.append(model(x.to(device)).cpu())
        targs.append(y)
    if not preds:
        return {}, None, None
    pred, targ = torch.cat(preds), torch.cat(targs)
    return field_metrics(pred, targ), pred, targ


def save_pred_png(pred, targ, path, stats):
    """One test sample: target / prediction / error for the last predicted slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = stats.denormalize(pred[0, -1]).numpy()
    t = stats.denormalize(targ[0, -1]).numpy()
    err = p - t
    vmax = max(abs(t).max(), abs(p).max())
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, title, vm in [(ax[0], t, "target", vmax), (ax[1], p, "prediction", vmax),
                              (ax[2], err, "error", abs(err).max())]:
        im = a.imshow(img, cmap="RdBu_r", vmin=-vm, vmax=vm)
        a.set_title(title); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", required=True, choices=["wave", "fno"])
    ap.add_argument("--mode", default="regression", choices=["regression"])
    ap.add_argument("--save_dir", default="outputs/field_run")
    # task
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--t-in", dest="t_in", type=int, default=10)
    ap.add_argument("--t-out", dest="t_out", type=int, default=None)
    # wave hyperparams
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=1)
    ap.add_argument("--cond", default="none", choices=["none", "physics"])
    ap.add_argument("--dynamic_filter", action="store_true")
    ap.add_argument("--gating", default="pointwise", choices=["pointwise", "hyena"])
    # fno hyperparams
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    # optim
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    splits = load_ns_splits(args.data, t_in=args.t_in, t_out=args.t_out, seed=args.seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    in_ch, out_ch = tr.in_channels, tr.out_channels
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)

    model = build_model(args, in_ch, out_ch).to(device)
    n_params = model.param_count() if hasattr(model, "param_count") else \
        sum(p.numel() for p in model.parameters())
    steps_per_epoch = max(1, len(tr) // args.batch_size)
    print(f"[{args.arch}] params={n_params:,}  in={in_ch} out={out_ch}  "
          f"train={len(tr)} val={len(va)} test={len(te)}  "
          f"steps/epoch={steps_per_epoch} total_steps={steps_per_epoch * args.epochs}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    log = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = relative_l2(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        sched.step()
        train_rl2 = tot / n

        if ep % args.log_every == 0 or ep == args.epochs:
            vm, _, _ = evaluate(model, va, device, args.batch_size)
            vr = vm.get("rel_l2", float("nan"))
            best_val = min(best_val, vr) if vr == vr else best_val
            print(f"  ep {ep:4d}  train {train_rl2:.4f}  val {vr:.4f}  "
                  f"({time.time() - t0:.0f}s)")
            log.append({"epoch": ep, "train_rel_l2": train_rl2, "val_rel_l2": vr})

    # Final test evaluation.
    tm, pred, targ = evaluate(model, te, device, args.batch_size)
    print(f"[{args.arch}] TEST  rel_l2={tm.get('rel_l2', float('nan')):.4f}  "
          f"spec_err={tm.get('spectrum_err', float('nan')):.4f}  "
          f"spec_err_log={tm.get('spectrum_err_log', float('nan')):.4f}  "
          f"corr={tm.get('correlation', float('nan')):.4f}")

    (save_dir / "config.json").write_text(json.dumps({**vars(args), "n_params": n_params,
                                                      "total_steps": steps_per_epoch * args.epochs}, indent=2))
    (save_dir / "metrics.json").write_text(json.dumps(
        {"test": tm, "best_val_rel_l2": best_val, "history": log, "n_params": n_params}, indent=2))
    if pred is not None:
        save_pred_png(pred, targ, save_dir / "pred.png", te.stats)
    print(f"  wrote {save_dir}/metrics.json, config.json, pred.png")


if __name__ == "__main__":
    main()
