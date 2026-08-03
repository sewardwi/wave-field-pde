"""
2D Navier–Stokes data generator — the FNO benchmark, self-contained.

Reproduces the data-generation scheme of Li et al. 2020 ("Fourier Neural
Operator for Parametric PDEs") so we own the pipeline (no fragile Google-drive
`.mat` links) and can reproduce FNO's published numbers on *our* data — which
rule #1 (sanity-check the baseline) requires.

Physics: 2D incompressible Navier–Stokes in vorticity form on the unit torus
[0,1]² with periodic BC:

    ∂_t w + u·∇w = ν Δw + f,      u = ∇^⊥ ψ,   Δψ = -w,   ∇·u = 0

with the fixed forcing f(x,y) = 0.1·(sin(2π(x+y)) + cos(2π(x+y))). The initial
vorticity is drawn from the Gaussian random field N(0, 7^{3/2}(-Δ+49I)^{-2.5}).

Solver: pseudo-spectral in space (FFT), Crank–Nicolson on the viscous term,
explicit on the advection, 2/3-rule dealiasing. Batched over the sample axis so
many trajectories integrate at once (the big speedup on GPU/MPS).

CLI:
    python -m datasets.ns_generator --n 1000 --res 64 --visc 1e-3 \
        --T 50 --dt 1e-4 --record-steps 50 --out data/ns_V1e-3_N1000.pt
Add --preview to also write a vorticity+spectrum PNG for an eyeball check.

The reference ν=1e-3 setup (T=50, dt=1e-4, 64², record 50 snapshots at t=1..50)
is what the deterministic Phase-1 wave-vs-FNO comparison trains on.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Gaussian random field initial condition (matches FNO's GaussianRF, 2D)
# ---------------------------------------------------------------------------

class GaussianRF:
    """
    Periodic Gaussian random field on an S×S torus with covariance
    σ²(-Δ + τ²I)^{-α}. The FNO NS default (alpha=2.5, tau=7) gives the operator
    7^{3/2}(-Δ+49I)^{-2.5}. sqrt_eig holds the sqrt of the covariance eigenvalues
    on the Fourier grid; a sample is ifft(sqrt_eig · white complex noise).
    """

    def __init__(self, size: int, alpha: float = 2.5, tau: float = 7.0,
                 sigma: float | None = None, device=None):
        self.size = size
        self.device = device
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - 2.0))     # dim = 2

        k_max = size // 2
        wn = torch.cat((torch.arange(0, k_max, device=device),
                        torch.arange(-k_max, 0, device=device)), 0).repeat(size, 1)
        k_y = wn
        k_x = wn.transpose(0, 1)
        # sqrt of covariance eigenvalues; the size² factor matches the (unnormalized)
        # torch ifft convention used in .sample().
        self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * \
            ((4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
        self.sqrt_eig[0, 0] = 0.0                        # zero-mean field

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw n fields → (n, S, S) real vorticity."""
        # real/imag each ~ N(0,1) (FNO convention), so E|coeff|² = 2 per mode.
        noise = torch.randn(n, self.size, self.size, 2, device=self.device,
                            generator=generator)
        coeff = torch.view_as_complex(noise) * self.sqrt_eig
        return torch.fft.ifft2(coeff).real


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

def forcing(size: int, device=None) -> torch.Tensor:
    """FNO's fixed forcing f = 0.1·(sin(2π(x+y)) + cos(2π(x+y))) on the torus."""
    t = torch.linspace(0, 1, size + 1, device=device)[:-1]
    x, y = torch.meshgrid(t, t, indexing="ij")
    return 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))


# ---------------------------------------------------------------------------
# Pseudo-spectral Crank–Nicolson solver (batched over samples)
# ---------------------------------------------------------------------------

@torch.no_grad()
def navier_stokes_2d(w0: torch.Tensor, f: torch.Tensor, visc: float,
                     T: float, dt: float, record_steps: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate 2D NS vorticity from w0 for time T, recording `record_steps`
    evenly-spaced snapshots.

    Args:
        w0:            (N, S, S) initial vorticity.
        f:             (S, S) forcing (broadcast over the batch).
        visc:          viscosity ν.
        T, dt:         final time and solver step.
        record_steps:  number of snapshots to save (at t = T/rec, 2T/rec, …, T).
    Returns:
        sol:   (N, record_steps, S, S) vorticity snapshots.
        sol_t: (record_steps,) snapshot times.
    """
    device = w0.device
    N, S = w0.shape[0], w0.shape[-1]
    k_max = S // 2
    steps = math.ceil(T / dt)
    record_time = max(1, steps // record_steps)

    w_h = torch.fft.rfft2(w0)                            # (N, S, S//2+1)
    f_h = torch.fft.rfft2(f)
    if f_h.dim() < w_h.dim():
        f_h = f_h.unsqueeze(0)

    # Wavenumbers on the rfft2 grid: dim 0 (rows) full, dim 1 (cols) half.
    wn = torch.cat((torch.arange(0, k_max, device=device),
                    torch.arange(-k_max, 0, device=device)), 0).repeat(S, 1)
    k_y = wn[..., :k_max + 1]
    k_x = wn.transpose(0, 1)[..., :k_max + 1]
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0                                     # avoid 0/0 for the mean mode
    dealias = ((k_x.abs() <= (2.0 / 3.0) * k_max) &
               (k_y.abs() <= (2.0 / 3.0) * k_max)).float().unsqueeze(0)

    sol = torch.zeros(N, record_steps, S, S, device=device)
    sol_t = torch.zeros(record_steps, device=device)
    c = 0
    t = 0.0
    for j in range(steps):
        # Stream function ψ̂ = ŵ / (-Δ), velocities u = ∇^⊥ ψ.
        psi_h = w_h / lap
        u = torch.fft.irfft2(2 * math.pi * 1j * k_y * psi_h, s=(S, S))    # ∂ψ/∂y
        v = torch.fft.irfft2(-2 * math.pi * 1j * k_x * psi_h, s=(S, S))   # -∂ψ/∂x
        w_x = torch.fft.irfft2(2 * math.pi * 1j * k_x * w_h, s=(S, S))
        w_y = torch.fft.irfft2(2 * math.pi * 1j * k_y * w_h, s=(S, S))
        # Nonlinear advection u·∇w in physical space → spectral, dealiased.
        adv_h = dealias * torch.fft.rfft2(u * w_x + v * w_y)
        # Crank–Nicolson on the viscous term, explicit on advection + forcing.
        w_h = (-dt * adv_h + dt * f_h + (1.0 - 0.5 * dt * visc * lap) * w_h) \
            / (1.0 + 0.5 * dt * visc * lap)
        t += dt
        if (j + 1) % record_time == 0 and c < record_steps:
            sol[:, c] = torch.fft.irfft2(w_h, s=(S, S))
            sol_t[c] = t
            c += 1
    return sol, sol_t


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(n: int, res: int, visc: float, T: float, dt: float,
             record_steps: int, batch: int = 64, device: str = "cpu",
             seed: int = 0) -> dict:
    """
    Generate `n` NS trajectories, batched. Returns a dict with the trajectory
    tensor (n, record_steps, res, res) on CPU plus the generation metadata.
    """
    dev = torch.device(device)
    grf = GaussianRF(res, device=dev)
    f = forcing(res, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    out = torch.empty(n, record_steps, res, res)
    times = None
    done = 0
    while done < n:
        b = min(batch, n - done)
        w0 = grf.sample(b, generator=gen)
        sol, sol_t = navier_stokes_2d(w0, f, visc, T, dt, record_steps)
        out[done:done + b] = sol.cpu()
        times = sol_t.cpu()
        done += b
        print(f"  generated {done}/{n} trajectories", flush=True)

    return {
        "data": out,                                   # (n, record_steps, res, res)
        "times": times,                                # (record_steps,)
        "meta": {"visc": visc, "T": T, "dt": dt, "res": res,
                 "record_steps": record_steps, "n": n, "seed": seed},
    }


def _preview(bundle: dict, path: str) -> None:
    """Save a vorticity snapshot + radial energy spectrum for an eyeball check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from metrics.field import radial_spectrum

    data = bundle["data"]
    w_last = data[0, -1]                               # final-time vorticity of traj 0
    spec, k = radial_spectrum(data[:, -1])             # mean spectrum at final time
    spec = spec.mean(0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im = ax[0].imshow(w_last, cmap="RdBu_r")
    ax[0].set_title(f"vorticity  t={bundle['times'][-1]:.1f}  ν={bundle['meta']['visc']:g}")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].loglog(k[1:], spec[1:])
    ax[1].set_xlabel("|k|"); ax[1].set_ylabel("E(k)"); ax[1].set_title("mean energy spectrum")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"  wrote preview → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 2D Navier–Stokes (FNO benchmark) data.")
    ap.add_argument("--n", type=int, default=1000, help="number of trajectories")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--visc", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--dt", type=float, default=1e-4, help="solver step (reference: 1e-4)")
    ap.add_argument("--record-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64, help="trajectories per solver batch")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/ns.pt")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    print(f"Generating {args.n} NS trajectories: res={args.res} ν={args.visc:g} "
          f"T={args.T} dt={args.dt:g} record={args.record_steps} on {args.device}")
    bundle = generate(args.n, args.res, args.visc, args.T, args.dt,
                      args.record_steps, args.batch, args.device, args.seed)

    d = bundle["data"]
    print(f"done. data {tuple(d.shape)}  mean={d.mean():.3e}  std={d.std():.3e}  "
          f"finite={torch.isfinite(d).all().item()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"  saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if args.preview:
        _preview(bundle, str(out.with_suffix(".png")))


if __name__ == "__main__":
    main()
