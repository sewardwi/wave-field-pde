"""
FieldOperator — the wave-field operator as a deterministic PDE regressor.

This is the Phase-1 model: the SAME wave machinery the diffusion denoiser uses
(WaveFieldBlock + 2D radial WaveFieldAttention2D, all reused unchanged from
wave_field/), wired as a plain operator u = G_θ(a) with no diffusion. It is the
honest, well-posed head-to-head against FNO: does the FFT damped-oscillator
operator match a real FNO on a PDE field at matched budget?

Design choices from the design review (docs/final-plan.md):
  - patch_size ∈ {1, 2}. At P=1 the "patchify" is a pointwise channel lift (like
    FNO's fc0) and the conv runs at FULL field resolution — the fair, FNO-matched
    setting. Larger P coarsens the conv grid (a capacity handicap vs FNO).
  - NO positional embedding: the circular conv is translation-equivariant, a good
    prior for homogeneous turbulence that FNO gets for free. Adding absolute
    pos-embeds would break it.

Conditioning (the WaveFieldBlock needs a conditioning embedding either way):
  - cond_mode='none'    — a single learned constant embedding → a static operator
                          (the kernel/FiLM modulation collapses to a learned
                          constant, i.e. an effectively static kernel).
  - cond_mode='physics' — embed the physical viscosity log10(ν) (and optional
                          horizon) → the kernel adapts to the physical *regime*.
                          The novel, review-flagged angle; only meaningful on a
                          MULTI-viscosity dataset (on a single ν it is constant).

Forward: (B, in_ch, H, W) → (B, out_ch, H, W), where in_ch=t_in, out_ch=t_out.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from wave_field.blocks import WaveFieldBlock, FinalLayer


class FieldOperator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int | tuple[int, int] = 64,
        patch_size: int = 1,
        dim: int = 192,
        depth: int = 4,
        num_heads: int = 8,
        timestep_dim: int = 128,
        cond_mode: str = "none",
        dynamic_filter: bool = False,
        gating: str = "pointwise",
        aniso_kernel: bool = False,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        assert cond_mode in ("none", "physics")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.cond_mode = cond_mode

        self.ph, self.pw = H // patch_size, W // patch_size
        self.num_patches = self.ph * self.pw
        self.in_patch_dim = in_channels * patch_size * patch_size
        self.out_patch_dim = out_channels * patch_size * patch_size

        # Pointwise (P=1) or patch lift; NO positional embedding (equivariance).
        self.patch_embed = nn.Linear(self.in_patch_dim, dim)

        # Conditioning embedding fed to every block. Blocks always run in
        # 'physics' mode so the kernel-modulation pathway is available; the
        # embedding source is what makes it static vs regime-adaptive.
        if cond_mode == "none":
            # A single learned constant → effectively-static kernel/FiLM.
            self.null_emb = nn.Parameter(torch.zeros(timestep_dim))
        else:
            # log10(ν) (+ optional horizon later) → timestep_dim embedding.
            self.phys_mlp = nn.Sequential(
                nn.Linear(1, timestep_dim), nn.SiLU(),
                nn.Linear(timestep_dim, timestep_dim),
            )

        self.blocks = nn.ModuleList([
            WaveFieldBlock(
                dim=dim, num_heads=num_heads, seq_len=self.num_patches,
                timestep_dim=timestep_dim, conditioning="physics",
                use_2d_kernel=True, height=self.ph, width=self.pw,
                dynamic_filter=dynamic_filter, gating=gating, aniso_kernel=aniso_kernel,
            )
            for _ in range(depth)
        ])
        self.final = FinalLayer(dim, self.out_patch_dim, timestep_dim, conditioning="physics")

        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    # ------------------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, C*P*P)."""
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, self.ph, P, self.pw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, self.num_patches, C * P * P)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, out_C*P*P) → (B, out_C, H, W)."""
        B = x.shape[0]
        P, C = self.patch_size, self.out_channels
        x = x.view(B, self.ph, self.pw, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, self.ph * P, self.pw * P)

    def _cond(self, x: torch.Tensor, visc: torch.Tensor | None) -> torch.Tensor:
        """Build the (B, timestep_dim) conditioning embedding."""
        B = x.shape[0]
        if self.cond_mode == "none":
            return self.null_emb[None].expand(B, -1)
        assert visc is not None, "cond_mode='physics' requires viscosity"
        if visc.dim() == 0:
            visc = visc.expand(B)
        feat = (torch.log10(visc.float()).clamp(-8, 0) + 4.0)[:, None]   # ~O(1), centered near ν=1e-4
        return self.phys_mlp(feat)

    def forward(self, x: torch.Tensor, visc: torch.Tensor | None = None) -> torch.Tensor:
        c = self._cond(x, visc)
        h = self.patch_embed(self._patchify(x))
        for block in self.blocks:
            h = block(h, c)
        return self._unpatchify(self.final(h, c))

    def param_count(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())


if __name__ == "__main__":
    # Shape + param-count smoke.
    for P in (1, 2):
        m = FieldOperator(in_channels=10, out_channels=2, image_size=64,
                          patch_size=P, dim=192, depth=4)
        x = torch.randn(3, 10, 64, 64)
        y = m(x)
        assert y.shape == (3, 2, 64, 64), y.shape
        print(f"P={P}: out {tuple(y.shape)}  params {m.param_count():,}")
    # physics conditioning path
    mp = FieldOperator(10, 2, cond_mode="physics", patch_size=2, dim=128, depth=3)
    y = mp(torch.randn(2, 10, 64, 64), visc=torch.tensor([1e-3, 1e-4]))
    print(f"physics-cond: out {tuple(y.shape)}  params {mp.param_count():,}")
    print("field_operator smoke PASSED")
