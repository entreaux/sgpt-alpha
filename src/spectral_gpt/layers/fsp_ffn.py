# ── src/spectral_gpt/layers/fsp_ffn.py ─────────────────────────────────────────
"""
Fractional‑Spectral Projection Feed‑Forward Network (FSP‑FFN)
Hardened v0.2.1 – MPS‑safe

Changes vs v0.2
---------------
* **Conditional spectral‑norm** – `torch.nn.utils.parametrizations.spectral_norm`
  is applied only when the current backend supports `torch.vdot`
  (CUDA & CPU); on M‑series GPUs we silently skip SN to avoid the
  `NotImplementedError`.
* Keeps all previous numerical‑stability upgrades (safe init, clamps, LN).

Set the environment variable `SGPT_SN_FORCE=1` if you still want SN on an
MPS run (requires `PYTORCH_ENABLE_MPS_FALLBACK=1`, which runs the power
iteration on CPU and is ≈5–10 % slower).
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Helper: detect whether the backend can run spectral_norm without hitting
# aten::vdot issues (MPS as of PyTorch 2.3 cannot).
# -----------------------------------------------------------------------------

def _spectral_norm_supported() -> bool:
    """Return True if the active backend implements torch.vdot."""
    if os.getenv("SGPT_SN_FORCE", "0") == "1":
        return True  # caller forces SN regardless of backend
    return not torch.backends.mps.is_available()


# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------

class FSPFFN(nn.Module):
    r"""
    Fractional–Spectral Projection FFN

        x ∈ ℝ^{B×L×d}
          —P→ u  ∈ ℝ^{B×L×r}          (learnable low‑rank projection)
          —ϕ→ z  ∈ ℝ^{B×L×r}          (fractional power per‑rank ω_j)
          —LN→                          (rank‑wise variance control)
          —Q→ y  ∈ ℝ^{B×L×d}          (reconstruction)

    Forward maths (per element):
        u  = x P
        z  = exp( ω · ln|u| − lnΓ(ω+1) ) * sign(u) * A
        y  = Q( LayerNorm(z) )

    Parameters
    ----------
    d_model : int
        Model hidden size.
    r : int
        Low‑rank bottleneck size.
    omega_min, omega_max : float
        Range clamp for fractional exponent ω.
    eps : float
        Clamp for log safety.
    spectral_norm : bool
        Whether to wrap P and Q with 1‑Lipschitz spectral normalisation
        **when supported by the backend**.
    """

    def __init__(
        self,
        d_model: int,
        r: int,
        *,
        omega_min: float = 0.3,
        omega_max: float = 1.7,
        eps: float = 1e-6,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.r = r
        self.eps = eps
        self.register_buffer("_omega_min", torch.tensor(omega_min))
        self.register_buffer("_omega_max", torch.tensor(omega_max))

        # ---------------- low‑rank projections ----------------
        self.P = nn.Linear(d_model, r,  bias=True)
        self.Q = nn.Linear(r,       d_model, bias=False)

        if spectral_norm and _spectral_norm_supported():
            self.P = nn.utils.parametrizations.spectral_norm(self.P)
            self.Q = nn.utils.parametrizations.spectral_norm(self.Q)

        # ---------------- learnable spectral params -----------
        # Initialise ω around 1.0 inside the [min,max] range
        init_frac = (1.0 - omega_min) / (omega_max - omega_min)
        init_raw  = math.log(init_frac) - math.log1p(-init_frac)  # logit
        self._raw_omega = nn.Parameter(torch.full((r,), init_raw))
        self.A          = nn.Parameter(torch.ones(r) / r)

        # Normalisation over the rank dimension
        self.ln = nn.LayerNorm(r)

    # ------------------------------------------------------------------ utils
    def _omega(self) -> torch.Tensor:
        """Sigmoid‑map raw parameter → [omega_min, omega_max]."""
        sig = torch.sigmoid(self._raw_omega)
        return self._omega_min + (self._omega_max - self._omega_min) * sig

    # ------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,d)
        u = self.P(x)                                    # (B,L,r)
        ω = self._omega()                                # (r,)

        # Fractional power in log‑space
        sign_u   = torch.sign(u)
        log_safe = torch.clamp(u.abs(), min=self.eps).log()
        log_gamma = torch.lgamma(ω + 1.0)                # (r,)

        # Broadcast ω and lnΓ over (B,L)
        ω_b, lg_b = ω.view(1, 1, -1), log_gamma.view(1, 1, -1)
        z = torch.exp(ω_b * log_safe - lg_b) * sign_u    # (B,L,r)
        z = z * self.A                                   # amplitude gate

        z = self.ln(z)
        return self.Q(z)                                 # (B,L,d)
