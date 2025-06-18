import torch
import torch.nn as nn
from torch.special import gamma

class FSPFFN(nn.Module):
    """
    Fractional Spectral Projection Feed-Forward Network (FSP-FFN).

    Args:
        d_model (int): hidden dimension
        r (int): low-rank projection dimension (<< d_model)
        omega_min (float): minimum fractional exponent
        omega_max (float): maximum fractional exponent
    """
    def __init__(self, d_model: int, r: int, omega_min: float = 0.5, omega_max: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.r = r
        self.omega_min = omega_min
        self.omega_max = omega_max

        # Low-rank projections P: d_model -> r, Q: r -> d_model
        self.P = nn.Linear(d_model, r, bias=False)
        self.Q = nn.Linear(r, d_model, bias=False)

        # Spectral kernel parameters: A_j and alpha_j per kernel
        # A: (r, r) where row j is A_j
        self.A = nn.Parameter(torch.randn(r, r))
        # alpha: (r,) scalar mixing weights
        self.alpha = nn.Parameter(torch.ones(r))

        # Fractional exponents omega_j
        self.omega = nn.Parameter(
            torch.rand(r) * (omega_max - omega_min) + omega_min
        )

        # Output bias
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        U = self.P(x)  # (B, L, r)

        # Compute Gamma(omega_j + 1) for normalization: shape (1,1,r)
        gamma_vals = gamma(self.omega + 1).view(1, 1, -1)

        # Fractional spectral expansion: V_j = U**omega_j / Gamma(omega_j+1)
        V = U.pow(self.omega.view(1, 1, -1)) / gamma_vals  # (B, L, r)

        # Prepare for mixing: expand dims
        V_exp = V.unsqueeze(-1)                 # (B, L, r, 1)
        A_exp = self.A.unsqueeze(0).unsqueeze(0) # (1, 1, r, r)
        alpha_exp = self.alpha.view(1, 1, -1, 1) # (1, 1, r, 1)

        # Z: mix across r kernels → (B, L, r)
        Z = (V_exp * A_exp * alpha_exp).sum(dim=2)

        # Reconstruct to d_model and add bias
        y = self.Q(Z) + self.bias  # (B, L, d_model)
        return y

