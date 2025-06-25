# src/spectral_gpt/layers/spectral_math.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DLT(nn.Module):
    """
    Discrete Log Transform (DLT) layer for spectral-domain transforms.
    Computes z = sign(u) * |u - delta|^omega, where omega is learned within bounds.
    """
    def __init__(self, r, delta=1e-3, eps=1e-6, omega_min=0.1, omega_max=10.0):
        super().__init__()
        self.r = r
        self.delta = delta
        self.eps = eps
        self.omega_min = omega_min
        self.omega_max = omega_max
        # Learnable raw parameter for omega, one per spectral channel
        self.omega_param = nn.Parameter(torch.zeros(r))
        
        init_A = torch.ones(r) + 1e-3 * torch.randn(r)
        self.A = nn.Parameter(init_A)
        

    def omega(self) -> torch.Tensor:
        # Map raw parameter through sigmoid to [omega_min, omega_max]
        sig = torch.sigmoid(self.omega_param)
        return sig * (self.omega_max - self.omega_min) + self.omega_min

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, r)
        sign_u = torch.sign(u)
        # Log of magnitude, with stability clamp
        log_u = torch.log(torch.clamp(torch.abs(u) - self.delta, min=self.eps))
        # Broadcast omega to match u
        omega = self.omega().view(1, 1, -1)
        A     = self.A.view(1, 1, -1)   
        # Exponentiate
        z = A * sign_u * torch.exp(omega * log_u)
        return z
