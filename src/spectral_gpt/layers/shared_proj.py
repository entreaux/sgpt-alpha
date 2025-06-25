import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_gpt.layers.logic import phi_logic
from spectral_gpt.layers.spectral_math import DLT


class SharedProj:
    """
    Shared projection for spectral attention:
    - forward_P: projects from d_model to spectral dimension r
    - forward_Q: projects from spectral dimension r back to d_model
    """
    def __init__(self, d_model: int, r: int):
        self.P_shared = nn.Linear(d_model, r, bias=True)
        self.Q_shared = nn.Linear(r, d_model, bias=False)

    def forward_P(self, x: torch.Tensor) -> torch.Tensor:
        return self.P_shared(x)

    def forward_Q(self, z: torch.Tensor) -> torch.Tensor:
        return self.Q_shared(z)