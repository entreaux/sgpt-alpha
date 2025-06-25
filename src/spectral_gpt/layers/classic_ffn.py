# src/spectral_gpt/layers/classic_ffn.py

import torch
import torch.nn as nn

class ClassicFFN(nn.Module):
    """
    A simple “classic” FFN: Linear→GELU→Linear.
    Only here to satisfy imports when ffn_type='classic'.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return self.net(x)
