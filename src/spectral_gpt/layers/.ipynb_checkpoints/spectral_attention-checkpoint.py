# src/spectral_gpt/layers/spectral_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def phi(x: torch.Tensor) -> torch.Tensor:
    """ELU(x) + 1 : non-negative feature map for linear attention."""
    return F.elu(x) + 1.0

class SpectralAttention(nn.Module):
    """
    Linear-time attention (O(L d r)) using prefix sums.
    r = head_dim.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d = d_model
        self.h = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        returns : (B, L, d_model)
        """
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each (B, L, d_model)

        # reshape to (B, h, L, head_dim)
        q = q.view(B, L, self.h, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.h, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.h, self.head_dim).transpose(1, 2)

        # feature map
        q = phi(q)                     # (B, h, L, r)
        k = phi(k)                     # (B, h, L, r)

        # prefix sums over sequence
        kv_cumsum = torch.cumsum(k * v, dim=2)    # (B, h, L, r)
        k_cumsum  = torch.cumsum(k, dim=2)        # (B, h, L, r)

        # linear attention
        out = (q * kv_cumsum) / (q * k_cumsum + 1e-9)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d)
        return self.out(out) * (self.head_dim ** -0.5)   # Parseval scaling
