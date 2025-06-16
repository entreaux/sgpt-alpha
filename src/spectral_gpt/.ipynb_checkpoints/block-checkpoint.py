import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_gpt.layers.spectral_attention import SpectralAttention

class SpectralBlock(nn.Module):
    """
    A single Spectral Transformer block: LayerNorm → SpectralAttention → MLP

    Args:
        d_model (int): hidden dimension
        n_heads (int): number of attention heads
        p_depth (float): dropout probability on depth channels
    """
    def __init__(self, d_model: int, n_heads: int, p_depth: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SpectralAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.p_depth = p_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        # Optional depth-wise dropout before attention
        if self.p_depth > 0 and self.training:
            x = F.dropout(x, p=self.p_depth, training=True)

        # Spectral self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x
