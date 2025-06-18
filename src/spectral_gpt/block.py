import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_gpt.layers.spectral_attention import SpectralAttention
from spectral_gpt.layers.fsp_ffn import FSPFFN

class SpectralBlock(nn.Module):
    """
    Spectral Transformer block with optional FSP-FFN.

    Args:
        d_model (int): hidden dimension
        n_heads (int): number of attention heads
        p_depth (float): dropout probability on depth channels
        use_fsp_ffn (bool): enable Fractional Spectral FFN
        fsp_rank (int): low-rank projection dimension for FSP-FFN
        omega_min (float): minimum fractional exponent
        omega_max (float): maximum fractional exponent
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_depth: float = 0.0,
        use_fsp_ffn: bool = False,
        fsp_rank: int = None,
        omega_min: float = 0.5,
        omega_max: float = 3.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SpectralAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.p_depth = p_depth

        # FFN selection
        if use_fsp_ffn:
            r = fsp_rank or (d_model // 8)
            self.ffn = FSPFFN(d_model, r, omega_min, omega_max)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional depth-wise dropout
        if self.p_depth > 0 and self.training:
            x = F.dropout(x, p=self.p_depth)

        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x