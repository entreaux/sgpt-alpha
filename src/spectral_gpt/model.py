import torch
import torch.nn as nn
from spectral_gpt.layers.dsn_embedding import DSNEmbedding
from spectral_gpt.layers.block import SpectralBlock

class QuectoCore(nn.Module):
    """
    Spectral‑GPT core with optional FSP-FFN.
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        depth: int = 8,
        Ω: int = 64,
        use_phase: bool = False,
        max_seq_len: int = 512,
        dynamic_depth: bool = False,
        p_depth: float = 0.0,
        use_fsp_ffn: bool = False,
        fsp_rank: int = None,
        omega_min: float = 0.5,
        omega_max: float = 3.0,
    ):
        super().__init__()
        self.embed = DSNEmbedding(
            max_depth=Ω,
            use_phase=use_phase,
            max_seq_len=max_seq_len,
            dynamic_depth=dynamic_depth,
        )
        in_channels = Ω * (2 if use_phase else 1)
        self.proj = nn.Linear(in_channels, d_model)

        self.blocks = nn.ModuleList([
            SpectralBlock(
                d_model=d_model,
                n_heads=n_heads,
                p_depth=p_depth,
                use_fsp_ffn=use_fsp_ffn,
                fsp_rank=fsp_rank,
                omega_min=omega_min,
                omega_max=omega_max,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)

