# src/spectral_gpt/model.py
import torch.nn as nn
from spectral_gpt.layers.dsn_embedding import DSNEmbedding
from spectral_gpt.block import SpectralBlock

class QuectoCore(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, depth: int = 4, Ω: int = 32):
        super().__init__()
        self.embed = DSNEmbedding(max_depth=Ω)
        self.proj  = nn.Linear(Ω, d_model)
        self.blocks = nn.ModuleList(
            SpectralBlock(d_model, n_heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)
        # tiny vocab for byte-level LM
        self.head = nn.Linear(d_model, 256, bias=False)

    def forward(self, x):                 # x: (B, L) bytes
        x = self.proj(self.embed(x))      # (B, L, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)               # logits (B, L, 256)
