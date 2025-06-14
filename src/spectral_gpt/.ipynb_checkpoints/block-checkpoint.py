# src/spectral_gpt/block.py
import torch.nn as nn
from spectral_gpt.layers.spectral_attention import SpectralAttention
from spectral_gpt.layers.feedforward import SpectralFFN

class SpectralBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = SpectralAttention(d_model, n_heads)
        self.ffn  = SpectralFFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
