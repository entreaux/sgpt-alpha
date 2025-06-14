import torch
from spectral_gpt.layers.spectral_attention import SpectralAttention

def test_spectral_attention_shapes():
    B, L, d, h = 2, 16, 32, 4
    sa = SpectralAttention(d_model=d, n_heads=h)
    x = torch.randn(B, L, d)
    y = sa(x)
    assert y.shape == (B, L, d)
    assert torch.isfinite(y).all()