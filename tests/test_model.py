# tests/test_model.py
import torch
from spectral_gpt.model import QuectoCore

def test_quecto_forward():
    B, L = 2, 16
    model = QuectoCore()
    x = torch.randint(0, 256, (B, L), dtype=torch.long)
    y = model(x)
    assert y.shape == (B, L, 256)
    assert torch.isfinite(y).all()
    # sanity-check parameter budget ≈100k
    assert sum(p.numel() for p in model.parameters()) < 105_000
