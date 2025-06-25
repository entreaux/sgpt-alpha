# tests/dsn_embedding_test.py
import torch
import pytest
from spectral_gpt.layers.dsn_embedding import DSNEmbedding

@pytest.mark.parametrize("Ω", [1, 4, 8, 16])
def test_dsn_embedding_output_shape_and_range(Ω):
    B, T = 2, 4
    model = DSNEmbedding(max_depth=Ω)
    x = torch.randint(low=0, high=256, size=(B, T), dtype=torch.long)
    out = model(x)
    # With use_phase=True, DSNEmbedding outputs 2*Ω channels
    assert out.shape == (B, T, 2 * Ω)
    assert torch.isfinite(out).all()
