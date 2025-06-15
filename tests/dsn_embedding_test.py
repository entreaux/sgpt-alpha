import torch
from spectral_gpt.layers.dsn_embedding import DSNEmbedding

def test_dsn_embedding_output_shape_and_range():
    B, T, Ω = 2, 4, 8
    model = DSNEmbedding(max_depth=Ω)
    x = torch.randint(low=0, high=256, size=(B, T), dtype=torch.long)
    out = model(x)
    # Assert output shape is (B, T, Ω)
    assert out.shape == (B, T, Ω)
    # All values should be finite floats
    assert torch.isfinite(out).all()
