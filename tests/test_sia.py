# tests/test_sia.py
import torch
import torch.nn as nn
import pytest
from spectral_gpt.layers.sia import SIA
from spectral_gpt.layers.shared_proj import SharedProj

@pytest.mark.parametrize("B,L,r", [(1,4,8), (2,5,16)])
def test_toeplitz_mixing_shape_and_finiteness(B, L, r):
    d_model = 32
    sia = SIA(d_model=d_model, r=r)
    x_norm = torch.randn(B, L, d_model)
    class StubShared:
        def forward_P(self, x): return torch.randn(B, L, r)
        def forward_Q(self, z): return torch.zeros(B, L, d_model)
    stub_shared = StubShared()
    delta_Q = torch.nn.Linear(r, d_model, bias=False)

    with torch.no_grad():
        u = stub_shared.forward_P(x_norm)
        z = sia.dlt(u)
        z_mixed = sia._apply_toeplitz(z)
    assert z_mixed.shape == (B, L, r)
    assert torch.isfinite(z_mixed).all()

def test_sia_forward_runs_and_preserves_shape():
    B, L, d_model, r = 2, 10, 32, 8
    sia = SIA(d_model=d_model, r=r)
    shared = SharedProj(d_model, r)
    delta_Q = torch.nn.Linear(r, d_model, bias=False)
    x = torch.randn(B, L, d_model)
    out = sia(
        x,
        shared=shared,
        delta_Q=delta_Q,
        step=0,
        layer_idx=0
    )
    assert out.shape == x.shape

def test_sia_forward_with_step_and_layer():
    B, L, d_model, r = 2, 5, 32, 8
    sia = SIA(d_model=d_model, r=r)
    shared = SharedProj(d_model, r)
    delta_Q = torch.nn.Linear(r, d_model, bias=False)
    x = torch.randn(B, L, d_model)
    out = sia(
        x,
        shared=shared,
        delta_Q=delta_Q,
        step=123,
        layer_idx=4
    )
    assert out.shape == x.shape
    # prev_phi should now be initialized
    assert hasattr(sia, "prev_phi")
    assert sia.prev_phi.shape == (L, r)
