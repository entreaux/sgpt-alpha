import torch
from spectral_gpt.layers.sia import SIA
from spectral_gpt.layers.shared_proj import SharedProj

def test_phi_W_receives_grad():
    B, L, d, r = 2, 8, 32, 8
    sia = SIA(d_model=d, r=r)
    shared = SharedProj(d, r)
    delta_Q = torch.nn.Linear(r, d, bias=False)

    x = torch.randn(B, L, d, requires_grad=True)
    out = sia(x, shared=shared, delta_Q=delta_Q, step=0, layer_idx=0)
    loss = out.square().mean()
    loss.backward()

    assert sia.phi_W.weight.grad is not None
    assert torch.any(sia.phi_W.weight.grad.abs() > 0), "phi_W receives zero grad"
