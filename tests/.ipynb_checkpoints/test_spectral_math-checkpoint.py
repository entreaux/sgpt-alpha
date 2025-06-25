# tests/test_spectral_math.py

import torch
import pytest
from spectral_gpt.layers.spectral_math import DLT

@pytest.mark.parametrize("r", [1, 4, 16])
@pytest.mark.parametrize("batch_seq", [(1,1), (2,5), (3,8)])
def test_dlt_stability_and_shape(r, batch_seq):
    B, L = batch_seq
    dlt = DLT(r=r, delta=1e-3, eps=1e-6)
    # create inputs near zero, negative, large
    u = torch.linspace(-1e-2, 1e-2, steps=B*L*r).view(B, L, r)
    # mix in a larger range
    u = u + torch.randn_like(u) * 10.0
    z = dlt(u)
    # shape must match
    assert z.shape == (B, L, r)
    # no NaN or Inf
    assert torch.isfinite(z).all()
    # amplitude scaling should preserve some sign variation
    assert (z.sign() == torch.sign(u)).all()

def test_omega_within_bounds():
    r = 8
    dlt = DLT(r=r, omega_min=0.5, omega_max=2.0)
    ω = dlt.omega()
    # ω in [min, max]
    assert ω.min() >= 0.5
    assert ω.max() <= 2.0
