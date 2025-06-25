import torch
from spectral_gpt.layers.logic import phi_logic

def test_phi_logic_length_and_repeat():
    # Test for various r, t, L
    for r in [4, 8, 16]:
        for t, L in [(0, 5), (3, 10), (9, 12)]:
            vec = phi_logic(t, L, r)
            # correct length
            assert vec.shape == (r,)
            # values are in [–1,1] or 0–1
            assert torch.all(vec <= 1.0) and torch.all(vec >= -1.0)

def test_phi_logic_edge_positions():
    # t=0 should set first and fourth entries =1
    vec0 = phi_logic(0, 5, 8)
    assert vec0[0] == 1.0
    assert vec0[3] == 1.0
    # t=L-1 should set index 4 =1
    vec_last = phi_logic(4, 5, 8)
    assert vec_last[4] == 1.0
