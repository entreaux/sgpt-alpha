import torch
import torch.nn.functional as F

EPS = 1e-8

def omega_entropy(dlt):
    """Shannon entropy of the learned angular frequencies ω."""
    # ω shape: (r,)  or (L, r) after broadcast – flatten
    w = dlt.omega().view(-1)
    p = F.softmax(w, dim=0).clamp(min=EPS)
    return -(p * p.log()).sum()

def amplitude_l1(dlt):
    """L1 sparsity penalty on amplitude matrix A."""
    return dlt.A.abs().mean()
