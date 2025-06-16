import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    """
    Linear-time spectral self-attention using prefix-sum kernel.

    Formula: SA(Q, K, V) = (φ(Q)) [∑_{i=1..t} φ(K_i) ⊙ V_i]
    where φ(x)=ELU(x)+1 (ensures non-negativity), ⊙ is elementwise.
    Applies per-head scaling 1/√head_dim for stable gradients.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        # projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # parseval scaling: 1/sqrt(r), here r=head_dim
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, D = x.shape
        # project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # feature map
        phi_q = F.elu(q) + 1  # (B, L, d_model)
        phi_k = F.elu(k) + 1  # (B, L, d_model)
        # reshape to heads
        phi_q = phi_q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, L, hd)
        phi_k = phi_k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v      = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # prefix-sum kernel (elementwise K ⊙ V)
        # accumulate over sequence dimension
        KV_sum = torch.cumsum(phi_k * v, dim=2)  # (B, h, L, hd)
        # apply Q gating
        context = phi_q * KV_sum               # (B, h, L, hd)
        # merge heads
        context = context.transpose(1, 2).reshape(B, L, D)
        # parseval scale
        context = context * self.scale
        # output projection
        return self.out_proj(context)
