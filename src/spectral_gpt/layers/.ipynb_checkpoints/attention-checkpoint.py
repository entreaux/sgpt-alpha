# src/spectral_gpt/layers/attention.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Standard causal multi-head self-attention, with optional rotary / dynamic depth.
    """

    def __init__(
        self,
        d_model:       int,
        n_heads:       int,
        omega:         int,
        use_phase:     bool,
        max_seq_len:   int,
        dynamic_depth: bool,
        p_depth:       float,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads       = n_heads
        self.head_dim      = d_model // n_heads
        self.scale         = self.head_dim ** -0.5
        self.wqkv          = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo            = nn.Linear(d_model, d_model, bias=False)
        self.use_phase     = use_phase
        self.max_seq_len   = max_seq_len
        self.dynamic_depth = dynamic_depth
        self.p_depth       = p_depth

        # rotary embeddings if use_phase=True
        if self.use_phase:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq)

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary embeddings to q/k, in-place.
        q,k: (B, nh, L, hd)
        """
        # build sinusoid table up to max_seq_len
        L = q.size(-2)
        t = torch.arange(L, device=q.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (L, hd)
        cos, sin = emb.cos(), emb.sin()
        # apply: ((q, k) rotate pairs)
        q2 = (q * cos.unsqueeze(0)) + (self.rotate_half(q) * sin.unsqueeze(0))
        k2 = (k * cos.unsqueeze(0)) + (self.rotate_half(k) * sin.unsqueeze(0))
        return q2, k2

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # rotate the last dim: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
        x1, x2 = x[..., : x.size(-1)//2], x[..., x.size(-1)//2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape
        qkv = self.wqkv(x)                              # (B, L, 3*d)
        q, k, v = qkv.split(self.n_heads * self.head_dim, dim=-1)
        # reshape (B, nh, L, hd)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # optional rotary phase embeddings
        if self.use_phase:
            q, k = self.apply_rotary(q, k)

        # scaled dot-product
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, L, L)

        # causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        y = attn @ v                                          # (B, nh, L, hd)
        y = y.transpose(1, 2).contiguous().view(B, L, -1)     # (B, L, d_model)

        return self.wo(y)
