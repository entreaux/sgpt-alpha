# src/spectral_gpt/block.py
import torch
import torch.nn as nn
from spectral_gpt.layers.sia import SIA

class SpectralBlock(nn.Module):
    """
    Single SGPT v3 block with dynamic α gating based on SIA & FFN output norms.
    """
    def __init__(
        self,
        d_model: int,
        fsp_rank: int,
        attn_type: str,
        sia_r: int,
        shared: nn.Module,
    ):
        super().__init__()
        self.shared = shared
        self.ln1 = nn.LayerNorm(d_model)
        if attn_type == "sia":
            self.self_interact = SIA(d_model, sia_r)
        else:
            raise ValueError(f"Unsupported attn_type={attn_type}")
        self.delta_Q = nn.Linear(sia_r, d_model, bias=False)
        # initialize delta_Q to small values for stability
        nn.init.constant_(self.delta_Q.weight, 0.01)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, fsp_rank),
            nn.GELU(),
            nn.Linear(fsp_rank, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int,
        layer_idx: int
    ) -> torch.Tensor:
        # 1) Spectral interaction output
        sia_out = self.self_interact(
            self.ln1(x),
            shared=self.shared,
            delta_Q=self.delta_Q,
            step=step,
            layer_idx=layer_idx,
        )
        # compute norms for gating
        norm_sia = sia_out.norm(p=2)

        # 2) FFN path on post-SIA input
        # apply full residual for stability in ffn input
        x_sia = x + sia_out
        z = self.ffn(self.ln2(x_sia))
        norm_ffn = z.norm(p=2)

        # 3) dynamic α computation
        eps = 1e-6
        alpha = norm_sia / (norm_sia + norm_ffn + eps)

        # 4) combine with gated SIA + FFN
        x = x + alpha * sia_out
        x = x + z
        return x