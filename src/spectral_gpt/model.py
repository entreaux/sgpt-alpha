import torch
import torch.nn as nn
from spectral_gpt.block import SpectralBlock
from spectral_gpt.layers.shared_proj import SharedProj

class QuectoCore(nn.Module):
    """
    SGPT v3 core: threads step and layer_idx through each block.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        depth: int,
        max_seq_len: int,
        fsp_rank: int,
        attn_type: str,
        sia_r: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.shared = SharedProj(d_model, sia_r)
        self.blocks = nn.ModuleList([
            SpectralBlock(
                d_model=d_model,
                fsp_rank=fsp_rank,
                attn_type=attn_type,
                sia_r=sia_r,
                shared=self.shared,
            )
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.LongTensor, step: int = 0) -> torch.Tensor:
        B, L = idx.shape
        device = idx.device

        tok = self.tok_emb(idx)  # (B,L,d_model)
        pos = self.pos_emb(
            torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        )
        x = tok + pos

        # pass step and layer index into each block
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(x, step=step, layer_idx=layer_idx)

        x = self.ln_f(x)
        return self.head(x)
