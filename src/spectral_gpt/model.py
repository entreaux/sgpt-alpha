import torch
import torch.nn as nn
from spectral_gpt.layers.dsn_embedding import DSNEmbedding
from spectral_gpt.block import SpectralBlock

class QuectoCore(nn.Module):
    """
    Spectral‑GPT core:
    * DSN embedding with optional φ‑rotary phase
    * Optional dynamic per‑token depth gating
    * Depth‑dropout inside each SpectralBlock

    Args:
        d_model (int): hidden size
        n_heads (int): attention heads
        depth (int): number of SpectralBlocks
        Ω (int): DSN depth channels
        use_phase (bool): enable φ‑rotary
        max_seq_len (int): L_max for phase schedule
        dynamic_depth (bool): enable gating MLP in DSN
        p_depth (float): dropout prob on depth channels (0‒1)
    """
    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        depth: int = 2,
        Ω: int = 8,
        use_phase: bool = False,
        max_seq_len: int = 512,
        dynamic_depth: bool = False,
        p_depth: float = 0.0,
    ):
        super().__init__()
        self.Ω = Ω
        self.use_phase = use_phase
        self.dynamic_depth = dynamic_depth

        # DSN embedding layer
        self.embed = DSNEmbedding(
            max_depth=Ω,
            use_phase=use_phase,
            max_seq_len=max_seq_len,
            dynamic_depth=dynamic_depth,
        )
        in_channels = Ω * (2 if use_phase else 1)

        # Projection Ω(or 2Ω) → d_model
        self.proj = nn.Linear(in_channels, d_model)

        # Stacked SpectralBlocks
        self.blocks = nn.ModuleList([
            SpectralBlock(d_model=d_model, n_heads=n_heads, p_depth=p_depth)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)

    # ---------------------------------------------------------
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (B, L)
        x = self.embed(x)   # (B, L, Ω or 2Ω)
        x = self.proj(x)    # (B, L, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x) # (B, L, 256)
