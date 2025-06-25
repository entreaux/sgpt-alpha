import math
import torch
import torch.nn as nn

class DSNEmbedding(nn.Module):
    """
    Discrete Spectral Number embedding with optional rotary phase
    and optional dynamic-depth gating per token.

    Args:
        max_depth (int): number of spectral depth channels (Ω)
        use_phase (bool): enable φ-rotary phase encoding
        max_seq_len (int): maximum sequence length for phase scheduling
        dynamic_depth (bool): enable per-token depth gating
    """
    def __init__(
        self,
        max_depth: int,
        use_phase: bool = True,
        max_seq_len: int = 512,
        dynamic_depth: bool = True
    ):
        super().__init__()
        self.max_depth = max_depth
        self.use_phase = use_phase
        self.dynamic_depth = dynamic_depth

        # Amplitude lookup table: (256 → Ω dimensions)
        self.lookup = nn.Embedding(256, max_depth)

        # Depth-gating MLP if dynamic_depth enabled
        if self.dynamic_depth:
            self.depth_gate = nn.Linear(max_depth, max_depth)

        # Rotary phase buffers if enabled
        if self.use_phase:
            self.max_seq_len = max_seq_len
            alpha = 2 * math.pi / max_seq_len
            self.register_buffer('pos_idx', torch.arange(max_seq_len, dtype=torch.float32))
            self.alpha = alpha

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) input byte tokens in [0,255]
        Returns:
            embed: (B, L, Ω) if no phase,
                   (B, L, 2Ω) if phase,
                   with dynamic gating applied if enabled.
        """
        # Look up raw amplitudes: (B, L, Ω)
        amp = self.lookup(x)

        # Dynamic per-token depth gating
        if self.dynamic_depth:
            # Compute gating weights per channel
            gate = torch.sigmoid(self.depth_gate(amp))  # (B, L, Ω)
            amp = amp * gate

        # If no rotary phase, return amplitude tensor
        if not self.use_phase:
            return amp

        # Compute phase angles for sequence positions
        B, L, _ = amp.size()
        phi = self.alpha * self.pos_idx[:L].to(amp.device)
        phi = phi.view(1, L, 1)

        # Real and imaginary components
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        amp_real = amp * cos_phi
        amp_imag = amp * sin_phi

        # Concatenate → (B, L, 2Ω)
        embed = torch.cat([amp_real, amp_imag], dim=-1)
        return embed
