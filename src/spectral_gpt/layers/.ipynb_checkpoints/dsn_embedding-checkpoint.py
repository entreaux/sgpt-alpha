import torch
import torch.nn as nn

class DSNEmbedding(nn.Module):
    """
    Simple Discrete Spectral Number embedding for byte tokens.
    - Ω = 8 (fixed max depth)
    - Given input: LongTensor of shape (B, T) with values in [0, 255].
    - Returns: FloatTensor of shape (B, T, Ω) holding A_0…A_7 for each byte.
    """

    def __init__(self, max_depth: int = 32):
        super().__init__()
        self.max_depth = max_depth
        # Learnable amplitude table: [256 bytes × Ω depths]
        self.byte2dsn = nn.Parameter(torch.randn(256, max_depth))

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        x: (B, T) of dtype torch.long, values ∈ [0,255]
        returns: (B, T, Ω) of dtype torch.float
        """
        # Gather the depth-vector for each byte
        # byte2dsn shape: [256, Ω]; x expands to [B, T, Ω]
        dsn_vectors = self.byte2dsn[x]  # PyTorch: advanced indexing
        return dsn_vectors
