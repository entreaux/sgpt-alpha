import torch

data class SpectralTuple:
    """
    Represents a spectral tuple Ψ^{A, ω, φ} for each token.
    Carries real, imag parts plus metadata.
    """
    real: torch.Tensor  # (B, L, r)
    imag: torch.Tensor  # (B, L, r)
    omega: torch.Tensor # (B, L, r) or (r,)
    amplitude: torch.Tensor # (B, L, r) or (r,)
    phi: torch.Tensor       # (B, L, r) or (r,)