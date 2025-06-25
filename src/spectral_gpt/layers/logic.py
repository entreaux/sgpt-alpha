# logic.py
import torch

def phi_logic(t: int, L: int, r: int, device = None) -> torch.Tensor:
    """
    Symbolic φ operator for position t in sequence of length L, across r channels.
    Returns a 1D tensor of length r.
    """
    base = []
    base.append( 1.0 if (t % 2) == 0 else -1.0 )
    base.append( 1.0 if (t % 3) == 0 else 0.0 )
    base.append( t / L )
    base.append( 1.0 if t == 0 else 0.0 )
    base.append( 1.0 if t == L - 1 else 0.0 )
    base.append( 1.0 if ((t // 10) % 2) == 1 else 0.0 )
    base.append( (t % 4) / 4.0 )
    base.append( 1.0 if ((t // 5) % 3) == 1 else 0.0 )

    vec = torch.tensor(base, dtype=torch.float32, device=device)
    if vec.numel() < r:
        # repeat until length ≥ r, then slice
        reps = (r + vec.numel() - 1) // vec.numel()
        vec = vec.repeat(reps)[:r]
    else:
        vec = vec[:r]
    return vec
