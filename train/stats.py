import torch
import torch.nn as nn

class PhiEntropyHook:
    """
    Computes and logs the average Shannon entropy of prev_phi buffers in SIA modules.
    Adds a small epsilon to avoid log(0).
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.logs: list[tuple[int, float]] = []  # (step, entropy)

    def __call__(self, step: int, sia_module: nn.Module):
        ph = sia_module.prev_phi.detach().cpu()  # (L, r)
        # softmax to get probabilities, then clamp for numeric stability
        p = torch.softmax(ph, dim=-1).clamp(min=self.eps)
        # entropy per position, then average
        ent = -(p * p.log()).sum(dim=-1).mean().item()
        self.logs.append((step, ent))


class OmegaSpectrumHook:
    """
    Records min, mean, max of omega values in each DLT module.
    """
    def __init__(self):
        self.logs: list[tuple[int, float, float, float]] = []  # (step, min, mean, max)

    def __call__(self, step: int, dlt_module: nn.Module):
        ω = dlt_module.omega().detach().cpu()
        wmin = ω.min().item()
        wmean = ω.mean().item()
        wmax = ω.max().item()
        self.logs.append((step, wmin, wmean, wmax))


class ASparsityHook:
    """
    Records amplitude sparsity AND basic stats (mean, std) for each DLT module.
    """
    def __init__(self, threshold: float = 5e-4):
        self.threshold = threshold
        self.logs: list[tuple[int, float, float, float]] = []  # (step, sparsity, mean, std)

    def __call__(self, step: int, dlt_module: nn.Module):
        A = dlt_module.A.detach().abs().cpu().view(-1)
        sparsity = (A < self.threshold).float().mean().item()
        mean_a = A.mean().item()
        std_a = A.std().item()
        self.logs.append((step, sparsity, mean_a, std_a))

class OmegaStdHook:
    def __init__(self): self.logs=[]
    def __call__(self, step, dlt):
        self.logs.append((step, float(dlt.omega().std().detach().cpu())))

class ASpHook:
    def __init__(self, thresh=0.05): self.thresh=thresh; self.logs=[]
    def __call__(self, step, dlt):
        a = dlt.A.detach().abs().cpu()
        sp = (a < self.thresh).float().mean().item()
        self.logs.append((step, sp))

