# src/spectral_gpt/layers/sia.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_gpt.layers.logic import phi_logic
from spectral_gpt.layers.spectral_math import DLT
from spectral_gpt.layers.shared_proj import SharedProj


class SIA(nn.Module):
    """
    Spectral Intra-Attention (SIA) block combining spectral transforms,
    Toeplitz mixing, and a learnable phase bias (phi_bias).
    """
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.d_model = d_model
        self.r = r

        # 0) Soft-start gate 
        self.g_logit = nn.Parameter(torch.Tensor([-1, 0])) 

        # 1) input normalization
        self.ln_in = nn.LayerNorm(d_model)

        # 2) spectral transform (Discrete Laplacian Transform)
        self.dlt = DLT(r=r)
        self.phi_W = nn.Linear(1, r, bias=False)

        # 3) Toeplitz mixing parameters: ensure odd kernel size
        k = r if (r % 2 == 1) else (r - 1 if r > 1 else 1)
        self.toep_params = nn.Parameter(torch.ones(k) / k)

    def _apply_toeplitz(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply a 1D Toeplitz-style convolution across the spectral channels.
        z: (B, L, r) --> returns (B, L, r)
        """
        B, L, r = z.shape
        z_flat = z.reshape(B * L, 1, r) 
        kernel = self.toep_params.view(1, 1, -1) 
        mixed = F.conv1d(z_flat, kernel, padding = 'same')
        return mixed.reshape(B, L, r) 

    def forward(
        self,
        x: torch.Tensor,
        shared: SharedProj,
        delta_Q: nn.Linear,
        step: int,
        layer_idx: int  # unused
    ) -> torch.Tensor:
        if shared.P_shared.weight.device != x.device:          # ← check one tensor
            shared.P_shared = shared.P_shared.to(x.device)
            shared.Q_shared = shared.Q_shared.to(x.device)
              
        """
        x: (B, L, d_model)
        shared: SharedProj instance providing forward_P
        delta_Q: nn.Linear mapping from spectral dim r to d_model
        step: current training step
        layer_idx: index of this SpectralBlock (unused here)

        returns: (B, L, d_model)
        """
        B, L, D = x.shape
        x_norm = self.ln_in(x)

        phi_det = torch.stack([phi_logic(t, L, self.r, device=x.device) for t in range(L)], dim = 0) 
        if not hasattr(self, 'phi_bias') or self.phi_bias.shape != phi_det.shape:
            self.phi_bias = nn.Parameter(torch.zeros_like(phi_det)) 

        feat = x_norm.mean(dim=-1, keepdim=True) 
        phi_data = self.phi_W(feat).mean(dim=0)  

        if step < 2000:
            det_scale = step / 2000.0
        else:
            det_scale = 1.0
        phi_det_scaled = det_scale * phi_det

        phi = phi_det_scaled + self.phi_bias + phi_data

        with torch.no_grad():
            ρ = phi.softmax(dim=-1) + 1e-9 
            H_t = (-ρ * ρ.log()).sum(-1) 
        self.last_phi_entropy = H_t.mean()
        self.prev_phi = phi.detach()
        

        # 2) aggregate phi to per-position scales
        scale = phi.sum(dim=1) / float(self.r)  # (L,)

        # 3) spectral transforms
        u = shared.forward_P(x_norm)      # (B, L, r)
        z = self.dlt(u)                  # (B, L, r)
        z_mixed = self._apply_toeplitz(z)  # (B, L, r)

        # 4) project back to model dimension
        y = delta_Q(z_mixed)  # (B, L, d_model)
        g = torch.sigmoid(self.g_logit) # (0, 1) 

        if g.numel() != y.size(-1):
            g = g.repeat_interleave(y.size(-1) // g.numel()) 
        g = g.view(1, 1, -1)

        # 5) scale and return
        if scale.numel() != y.size(-1):
            r = y.size(-1) // scale.numel() 
            scale = scale.repeat_interleave(r)
        scale = scale.view(1, 1, -1)
        
        out = g * y * scale
        return out
