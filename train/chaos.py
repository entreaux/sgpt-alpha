# train/chaos.py
import torch, itertools

def inject_gaussian_noise(model, sigma_omega=1e-3, sigma_A=1e-3):
    """
    Add N(0,σ) noise to each DLT's raw omega and amplitude A.
    """
    for dlt in itertools.chain.from_iterable(
            m.modules() for m in model.modules()    # walk once, filter inside
        ):
        if dlt.__class__.__name__ == "DLT":          # cheap type check
            if hasattr(dlt, "_raw_omega"):
                dlt._raw_omega.data.add_(torch.randn_like(dlt._raw_omega) * sigma_omega)
            if hasattr(dlt, "A"):
                dlt.A.data.add_(torch.randn_like(dlt.A) * sigma_A)
