import torch
import time
from spectral_gpt.model import QuectoCore

def time_forward(L):
    model = QuectoCore(
        vocab_size=256, d_model=128, depth=4,
        max_seq_len=L, fsp_rank=64, attn_type="sia", sia_r=64
    ).eval()
    x = torch.randint(0, 256, (1, L))
    # Warmup
    for _ in range(5): _ = model(x, step=0)
    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    _ = model(x, step=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time() - t0

for L in [64, 128, 256, 512, 1024]:
    t = time_forward(L)
    print(f"L={L:4d} → {t*1000:.2f} ms")
