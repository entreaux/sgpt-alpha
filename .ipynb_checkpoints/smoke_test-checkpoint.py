import torch
from spectral_gpt.model import QuectoCore

# instantiate a tiny SGPT-v3
model = QuectoCore(
    vocab_size=256,
    d_model=32,
    depth=2,
    max_seq_len=16,
    fsp_rank=16,
    attn_type="sia",
    sia_r=16,
)
x = torch.randint(0, 256, (2, 16))        # batch of 2, seq len 16
logits = model(x)                         # forward pass
print("Output shape:", logits.shape)
