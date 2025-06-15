# train/train.py -------------------------------------------------------------
import argparse, math, time, pathlib
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── dataset ──────────────────────────
class ByteDataset(Dataset):
    """
    Infinite sampler of random byte windows from a raw .bin file.
    """
    def __init__(self, path: str, seq_len: int):
        raw = pathlib.Path(path).read_bytes()
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):                      # artificial – treated as endless
        return 10_000_000

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = torch.randint(
            0, len(self.data) - self.seq_len - 1, ()
        ).item()
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + 1 + self.seq_len]
        return x, y

# ───────────────────────── helpers ──────────────────────────
LN2 = math.log(2.0)

def bpb(loss_nats: torch.Tensor) -> float:
    """Cross-entropy (nats) → bits-per-byte."""
    return loss_nats.item() / LN2

def device_select() -> torch.device:
    if torch.backends.mps.is_available():
        print("🔋  Using Apple GPU (MPS)")
        return torch.device("mps")
    print("⚙️  Using CPU")
    return torch.device("cpu")

def save_ckpt(model: nn.Module, step: int, out_dir: pathlib.Path):
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"sgpt_step{step}.pt"
    torch.save(model.state_dict(), fname)
    print(f"💾  checkpoint → {fname}")

# ───────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      required=True,            help="raw .bin file")
    ap.add_argument("--epochs",    type=int,   default=1)
    ap.add_argument("--bs",        type=int,   default=64)
    ap.add_argument("--seq",       type=int,   default=256)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--dmodel",    type=int,   default=64)
    ap.add_argument("--depth",     type=int,   default=4)
    ap.add_argument("--omega",     type=int,   default=32)
    args = ap.parse_args()

    # build dataset / loader
    ds = ByteDataset(args.data, args.seq)
    dl = DataLoader(ds, batch_size=args.bs, pin_memory=True)

    # build model
    from spectral_gpt.model import QuectoCore
    device = device_select()
    model = QuectoCore(
        d_model=args.dmodel,
        n_heads=max(1, args.dmodel // 16),
        depth=args.depth,
        Ω=args.omega  # use_phase=False by default
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tokens_seen = 0
    wall_start  = time.time()
    step        = 0

    for epoch in range(args.epochs):
        for x, y in dl:
            step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss   = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                     )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # FP16 safety
            optimizer.step()

            # stats
            tokens_seen += x.numel()
            if step % 100 == 0:
                elapsed = max(time.time() - wall_start, 1e-6)
                tps     = tokens_seen / elapsed
                print(f"EP{epoch}-STEP{step:<5} "
                      f"loss={loss.item():.4f}  "
                      f"bpb={bpb(loss):.2f}  "
                      f"tok/s={tps:,.0f}  "
                      f"dev={device}")

            # checkpoint
            if step % 500 == 0:
                save_ckpt(model, step, pathlib.Path("chkpts"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
