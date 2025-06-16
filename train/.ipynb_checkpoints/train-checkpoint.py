# -------------------------------------------------------------
# Spectral-GPT training script
# • optional entropy-band sampling
# • optional rotary φ-phase, dynamic depth
# • validation loop, best-ckpt, early-stop
# -------------------------------------------------------------
import argparse
import math
import time
import pathlib
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ───────────────────────── dataset classes ─────────────────────────
class ByteDataset(Dataset):
    """
    Infinite sampler of random byte windows from a raw .bin file.
    """
    def __init__(self, path: str, seq_len: int):
        raw = pathlib.Path(path).read_bytes()
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return 10_000_000  # virtual size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = torch.randint(
            0, len(self.data) - self.seq_len - 1, ()
        ).item()
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + 1 + self.seq_len]
        return x, y


# Sequential dataset for deterministic validation
class SeqByteDataset(Dataset):
    """
    Deterministic, non-shuffled windows for validation.
    """
    def __init__(self, path: str, seq_len: int):
        raw = pathlib.Path(path).read_bytes()
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + 1 + self.seq_len]
        return x, y


# Optional entropy-banded dataset
try:
    from train.dataset import EntropyBandedDataset   # noqa: F401
except ImportError:
    EntropyBandedDataset = None  # type: ignore


# ───────────────────────── helpers ─────────────────────────────
LN2 = math.log(2.0)


def bpb(loss_nats: torch.Tensor) -> float:
    return float(loss_nats) / LN2


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("🔋  Using Apple GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("🟢  Using CUDA")
        return torch.device("cuda")
    print("⚙️  Using CPU")
    return torch.device("cpu")


def save_ckpt(model: nn.Module, step: int, out_dir: pathlib.Path, tag: str = ""):
    out_dir.mkdir(exist_ok=True)
    name = f"sgpt_step{step}.pt" if not tag else f"{tag}.pt"
    path = out_dir / name
    torch.save(model.state_dict(), path)
    print(f"💾  checkpoint → {path}")


# ───────────────────────── validation routine ─────────────────────────
@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    nats_sum, tok_count = 0.0, 0
    for i, (vx, vy) in enumerate(loader):
        if i >= max_batches:
            break
        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
        logits = model(vx)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 256),
            vy.view(-1),
            reduction="sum",
        )
        nats_sum += loss.item()
        tok_count += vy.numel()
    model.train()
    return (nats_sum / tok_count) / LN2


# ───────────────────────── main ────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    # embedding options
    p.add_argument("--use_phase", action="store_true",
                   help="enable rotary φ-phase in DSN embedding")
    p.add_argument("--max_seq_len", type=int, default=256,
                   help="max sequence length for rotary schedule")
    p.add_argument("--dynamic_depth", action="store_true",
                   help="enable dynamic per-token depth gating")
    # block options
    p.add_argument("--p_depth", type=float, default=0.0,
                   help="depth-channel dropout probability")
    # entropy sampling
    p.add_argument("--entropy_idx", type=str,
                   help=".npz index for entropy sampling")
    p.add_argument("--low_pct", type=float, default=0.33)
    p.add_argument("--high_pct", type=float, default=0.33)
    # validation / early-stop
    p.add_argument("--val", type=str,
                   help="hold-out .bin file for validation")
    p.add_argument("--eval_every", type=int, default=500,
                   help="steps between validation passes")
    p.add_argument("--max_val_batches", type=int, default=100,
                   help="batches evaluated per validation run")
    p.add_argument("--patience", type=int, default=3,
                   help="early-stop patience (val BPB)")
    # basic training args
    p.add_argument("--data",   type=str, required=True,
                   help="raw .bin training file")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs",     type=int, default=64)
    p.add_argument("--seq",    type=int, default=512)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--dmodel", type=int, default=128)
    p.add_argument("--depth",  type=int, default=8)
    p.add_argument("--omega",  type=int, default=64)
    args = p.parse_args()

    device = select_device()

    # ───────── training dataset ─────────
    if args.entropy_idx and EntropyBandedDataset is not None:
        ds = EntropyBandedDataset(
            bin_path=args.data,
            idx_path=args.entropy_idx,
            seq_len=args.seq,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
        )
        print(f"🔍 Entropy-band sampling: low={args.low_pct}, high={args.high_pct}")
    else:
        ds = ByteDataset(args.data, args.seq)
        if args.entropy_idx:
            print("⚠️  Entropy index provided but class unavailable; using ByteDataset.")

    loader = DataLoader(ds, batch_size=args.bs, pin_memory=True)

    # ───────── validation dataset ─────────
    val_loader: Optional[DataLoader] = None
    best_val = float("inf")
    if args.val:
        val_loader = DataLoader(
            SeqByteDataset(args.val, args.seq),
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
        )
        from train.callbacks import EarlyStop
        stopper = EarlyStop(patience=args.patience)
        print("🧪 Validation enabled")

    # ───────── model & optimiser ─────────
    from spectral_gpt.model import QuectoCore
    model = QuectoCore(
        d_model=args.dmodel,
        n_heads=max(1, args.dmodel // 16),
        depth=args.depth,
        Ω=args.omega,
        use_phase=args.use_phase,
        max_seq_len=args.max_seq_len,
        dynamic_depth=args.dynamic_depth,
        p_depth=args.p_depth,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ───────── training loop ─────────
    tokens_seen, t0, step = 0, time.time(), 0
    for epoch in range(args.epochs):
        for x, y in loader:
            step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 256),
                y.view(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens_seen += x.numel()

            # training log
            if step % 100 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                tps = tokens_seen / elapsed
                print(f"EP{epoch}-STEP{step:<6} "
                      f"loss={loss.item():.4f}  "
                      f"bpb={bpb(loss):.2f}  "
                      f"tok/s={tps:,.0f}  "
                      f"dev={device}")

            # validation pass
            if val_loader and step % args.eval_every == 0:
                val_bpb = run_validation(
                    model, val_loader, device, args.max_val_batches
                )
                print(f"🧪  step {step}: val_bpb={val_bpb:.4f}")

                if val_bpb < best_val:
                    best_val = val_bpb
                    save_ckpt(model, step, pathlib.Path("chkpts"), tag="best")
                    print("⭐  new best model saved")
                else:
                    if stopper(best_val):  # type: ignore
                        print("⏹️  early stopping triggered")
                        return

            # periodic checkpoint if no validation
            if not val_loader and step % 500 == 0:
                save_ckpt(model, step, pathlib.Path("chkpts"))

    # final save
    save_ckpt(model, step, pathlib.Path("chkpts"), tag="final")
    print("🏁 Training complete")


if __name__ == "__main__":
    main()
