import argparse
import math
import time
import pathlib
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spectral_gpt.datasets.adaptive_entropy import AdaptiveEntropySampler
from spectral_gpt.datasets.entropy_index import load_entropy_index
from spectral_gpt.utils.ema import EMA
from spectral_gpt.model import QuectoCore

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
        return 10_000_000

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = torch.randint(0, len(self.data) - self.seq_len - 1, ()).item()
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + 1 + self.seq_len]
        return x, y

class SeqByteDataset(Dataset):
    """
    Deterministic sequence windows for validation.
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

# ───────────────────── helpers ─────────────────────
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
    fname = f"sgpt_step{step}.pt" if not tag else f"{tag}.pt"
    path = out_dir / fname
    torch.save(model.state_dict(), path)
    print(f"💾  checkpoint → {path}")

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
        vx, vy = vx.to(device), vy.to(device)
        logits = model(vx)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), vy.view(-1), reduction="sum"
        )
        nats_sum += loss.item()
        tok_count += vy.numel()
    model.train()
    return (nats_sum / tok_count) / LN2

# ───────────────────────── main ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # core training args
    parser.add_argument("--data",   type=str, required=True)
    parser.add_argument("--val",    type=str)
    parser.add_argument("--entropy_idx", type=str)
    parser.add_argument("--seq",    type=int, default=256)
    parser.add_argument("--bs",     type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--lr_schedule", type=str, choices=["none","cosine"], default="none")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--ema",    type=float, default=0.0)
    parser.add_argument("--ema_start", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--patience",    type=int, default=3)

    # model config
    parser.add_argument("--dmodel", type=int, default=128)
    parser.add_argument("--depth",  type=int, default=8)
    parser.add_argument("--omega",  type=int, default=64)
    parser.add_argument("--use_phase", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dynamic_depth", action="store_true")
    parser.add_argument("--p_depth", type=float, default=0.0)
    # FSP-FFN flags
    parser.add_argument("--use_fsp_ffn", action="store_true")
    parser.add_argument("--fsp_rank", type=int, default=None)
    parser.add_argument("--omega_min", type=float, default=0.5)
    parser.add_argument("--omega_max", type=float, default=3.0)
    args = parser.parse_args()

    device = select_device()

    # dataset
    if args.entropy_idx:
        train_ds = AdaptiveEntropySampler(args.data, args.entropy_idx, args.seq)
        print("🔄  Adaptive Entropy Scheduling enabled")
    else:
        train_ds = ByteDataset(args.data, args.seq)
    train_loader = DataLoader(train_ds, batch_size=args.bs)

    # validation
    val_loader = None
    if args.val:
        val_ds = SeqByteDataset(args.val, args.seq)
        low_idx, mid_idx, high_idx = load_entropy_index(args.entropy_idx)
        # build per-band val loaders
        val_loader = DataLoader(val_ds, batch_size=args.bs)
        print(f"🔍 Val set size: {len(val_ds)} sequences")

    # model & optimizer
    model = QuectoCore(
        d_model=args.dmodel,
        n_heads=max(1, args.dmodel//16),
        depth=args.depth,
        Ω=args.omega,
        use_phase=args.use_phase,
        max_seq_len=args.max_seq_len,
        dynamic_depth=args.dynamic_depth,
        p_depth=args.p_depth,
        use_fsp_ffn=args.use_fsp_ffn,
        fsp_rank=args.fsp_rank,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # LR scheduler
    total_steps = args.epochs * args.steps_per_epoch
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=args.min_lr
        )
        print(f"🔄 Using cosine LR: start={args.lr:.1e}, min={args.min_lr:.1e}, steps={total_steps}")

    # EMA
    ema = EMA(model, decay=args.ema) if args.ema>0 else None
    if ema:
        print(f"🌀 EMA enabled (decay={args.ema}, start={args.ema_start})")

    # training loop
    best_val = float('inf')
    stopper_count = 0
    step = 0
    for epoch in range(args.epochs):
        for x,y in train_loader:
            if step == args.ema_start and ema:
                ema.register()  # start tracking
                print(f"ℹ️  EMA tracking begins at step {step}")

            step += 1
            model.train()
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1,256), y.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step()
            if args.lr_schedule=="cosine": scheduler.step()
            if ema and step>args.ema_start: ema.update()

            if step % 100 == 0:
                lrs = optim.param_groups[0]['lr']
                print(f"EP{epoch}-STP{step:<5} loss={loss.item():.4f} bpb={bpb(loss):.2f} lr={lrs:.1e}")

            # validation
            if args.val and step % args.eval_every == 0:
                val_bpb = run_validation(ema.ema_model if ema else model,
                                         val_loader, device, args.steps_per_epoch//10)
                print(f"🧪 step {step}: val_bpb={val_bpb:.4f}")
                if val_bpb < best_val:
                    best_val = val_bpb; stopper_count=0
                    save_ckpt(ema.ema_model if ema else model, step, pathlib.Path("chkpts"), "best")
                    print("⭐ new best saved")
                else:
                    stopper_count +=1
                    if stopper_count>=args.patience:
                        print("⏹️ early stopping")
                        save_ckpt(model, step, pathlib.Path("chkpts"), "final")
                        return

            if step >= (epoch+1)*args.steps_per_epoch:
                save_ckpt(model, step, pathlib.Path("chkpts"), f"sgpt_step{step}")
                break

    save_ckpt(model, step, pathlib.Path("chkpts"), "final")
    print("🏁 Training complete")

if __name__ == '__main__':
    main()
