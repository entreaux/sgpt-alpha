# -------------------------------------------------------------
# Spectral-GPT training script
#  • Adaptive-Entropy curriculum        (optional)
#  • Rotary φ-phase & dynamic depth     (optional)
#  • Validation / early-stop / best-ckpt
#  • Exponential-Moving-Average (EMA)   (optional, delayed start)
#  • Cosine LR annealing                (optional)
#  • Per-band validation loaders        (fixed index filtering)
# -------------------------------------------------------------
import argparse
import itertools
import math
import time
import pathlib
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from spectral_gpt.datasets.adaptive_entropy import AdaptiveEntropySampler
from spectral_gpt.datasets.entropy_index import load_entropy_index
from spectral_gpt.utils.ema import EMA

# ───────────────────────── helpers ─────────────────────────
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

def save_ckpt(model: nn.Module, step: int, tag: str = ""):
    out = pathlib.Path("chkpts"); out.mkdir(exist_ok=True)
    name = f"sgpt_step{step}.pt" if tag == "" else f"{tag}.pt"
    torch.save(model.state_dict(), out / name)
    print(f"💾  checkpoint → {out/name}")

@torch.no_grad()
def eval_bpb(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             max_batches: int) -> float:
    model.eval()
    nats_sum, tok_sum = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        nats_sum += loss.item()
        tok_sum += y.numel()
    model.train()
    return (nats_sum / tok_sum) / LN2

# ───────────────────────── datasets ────────────────────────
class ByteDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        data = pathlib.Path(path).read_bytes()
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return 10_000_000  # virtual

    def __getitem__(self, _):
        start = torch.randint(0, len(self.data) - self.seq_len - 1, ()).item()
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + 1 + self.seq_len]
        return x, y

class SeqByteDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        data = pathlib.Path(path).read_bytes()
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.data[s:s + self.seq_len]
        y = self.data[s + 1:s + 1 + self.seq_len]
        return x, y

# ───────────────────────── main ────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # curriculum & epochs
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    # EMA
    parser.add_argument("--ema", type=float, default=0.0,
                        help="EMA decay (0 disables)")
    parser.add_argument("--ema_start", type=int, default=5000,
                        help="first step when EMA is used for validation")
    # LR schedule
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"],
                        default="constant", help="LR schedule to use")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="minimum LR for cosine schedule")
    # embedding / model
    parser.add_argument("--use_phase", action="store_true")
    parser.add_argument("--dynamic_depth", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--p_depth", type=float, default=0.0)
    # data paths
    parser.add_argument("--data", required=True)
    parser.add_argument("--entropy_idx")
    parser.add_argument("--val")
    # training hyper-params
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dmodel", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--omega", type=int, default=64)
    # validation / early-stop
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_val_batches", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()
    device = select_device()

    # ───────── training dataset ─────────
    if args.entropy_idx:
        train_ds = AdaptiveEntropySampler(args.data, args.entropy_idx, args.seq)
        print("🔄  Adaptive Entropy Scheduling enabled")
    else:
        train_ds = ByteDataset(args.data, args.seq)
    train_loader = DataLoader(train_ds, batch_size=args.bs, pin_memory=True)

    # ───────── validation loaders ──────
    val_loader = None
    val_low_loader = val_mid_loader = val_high_loader = None
    if args.val:
        val_loader = DataLoader(
            SeqByteDataset(args.val, args.seq),
            batch_size=args.bs, shuffle=False, pin_memory=True
        )
        from train.callbacks import EarlyStop
        stopper = EarlyStop(patience=args.patience)
        print("🧪  Validation enabled")

        if args.entropy_idx:
            offsets, bands, _ = load_entropy_index(args.entropy_idx)
            idxs = (offsets // args.seq).astype(int)
            val_ds = SeqByteDataset(args.val, args.seq)
            max_idx = len(val_ds)

            # build per-band index lists, filtering out-of-range
            band_idxs = {}
            for b in (0, 1, 2):
                raw = [i for i, bb in zip(idxs, bands) if bb == b]
                band_idxs[b] = [i for i in raw if 0 <= i < max_idx]

            val_low_loader = DataLoader(
                val_ds, batch_size=args.bs,
                sampler=SubsetRandomSampler(band_idxs[0]),
                pin_memory=True,
            )
            val_mid_loader = DataLoader(
                val_ds, batch_size=args.bs,
                sampler=SubsetRandomSampler(band_idxs[1]),
                pin_memory=True,
            )
            val_high_loader = DataLoader(
                val_ds, batch_size=args.bs,
                sampler=SubsetRandomSampler(band_idxs[2]),
                pin_memory=True,
            )
            print(f"🔍  Per-band val sizes: "
                  f"low={len(band_idxs[0])}, "
                  f"mid={len(band_idxs[1])}, "
                  f"high={len(band_idxs[2])}")

    # ───────── model & optimizer ──────
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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ───────── LR scheduler ───────────
    total_steps = args.epochs * args.steps_per_epoch
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=args.min_lr
        )
        print(f"🔄  Using cosine LR: start={args.lr:.1e}, "
              f"min={args.min_lr:.1e}, steps={total_steps}")
    else:
        scheduler = None

    # ───────── EMA shadow model ───────
    ema = EMA(model, args.ema) if args.ema > 0.0 else None
    if ema:
        ema_model = deepcopy(model).eval().requires_grad_(False)
        print(f"🌀  EMA enabled (decay={args.ema}, start={args.ema_start})")

    # ───────── training loop ───────────
    data_iter = itertools.cycle(train_loader)
    global_step, best_val = 0, float("inf")
    tokens_seen, t0 = 0, time.time()

    for epoch in range(args.epochs):
        if isinstance(train_ds, AdaptiveEntropySampler):
            train_ds.set_epoch(epoch)
            print(f"📊  AES schedule epoch {epoch}: {train_ds.weights}")

        for _ in range(args.steps_per_epoch):
            global_step += 1
            x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if scheduler:
                scheduler.step()
            if ema:
                ema.update(model)

            tokens_seen += x.numel()

            # ─── training log ───
            if global_step % 100 == 0:
                tok_per_s = tokens_seen / max(time.time() - t0, 1e-6)
                lr = scheduler.get_last_lr()[0] if scheduler else args.lr
                print(f"EP{epoch}-STP{global_step:<6} "
                      f"loss={loss.item():.4f}  "
                      f"bpb={bpb(loss):.2f}  "
                      f"lr={lr:.1e}  "
                      f"tok/s={tok_per_s:,.0f}")
                tokens_seen, t0 = 0, time.time()

            # ─── validation ───
            if val_loader and global_step % args.eval_every == 0:
                use_ema = ema and global_step >= args.ema_start
                target = ema_model if use_ema else model
                if use_ema and global_step == args.ema_start:
                    print(f"ℹ️  EMA validation begins at step {global_step}")

                val_b = eval_bpb(target, val_loader, device, args.max_val_batches)
                print(f"🧪  step {global_step}: val_bpb={val_b:.4f}")

                if val_low_loader:
                    b0 = eval_bpb(target, val_low_loader, device, args.max_val_batches)
                    b1 = eval_bpb(target, val_mid_loader, device, args.max_val_batches)
                    b2 = eval_bpb(target, val_high_loader, device, args.max_val_batches)
                    print(f"        band0={b0:.4f}  band1={b1:.4f}  band2={b2:.4f}")

                if val_b < best_val:
                    best_val = val_b
                    save_ckpt(model, global_step, tag="best")
                    print("⭐  new best saved")
                else:
                    from train.callbacks import EarlyStop
                    if stopper(best_val):  # type: ignore
                        print("⏹️  early stopping")
                        save_ckpt(model, global_step, tag="final")
                        return

        # end of epoch checkpoint
        save_ckpt(model, global_step)

    save_ckpt(model, global_step, tag="final")
    print("🏁  Training complete")


if __name__ == "__main__":
    main()
