#!/usr/bin/env python3
"""train.py – SGPT v3 training CLI with spectral‐symbolic diagnostics"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler

from spectral_gpt.model import QuectoCore
from train.aux_losses import omega_entropy, amplitude_l1
from train.stats import PhiEntropyHook, OmegaSpectrumHook, ASparsityHook


class ByteDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int):
        arr = np.fromfile(path, dtype=np.uint8)
        self.data = torch.from_numpy(arr).long()
        self.seq_len = seq_len
        self.n_samples = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.data[s : s + self.seq_len]
        y = self.data[s + 1 : s + 1 + self.seq_len]
        return x, y


@torch.inference_mode()
def eval_bpb(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    nll, ntok = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb, step=0)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum"
        )
        nll += loss.item()
        ntok += yb.numel()
    return nll / (ntok * np.log(2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train SGPT v3")
    p.add_argument("--data", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_schedule", choices=["none", "cosine"], default="none")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dmodel", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--fsp_rank", type=int, required=True, help="low-rank FFN")
    p.add_argument("--attn_type", choices=["sia"], default="sia")
    p.add_argument("--sia_r", type=int, required=True, help="spectral channels")
    p.add_argument("--diag_every", type=int, default=0, help="print diagnostics every N steps")
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--sigma_omega",  type=float, default=1e-3)
    p.add_argument("--sigma_A",      type=float, default=1e-3)
    p.add_argument("--noise_interval", type=int, default=200)
    p.add_argument("--lambda_H", type=float, default=1e-3)
    p.add_argument("--lambda_A", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"🔋 Using device: {device}")

    # Data loaders
    train_ds = ByteDataset(args.data, args.seq)
    train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=RandomSampler(train_ds))
    val_ds = ByteDataset(args.val, args.seq)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = QuectoCore(
        vocab_size=args.vocab_size,
        d_model=args.dmodel,
        depth=args.depth,
        max_seq_len=args.max_seq_len,
        fsp_rank=args.fsp_rank,
        attn_type=args.attn_type,
        sia_r=args.sia_r,
    ).to(device)

    # Spectral‐symbolic hooks
    phi_hook       = PhiEntropyHook(eps=1e-8)
    omega_hook     = OmegaSpectrumHook()
    asparsity_hook = ASparsityHook(threshold=1e-3)

    # Optimizer & scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = (
        optim.lr_scheduler.LambdaLR(opt, lambda step: (
            step / args.warmup_steps if step < args.warmup_steps else
            0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) /
                                (args.epochs * args.steps_per_epoch - args.warmup_steps)))
            if args.lr_schedule == "cosine" else 1.0
        ))
        if (args.warmup_steps or args.lr_schedule == "cosine") else None
    )



    # Training state
    patience, best_bpb = 0, float("inf")
    t_last, tok_last = time.time(), 0
    ema = optim.swa_utils.AveragedModel(model, device=device) if args.ema > 0 else None

    step = 0

    # Main loop
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            if step < 100:
                print(f"[Epoch {epoch} • Batch {batch_idx}] xb={xb.shape}", flush=True)

            # Forward & backward
            if step and step % args.noise_interval == 0:
                from train.chaos import inject_gaussian_noise 
                inject_gaussian_noise(model,
                                      sigma_omega=args.sigma_omega,
                                      sigma_A=args.sigma_A)
                
            logits = model(xb, step=step)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1)
            )
            
            if sched:
                sched.step()
            if ema and step >= args.ema_start:
                ema.update_parameters(model)

            aux_loss = 0.0
            H_all, A_all = [], [] 

            for blk in model.blocks:
                dlt = blk.self_interact.dlt 
                if args.lambda_H:
                    h = omega_entropy(dlt)
                    H_all.append(h)
                    aux_loss += args.lambda_H * h

                if args.lambda_A:
                    a = amplitude_l1(dlt)
                    A_all.append(a)
                    aux_loss += args.lambda_A * a

            if H_all:
                H_all = torch.stack(H_all)
            if A_all:
                A_all = torch.stack(A_all) 

            total_loss = ce_loss + aux_loss 
                    
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step < 1000:
                ramp = step / 1000.0
                target = torch.logit(torch.tensor(ramp + 1e-4, device=device))
                for blk in model.blocks:
                    blk.self_interact.g_logit.data.copy_(target) 

            # Spectral‐symbolic diagnostics
            if args.diag_every and step > 0 and step % args.diag_every == 0:
                total_norm = math.sqrt(
                    sum((p.grad.norm().item() ** 2) for p in model.parameters() if p.grad is not None)
                )
                for m in model.modules():
                    if hasattr(m, "prev_phi") and m.prev_phi is not None:
                        phi_hook(step, m)
                    if hasattr(m, "dlt"):
                        omega_hook(step, m.dlt)
                        asparsity_hook(step, m.dlt)
                if batch_idx % args.diag_every == 0:
                    for layer in model.blocks:
                        omega_hook(step, layer.self_interact.dlt)   # new
                        asparsity_hook(step, layer.self_interact.dlt)
                        
                def grad_ratio(model: torch.nn.Module) -> float:
                    sia_g2, ffn_g2 = 0.0, 0.0
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g2 = p.grad.detach().float().pow(2).sum().item()
                        if ".self_interact" in name:          # SIA branch
                            sia_g2 += g2
                        elif ".ffn" in name or ".mlp" in name:  # feed-forward branch
                            ffn_g2 += g2
                    return grad_ratio(model)
# ---------------------------------------------------------------------------




                phi_ent = phi_hook.logs[-1][1] if phi_hook.logs else 0.0
                ω_min, ω_mean, ω_max = omega_hook.logs[-1][1:] if omega_hook.logs else (0.0, 0.0, 0.0)
                sp, mean_a, std_a = asparsity_hook.logs[-1][1:] if asparsity_hook.logs else (0.0, 0.0, 0.0)

                

                print(
                    f"🩺 diag@{step}: grad_norm={total_norm:.4f} "
                    f"φ_ent={phi_ent:.4f} ω=[{ω_min:.2f},{ω_mean:.2f},{ω_max:.2f}] "
                    f"A_sp={sp:.3f} A_mean={mean_a:.3f} A_std={std_a:.3f}",
                    flush=True
                )
                phi_hook.logs.clear()
                omega_hook.logs.clear()
                asparsity_hook.logs.clear()

            # Progress logging every 100 steps
            if step % 100 == 0:
                now = time.time()
                tok_s = tok_last / (now - t_last + 1e-6)
                t_last, tok_last = now, 0
                bpb = ce_loss.item() / math.log(2)
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"EP{epoch}-STP{step:<5} loss={ce_loss:.4f} "
                    f"bpb={bpb:.2f} lr={lr_now:.1e} tok/s={tok_s:,.0f}",
                    flush=True
                )

            # Validation
            if step % args.eval_every == 0:
                ref_model = ema.module if ema else model
                val_bpb = eval_bpb(ref_model, val_loader, device)
                print(f"🧪 step {step}: val_bpb={val_bpb:.4f}", flush=True)
                if val_bpb < best_bpb:
                    best_bpb = val_bpb
                    Path("chkpts").mkdir(exist_ok=True)
                    torch.save(ref_model.state_dict(), "chkpts/best.pt")
                    print("⭐ best saved", flush=True)
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print("⏸ Early stopping", flush=True)
                        return

            step += 1
            tok_last += xb.numel()

            # Early stop end-of-epoch
            if step >= args.steps_per_epoch * (epoch + 1):
                print(f"Reached {args.steps_per_epoch} steps, ending epoch.", flush=True)
                break

    print("✔ Training complete.", flush=True)


if __name__ == "__main__":
    main()
