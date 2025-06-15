# train/train.py
import torch, time, math, argparse
from torch.utils.data import DataLoader
from spectral_gpt.model import QuectoCore
from train.dataset import ByteDataset

def bits_per_byte(loss):
    return loss / math.log(2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/tiny.bin")
    p.add_argument("--epochs",  type=int, default=1)
    p.add_argument("--bs",      type=int, default=64)
    p.add_argument("--seq",     type=int, default=256)
    p.add_argument("--lr",      type=float, default=2e-3)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    cfg = p.parse_args()

    ds = ByteDataset(cfg.data, cfg.seq)
    dl = DataLoader(ds, cfg.bs, shuffle=True, pin_memory=True)
    model = QuectoCore().to(cfg.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    tot_tokens = 0
    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for step, (x, y) in enumerate(dl):
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)                      # (B, L, 256)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), y.view(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 and step > 0:          # every 100 updates
                 torch.save(model.state_dict(), "chkpt_latest.pt")


            tot_tokens += x.numel()
            if step % 100 == 0:
                bpb = bits_per_byte(loss.item())
                tps = tot_tokens / (time.time() - start)
                print(f"EP{epoch}-STEP{step}  loss={loss:.4f}  bpb={bpb:.2f}  tok/s={tps:.0f}")
                print("MPS available:", torch.backends.mps.is_available())
                print("Device in use :", next(model.parameters()).device)


        torch.save(model.state_dict(), f"chkpt_ep{epoch}.pt")

if __name__ == "__main__":
    main()



