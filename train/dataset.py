# train/dataset.py
import torch
from pathlib import Path

class ByteDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_len: int = 256):
        self.block_len = block_len
        data = Path(path).read_bytes()          # raw bytes
        self.tokens = torch.tensor(list(data), dtype=torch.uint8)

    def __len__(self):
        return len(self.tokens) - self.block_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_len + 1]  # +1 for label shift
        x  = chunk[:-1].long()      # input bytes
        y  = chunk[1:].long()       # next-byte labels
        return x, y
