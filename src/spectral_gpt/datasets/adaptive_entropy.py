from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict

from spectral_gpt.datasets.entropy_index import load_entropy_index


class AdaptiveEntropySampler(Dataset):
    """
    Curriculum sampler that changes entropy-band mix every few epochs.
    Schedule default → one row ≈ 2 training epochs.

    idx | low  mid  high
    ---------------------
     0 | 1.0  0.0  0.0   (epoch 0-1)
     1 | 0.6  0.3  0.1   (epoch 2-3)
     2 | 0.3  0.5  0.2   (epoch 4-5)
     3 | 0.1  0.4  0.5   (epoch 6-7)
     4 | 0.0  0.3  0.7   (epoch ≥8)
    """

    DEFAULT_SCHEDULE: List[Dict[str, float]] = [
        {"low": 1.0, "mid": 0.0, "high": 0.0},
        {"low": 0.6, "mid": 0.3, "high": 0.1},
        {"low": 0.3, "mid": 0.5, "high": 0.2},
        {"low": 0.1, "mid": 0.4, "high": 0.5},
        {"low": 0.0, "mid": 0.3, "high": 0.7},
    ]

    def __init__(
        self,
        bin_path: str | Path,
        idx_path: str | Path,
        seq_len: int,
        schedule: List[Dict[str, float]] | None = None,
    ):
        raw = Path(bin_path).read_bytes()
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len

        offsets, bands, _ = load_entropy_index(idx_path)
        self.band_offsets = {
            0: offsets[bands == 0],
            1: offsets[bands == 1],
            2: offsets[bands == 2],
        }

        self.schedule = schedule or self.DEFAULT_SCHEDULE
        self.set_epoch(0)

    # ------------- curriculum API -------------
    def set_epoch(self, epoch: int):
        idx = min(epoch // 2, len(self.schedule) - 1)
        s   = self.schedule[idx]
        self.weights = {
            0: s["low"],
            1: s["mid"],
            2: s["high"],
        }

    # ------------- torch Dataset --------------
    def __len__(self):
        return 10_000_000  # virtual

    def __getitem__(self, idx):  # noqa: D401
        r = np.random.rand()
        if r < self.weights[0]:
            band = 0
        elif r < self.weights[0] + self.weights[1]:
            band = 1
        else:
            band = 2

        start = int(np.random.choice(self.band_offsets[band]))
        start = min(start, len(self.data) - self.seq_len - 2)

        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + 1 + self.seq_len]
        return x, y
