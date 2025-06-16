import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

class EntropyBandedDataset(Dataset):
    """
    Dataset that samples windows of bytes from a .bin file,
    stratified by local byte-level entropy bands.

    Args:
      bin_path: Path to raw .bin file
      idx_path: Path to .npz file containing 'offsets' and 'entropies'
      seq_len: Length of each sample window
      low_pct: Fraction of low-entropy samples in each batch (0-1)
      high_pct: Fraction of high-entropy samples (0-1)
      window: size used for entropy computation (must match index)
      stride: stride used for entropy computation
    """
    def __init__(
        self,
        bin_path: str,
        idx_path: str,
        seq_len: int,
        low_pct: float = 0.33,
        high_pct: float = 0.33,
    ):
        # load raw data
        data = pathlib.Path(bin_path).read_bytes()
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.seq_len = seq_len
        # load entropy index
        npz = np.load(idx_path)
        offsets = npz['offsets']
        entropies = npz['entropies']
        # compute thresholds
        low_th = np.quantile(entropies, low_pct)
        high_th = np.quantile(entropies, 1 - high_pct)
        # assign bands
        self.low_idxs  = offsets[entropies <= low_th]
        self.high_idxs = offsets[entropies >= high_th]
        mid_mask = (entropies > low_th) & (entropies < high_th)
        self.mid_idxs  = offsets[mid_mask]
        # sampling fractions
        self.low_frac  = low_pct
        self.high_frac = high_pct

    def __len__(self):
        # virtually infinite sampling
        return 10_000_000

    def __getitem__(self, idx):
        # randomly choose a band
        r = np.random.rand()
        if r < self.low_frac and len(self.low_idxs):
            pool = self.low_idxs
        elif r > 1 - self.high_frac and len(self.high_idxs):
            pool = self.high_idxs
        else:
            pool = self.mid_idxs
        # pick random offset from band
        start = int(np.random.choice(pool))
        # ensure we don't run off the end
        max_start = len(self.data) - self.seq_len - 1
        start = min(start, max_start)
        x = self.data[start:start + self.seq_len]
        y = self.data[start+1:start + 1 + self.seq_len]
        return x, y
