from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

DEFAULT_LOW_PCT  = 0.50   # bottom 50 %
DEFAULT_HIGH_PCT = 0.75   # top   25 %

def load_entropy_index(
    npz_path: str | Path,
    low_pct:  float = DEFAULT_LOW_PCT,
    high_pct: float = DEFAULT_HIGH_PCT,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
        offsets   : int64 array of window start positions
        bands     : int8  array (0=low, 1=mid, 2=high)
        thresholds: dict(low_th=…, high_th=…)
    """
    npz = np.load(npz_path)
    offsets   = npz["offsets"]
    entropies = npz["entropies"]

    low_th  = np.quantile(entropies, low_pct)
    high_th = np.quantile(entropies, high_pct)

    bands = np.zeros_like(entropies, dtype=np.int8)
    bands[entropies >= low_th]  = 1        # mid
    bands[entropies >= high_th] = 2        # high

    return offsets, bands, {"low_th": float(low_th), "high_th": float(high_th)}
