# tests/test_train.py
import torch
from spectral_gpt.model import QuectoCore
from train.dataset import ByteDataset

def test_one_step():
    ds = ByteDataset("data/tiny.bin", 64)
    x, y = ds[0]
    model = QuectoCore()
    logits = model(x.unsqueeze(0))
    assert logits.shape == (1, 64, 256)
