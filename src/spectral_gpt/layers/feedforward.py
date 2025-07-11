# src/spectral_gpt/layers/feedforward.py
import torch.nn as nn
import torch.nn.functional as F

class SpectralFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = F.elu(self.fc1(x)) + 1.0
        x = F.elu(self.fc2(x)) + 1.0
        return x
