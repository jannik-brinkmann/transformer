import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SinusoisalEncodings(nn.Module):
    """sine and cosine function-based positional encodings"""

    def __init__(
        self, *,
        d_model: int = 512,
        seq_len: int = 8,
        device: str = 'cuda'
    ) -> None:
        super().__init__()

        # create a matrix to store sinusoidal positional encodings
        self.encodings = torch.zeros(seq_len, d_model).to(device)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encodings[:, 0::2] = torch.sin(positions * denominator)
        self.encodings[:, 1::2] = torch.cos(positions * denominator)
        self.encodings.unsqueeze(0)

    def forward(
        self, *, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """add positional encoding to token embedding"""
        return inputs + self.encodings[:inputs.size(1)]