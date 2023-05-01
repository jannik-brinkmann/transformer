import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)
