import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(
        self, *, 
        d_model: int = 512,
        d_ff: int = 2048,
        p_dropout: float = 0.1
    ):
        super(FeedForward, self).__init__()

        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

        self.activation = F.relu()
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, *,
        x: torch.Tensor
    ):

        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        x = self.dropout(self.activation(self.layer1(x)))
        x = self.layer2(x)
        return x
