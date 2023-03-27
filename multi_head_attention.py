from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self, *,
        d_model: int = 512, 
        n_heads: int = 8, 
        p_dropout: float = 0.1
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # intermediate and output dimension
        self.n_heads = n_heads  # number of attention heads

        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)

        self.U = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_dropout)


    def forward(
        self, *,
        q: torch.Tensor, 
        k: torch.Tensor,  
        v: torch.Tensor,  
        mask: Optional[torch.Tensor]=None
    ):  
        b, t, k = q.size()  # batch size, number of tokens, embedding dimension
        h = self.n_heads
        s = k // h  # output dimension of each attention head

        # compute projection for all attention heads; note: q, k, v are the same input
        q = self.W_q(q).transpose(1, 2).contiguous().view(b * h, t, s)
        k = self.W_q(k).transpose(1, 2).contiguous().view(b * h, t, s)
        v = self.W_q(v).transpose(1, 2).contiguous().view(b * h, t, s)

        # scaled dot-product attention for all attention heads
        dot = torch.bmm(q, k.transpose(1, 2))
        dot = dot / (k ** (1 / 2))
        dot = F.softmax(dot, dim=2)
        output = torch.bmm(dot, v).view(b, h, t, s)
        output = output.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.U(output)
