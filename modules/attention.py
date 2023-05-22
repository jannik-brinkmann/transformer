import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """one head of attention"""

    def __init__(self, d_model: int, d_head: int, seq_len: int, p_dropout: float) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = q.shape

        # compute linear transformation
        q = self.W_q(q)  # shape: [batch_size, seq_len, d_model] 
        k = self.W_k(k)  
        v = self.W_v(v)

        # compute scaled dot-product attention
        dot = q @ k.transpose(-2,-1) * k.shape[-1] ** -(1 / 2)  # shape: [batch_size, seq_len, seq_len]
        dot = dot.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        dot = F.softmax(dot, dim=-1)
        dot = self.dropout(dot)
        outputs = dot @ v 
        return outputs


class MultiHeadAttention(nn.Module):
    """multiple heads of attention"""
    
    def __init__(self, d_model: int, n_heads: int, seq_len: int, p_dropout: float) -> None:
        super().__init__()
        d_head = d_model // n_heads

        self.heads = nn.ModuleList([Head(d_model, d_head, seq_len, p_dropout) for _ in range(n_heads)])
        self.U = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do_masking: bool = False) -> torch.Tensor:
        outputs = torch.cat([h(q, k, v, do_masking) for h in self.heads], dim=-1)
        outputs = self.dropout(self.U(outputs))
        return outputs
