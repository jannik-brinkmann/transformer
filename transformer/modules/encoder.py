import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderBlock(nn.Module):
    """one encoder block"""

    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, seq_len, p_dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff, p_dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # self-attention with residual connection
        x = self.ln1(x + self.self_attn(x, x, x, mask))

        # feed-forward with residual connection
        x = self.ln2(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, seq_len: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, seq_len, d_ff, p_dropout)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
        for block in self.blocks:
            x = block(x, mask)
        return x
