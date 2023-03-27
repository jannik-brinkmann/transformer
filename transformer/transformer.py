from torch import nn
import torch
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(
        self, *,
        d_model: int = 512, 
        n_heads: int = 8,
        p_dropout: float = 0.1
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(
        self, *,
        x: torch.Tensor,
    ):

        # multi-head attention
        self_attn = self.self_attn(x=x)

        # add & norm
        x = self.norm(x + self.dropout(self_attn))

        # feed-forward

        # add & norm

        return x