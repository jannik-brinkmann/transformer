import torch
from torch import nn

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class EncoderModule(nn.Module):

    def __init__(
        self, *,
        d_model: int = 512, 
        n_heads: int = 8,
        p_dropout: float = 0.1
    ):
        super(EncoderModule, self).__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)

        self.ff = FeedForward(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_dropout)
    
    def forward(
        self, *,
        x: torch.Tensor,
    ):

        # multi-head attention
        self_attn = self.self_attn(x=x)

        # add & norm
        x = self.norm1(x + self.dropout1(self_attn))

        # feed-forward
        ff = self.ff(x)
        
        # add & norm
        x = self.norm2(x + self.dropout2(ff))

        return x


class Encoder(nn.Module):

    def __init__(
        self, *,
        n_modules: int = 6,
        d_model: int = 512, 
        n_heads: int = 8,
        p_dropout: float = 0.1
    ):
        super(Encoder, self).__init__()

        self.modules = nn.ModuleList([
            EncoderModule(d_model=d_model, n_heads=n_heads, p_dropout=p_dropout)
            for _ in range(n_modules)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, *,
        x: torch.Tensor
    ):
        for m in self.modules:
            x = m(x=x)
        return self.norm(x)
