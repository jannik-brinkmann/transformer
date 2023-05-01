import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderBlock(nn.Module):
    """one decoder block"""

    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, seq_len, p_dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, seq_len, p_dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, p_dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, src_mask, tgt_mask) -> torch.Tensor:

        # self-attention with residual connection
        x = self.ln1(x + self.self_attn(x, x, x, tgt_mask))

        # cross-attention with residual connection
        x = self.ln2(x + self.cross_attn(x, encoder_outputs, encoder_outputs, src_mask))

        # feed-forward with residual connection
        x = self.ln3(x + self.ff(x))
        return x


class Decoder(nn.Module):
    
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, seq_len: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, seq_len, d_ff, p_dropout)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, src_mask, tgt_mask) -> torch.Tensor: 
        for block in self.blocks:
            x = block(x, encoder_outputs, src_mask, tgt_mask)
        return x
