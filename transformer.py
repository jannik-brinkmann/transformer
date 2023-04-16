import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SinusoisalPositionalEncoding(nn.Module):
    """sine and cosine function-based positional encodings"""

    def __init__(self, *, seq_len: int, d_model: int, device: str) -> None:
        super().__init__()

        # create a matrix to store sinusoidal positional encodings
        self.encodings = torch.zeros(seq_len, d_model).to(device)
        self.encodings.requires_grad = False

        # compute sine and cosine function-based positional encodings
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encodings[:, 0::2] = torch.sin(positions * denominator)
        self.encodings[:, 1::2] = torch.cos(positions * denominator)
        self.encodings.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encodings[:x.size(1)]


class EmbeddingWithPositionalEncoding(nn.Module):
    """combines learned token embeddings and static positional encodings"""

    def __init__(self, n_tokens: int, d_model: int, seq_len: int, device: str) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, d_model)
        self.positional_encodings = SinusoisalPositionalEncoding(seq_len=seq_len, d_model=d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_encodings = self.positional_encodings(x)
        return token_embeddings + positional_encodings


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
        dot = q @ k.transpose(-2,-1) * k.shape[-1] ** (1 / 2)  # shape: [batch_size, seq_len, seq_len]
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = torch.cat([h(q, k, v) for h in self.heads], dim=-1)
        outputs = self.dropout(self.U(outputs))
        return outputs


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


class EncoderBlock(nn.Module):
    """one encoder block"""

    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int, p_dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, seq_len, p_dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, p_dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # self-attention with residual connection
        x = x + self.self_attn(*[self.ln1(x)] * 3)

        # feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x


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

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        # self-attention with residual connection
        x = x + self.self_attn(*[self.ln1(x)] * 3)

        # cross-attention with residual connection
        x = x + self.cross_attn(*[self.ln2(x)] + [encoder_outputs] * 2)

        # feed-forward with residual connection
        x = x + self.ff(self.ln3(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self, 
        n_tokens: int, 
        d_model: int,
        seq_len: int,
        n_encoder_blocks: int, 
        n_decoder_blocks: int, 
        n_encoder_heads: int, 
        n_decoder_heads: int, 
        d_ff: int, 
        p_dropout: float, 
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.embeddings = EmbeddingWithPositionalEncoding(n_tokens, d_model, seq_len, 'cuda')
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_encoder_heads, seq_len, d_ff, p_dropout)
            for _ in range(n_encoder_blocks)
        ])
        self.ln1 = nn.LayerNorm(d_model)

        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_decoder_heads, seq_len, d_ff, p_dropout)
            for _ in range(n_decoder_blocks)
        ])
        self.ln2 = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, n_tokens)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # transform inputs to embeddings with positional encodings
        x = self.embeddings(x)

        # encoder with self-attention and feed-forward
        encoder_outputs = x.detach().clone()
        for block in self.encoder:
            encoder_outputs = block(encoder_outputs)
        encoder_outputs = self.ln1(encoder_outputs)

        # decoder with self-attention, cross-attention, and feed-forward
        for block in self.decoder:
            x = block(x, encoder_outputs)
        x = self.ln2(x)

        # transform output embeddings to probability distribution
        return self.head(x)

    def generate(self, context: torch.Tensor, n_predictions: int) -> torch.Tensor:
        
        for _ in range(n_predictions):

            # select the last tokens as input to the model
            context = context[:, -self.seq_len:]

            # get predictions
            logits = self(context)

            # select next-token prediction
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append next-token to the context
            context = torch.cat((context, next_token), dim=1)
        return context
