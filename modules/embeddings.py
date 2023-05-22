import math

import torch
import torch.nn as nn


class SinusoisalPositionalEncodings(nn.Module):
    """sine and cosine function-based positional encodings"""

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()

        # create a matrix to store sinusoidal positional encodings
        encodings = torch.zeros(seq_len, d_model)

        # compute sine and cosine function-based positional encodings
        positions = torch.arange(0, seq_len).float().unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encodings[:, 0::2] = torch.sin(positions * denominator)
        encodings[:, 1::2] = torch.cos(positions * denominator)

        self.register_buffer('encodings', encodings.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encodings[:, :x.size(1)]


class TokenEmbeddingsWithSinusoidalPositionalEncodings(nn.Module):
    """combines learned token embeddings with static positional encodings"""

    def __init__(self, n_tokens: int, d_model: int, seq_len: int, p_dropout: float) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, d_model)
        self.positional_encodings = SinusoisalPositionalEncodings(seq_len, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_encodings = self.positional_encodings(x)
        return self.dropout(token_embeddings + positional_encodings)


class TokenEmbeddingsWithLearnedPositionalEmbeddings(nn.Module):
    """combines learned token embeddings and learned positional embeddings"""

    def __init__(self, n_tokens: int, d_model: int, seq_len: int, p_dropout: float) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, d_model)
        self.positional_embeddings = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_encodings = self.positional_embeddings(torch.arange(x.size(1), device='cuda'))
        return self.dropout(token_embeddings + positional_encodings)
