import math

import torch
import torch.nn as nn


class SinusoisalPositionalEncodings(nn.Module):
    """sine and cosine function-based positional encodings"""

    def __init__(self, seq_len: int, d_model: int, device: str) -> None:
        super().__init__()

        # create a matrix to store sinusoidal positional encodings
        self.encodings = torch.zeros(seq_len, d_model).to(device)
        self.encodings.requires_grad = False

        # compute sine and cosine function-based positional encodings
        positions = torch.arange(0, seq_len).float().unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encodings[:, 0::2] = torch.sin(positions * denominator)
        self.encodings[:, 1::2] = torch.cos(positions * denominator)
        self.encodings.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encodings[:x.size(1)]


class TokenEmbeddingsWithSinusoidalPositionalEncodings(nn.Module):
    """combines learned token embeddings with static positional encodings"""

    def __init__(self, n_tokens: int, d_model: int, seq_len: int, device: str) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, d_model)
        self.positional_encodings = SinusoisalPositionalEncodings(seq_len, d_model, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_encodings = self.positional_encodings(x)
        return token_embeddings + positional_encodings


class TokenEmbeddingsWithLearnedPositionalEmbeddings(nn.Module):
    """combines learned token embeddings and learned positional encodings"""

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, d_model)
        self.positional_embeddings = nn.Embedding(n_tokens, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_encodings = self.positional_embeddings(x)
        return token_embeddings + positional_encodings
