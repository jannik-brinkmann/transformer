import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import Encoder, Decoder, TokenEmbeddingsWithLearnedPositionalEmbeddings


class GPT(nn.Module):
    """decoder Transformer model"""

    def __init__(
        self, 
        n_tokens: int, 
        d_model: int,
        seq_len: int,
        n_decoder_blocks: int, 
        n_decoder_heads: int, 
        d_ff: int, 
        p_dropout: float, 
        device: str = 'cuda',
        *args, **kwargs
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.embeddings = TokenEmbeddingsWithLearnedPositionalEmbeddings(n_tokens, d_model, seq_len, p_dropout)
        self.ln_f = nn.LayerNorm(d_model)
        self.decoder = Decoder(d_model, n_decoder_blocks, n_decoder_heads, seq_len, d_ff, p_dropout)
        self.head = nn.Linear(d_model, n_tokens)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.embeddings(idx) # (B,T,C)
        x = self.decoder(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.seq_len:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx

class Transformer(nn.Module):
    """encoder-decoder Transformer model"""

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
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.embeddings = TokenEmbeddingsWithSinusoidalPositionalEncodings(n_tokens, d_model, seq_len, p_dropout, device)
        
        self.encoder = Encoder(d_model, n_encoder_blocks, n_encoder_heads, seq_len, d_ff, p_dropout)
        self.decoder = Decoder(d_model, n_decoder_blocks, n_decoder_heads, seq_len, d_ff, p_dropout)
        
        self.head = nn.Linear(d_model, n_tokens)

        self.apply(self._init_weights)

    def generate_mask(self, src, tgt):
        src_mask = (src != -1).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != -1).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to('cuda')
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.generate_mask(inputs, outputs)

        # maps input sequence of symbol representations to sequence of continuous representations
        input_embedding = self.embeddings(inputs)
        z = self.encoder(input_embedding, src_mask)

        # generates output sequence of symbols one element at a time
        output_embedding = self.embeddings(outputs)
        x = self.decoder(output_embedding, z, src_mask, tgt_mask)

        # linear transformation and softmax function to convert decoder output to next-token probabilities
        x = self.head(x)
        
        return x

    def generate(self, context: torch.Tensor, n_predictions: int) -> torch.Tensor:
        
        for _ in range(n_predictions):

            # select the last tokens as input to the model
            inputs = context[:, -self.seq_len:]

            # get predictions
            logits = self(inputs, inputs[:, :-1])

            # select next-token prediction
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append next-token to the context
            context = torch.cat((context, next_token), dim=1)
        return context
