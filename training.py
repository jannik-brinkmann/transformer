"""
    This training script can be used to train a Transformer on the TinyShakespeare dataset. 
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.nn import functional as F

from transformer import GPT


@dataclass
class ModelArguments:
    n_encoder_blocks: Optional[int] = field(
        default = 2, 
        metadata = {"description": ""}
    )
    n_decoder_blocks: Optional[int] = field(
        default = 2, 
        metadata = {"description": ""}
    )
    n_encoder_heads: Optional[int] = field(
        default = 2, 
        metadata = {"description": ""}
    )
    n_decoder_heads: Optional[int] = field(
        default = 2, 
        metadata = {"description": ""}
    )
    d_model: Optional[int] = field(
        default = 128, 
        metadata = {"description": ""}
    )
    d_ff: Optional[int] = field(
        default = 512, 
        metadata = {"description": ""}
    )
    p_dropout: Optional[float] = field(
        default = 0.2, 
        metadata = {"description": ""}
    )
    n_tokens: Optional[float] = field(
        default = 50304, 
        metadata = {"description": ""}
    )
    seq_len: Optional[int] = field(
        default = 8, 
        metadata = {"description": ""}
    )


@dataclass
class TrainingArguments:
    training_steps: Optional[int] = field(
        default = 1000, 
        metadata = {"description": ""}
    )
    data_dir: Optional[str] = field(
        default = "./data/", 
        metadata = {"description": ""}
    )
    train_size: Optional[float] = field(
        default = 0.9, 
        metadata = {"description": ""}
    )
    batch_size: Optional[int] = field(
        default = 32, 
        metadata = {"description": ""}
    )
    learning_rate: Optional[float] = field(
        default = 3e-4, 
        metadata = {"description": ""}
    )
    weight_decay: Optional[float] = field(
        default = 1e-1, 
        metadata = {"description": ""}
    )
    beta1: Optional[float] = field(
        default = 0.9, 
        metadata = {"description": ""}
    )
    beta2: Optional[float] = field(
        default = 0.95, 
        metadata = {"description": ""}
    )
    device: Optional[str] = field(
        default = 'cuda', 
        metadata = {"description": ""}
    )
    seed: Optional[int] = field(
        default = 1337, 
        metadata = {"description": ""}
    )
    evaluation_interval: Optional[int] = field(
        default = 20, 
        metadata = {"description": ""}
    )
    evaluation_iterations: Optional[int] = field(
        default = 200, 
        metadata = {"description": ""}
    )



def dataclass_to_args(dataclass_obj):
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(dataclass_obj):
        parser.add_argument(f"--{field.name}", default=getattr(dataclass_obj, field.name))
    args = parser.parse_args()
    return args


def main():

    training_args = dataclass_to_args(TrainingArguments())
    training_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(training_args.seed)

    # downloaded from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('./data/shakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # extract all unique characters in the input text that will make up the vocabulary of the model
    chars = sorted(list(set(text)))
    n_tokens = len(chars)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] 
    decode = lambda l: ''.join([itos[i] for i in l])

    # train-test split
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(training_args.train_size * len(data))
    training_data = data[:split]
    validation_data = data[split:]

    def get_batch(split):
        data = training_data if split == 'training' else validation_data
        samples = torch.randint(len(data) - model_args.seq_len, (training_args.batch_size,))
        X = torch.stack([data[i:i + model_args.seq_len] for i in samples])
        Y = torch.stack([data[i + 1:i + model_args.seq_len + 1] for i in samples])
        X, Y = X.to(training_args.device), Y.to(training_args.device)
        return X, Y

    # setup model
    model_args = ModelArguments(n_tokens=n_tokens)
    model = GPT(**dataclasses.asdict(model_args))
    model.to(training_args.device)

    # copied and edited from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
    @torch.no_grad()
    def estimate_loss():
        outputs = {}
        model.eval()
        for split in ['training', 'validation']:
            samples = torch.zeros(training_args.evaluation_iterations)
            for k in range(training_args.evaluation_iterations):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                samples[k] = loss.item()
            outputs[split] = samples.mean()
        model.train()
        return outputs

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    for step in range(training_args.training_steps):

        if step % training_args.evaluation_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['training']:.4f}, val loss {losses['validation']:.4f}")

        X, Y = get_batch('training')
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
