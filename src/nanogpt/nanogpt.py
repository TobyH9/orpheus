import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

from .transformer_layer import TransformerLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).parent.parent.parent


class NanoGPT(nn.Module):

    def __init__(
        self,
        block_size=256,
        vocab_size=65,
        n_embed=384,
        dropout=0.2,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.dropout = dropout
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, self.n_embed)
        self.layers = nn.Sequential(
            TransformerLayer(
                num_heads=5,
                n_embed=n_embed,
                block_size=self.block_size,
                dropout=self.dropout,
            ),
            TransformerLayer(
                num_heads=5,
                n_embed=n_embed,
                block_size=self.block_size,
                dropout=self.dropout,
            ),
            TransformerLayer(
                num_heads=5,
                n_embed=n_embed,
                block_size=self.block_size,
                dropout=self.dropout,
            ),
            TransformerLayer(
                num_heads=5,
                n_embed=n_embed,
                block_size=self.block_size,
                dropout=self.dropout,
            ),
            TransformerLayer(
                num_heads=5,
                n_embed=n_embed,
                block_size=self.block_size,
                dropout=self.dropout,
            ),
            nn.LayerNorm(self.n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3)

    def forward(self, batch: torch.Tensor, targets: torch.Tensor | None = None):
        # ensure indices are longs for embedding lookup
        if batch.dtype != torch.long:
            batch = batch.long()
        B, T = batch.shape  # B = batch_size, T = block_size
        C = self.n_embed
        # batch and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(
            batch
        )  # replaces each token (int) in each sequence with an embedding vector such that tok_emb is (B,T,C)
        pos_emb = self.positional_embedding_table(
            torch.arange(T, device=batch.device)
        )  # (T,C)
        x = (
            tok_emb + pos_emb
        )  # Note torch uses broadcasting to convert pos_emb (T,C) to pos_emb (B,T,C) for this calculation (stacks B copies aling dimension 1)
        x = self.layers(x)  # attention, feedforward layers, (B,T,C)
        logits = self.lm_head(
            x
        )  # (B,T,vocab_size) if L = nn.Linear(input_dim, output_dim), W = (output_dim, input_dim) and b = (output_dim)
        if targets is None:
            loss = None
        else:
            if targets.dtype != torch.long:
                targets = targets.long()
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # idx is (B,T) array of indices in the current context
        model_device = next(self.parameters()).device
        if idx.device != model_device:
            idx = idx.to(model_device)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (to be compatible with the postional encodings)
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(
                batch=idx_cond
            )  # this is the nn.Module way of calling self.foward(batch=idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get the probabilities, apply to each batch's embedding of length 'vocab_size'
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.concat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


"""    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out"""
