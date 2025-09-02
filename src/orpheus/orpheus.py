import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig

from transformer_layer import TransformerLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).parent.parent.parent


class Orpheus(pl.LightningModule):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.model.block_size
        self.vocab_size = cfg.model.vocab_size
        self.n_embed = cfg.model.n_embed
        self.dropout = cfg.train.dropout
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.positional_embedding_table = nn.Embedding(self.block_size, self.n_embed)

        # Define the architecture for the decoder
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    num_heads=self.num_heads,
                    n_embed=self.n_embed,
                    block_size=self.block_size,
                    dropout=self.dropout,
                )
            )
        layers.append(nn.LayerNorm(self.n_embed))
        self.layers = nn.ModuleList(layers)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)

    def forward(self, x: torch.Tensor):
        # ensure indices are longs for embedding lookup
        if x.dtype != torch.long:
            x = x.long()
        B, T = x.shape  # B = batch_size, T = block_size
        C = self.n_embed
        # batch and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(
            x
        )  # replaces each token (int) in each sequence with an embedding vector such that tok_emb is (B,T,C)
        pos_emb = self.positional_embedding_table(
            torch.arange(T, device=x.device)
        )  # (T,C)
        x = (
            tok_emb + pos_emb
        )  # Note torch uses broadcasting to convert pos_emb (T,C) to pos_emb (B,T,C) for this calculation (stacks B copies aling dimension 1)
        x = self.layers(x)  # attention, feedforward layers, (B,T,C)
        logits = self.lm_head(
            x
        )  # (B,T,vocab_size) if L = nn.Linear(input_dim, output_dim), W = (output_dim, input_dim) and b = (output_dim)

        return logits

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits, loss = self.forward(x)
        if y.dtype != torch.long:
            y = y.long()
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)
        loss = F.cross_entropy(logits, y)
        self.log("train/Loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits, loss = self.forward(x)
        if y.dtype != torch.long:
            y = y.long()
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)
        loss = F.cross_entropy(logits, y)
        self.log("val/Loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            lr=self.cfg.model.optimizer.lr,
            betas=self.cfg.model.optimizer.adam_betas,
            eps=self.cfg.model.optimizer.adam_eps,
            weight_decay=self.cfg.model.optimizer.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=self.cfg.model.optimizer.warmup_updates,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.model.optimizer.end_learning_rate
            / self.cfg.model.optimizer.lr,
            total_iters=int(0.9 * int(self.cfg.model.optimizer.total_num_update)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.cfg.model.optimizer.warmup_updates],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # Remove dropout by putting model into eval mode
        self.eval()
        # idx is (B,T) array of indices in the current context
        model_device = next(self.parameters()).device
        if idx.device != model_device:
            idx = idx.to(model_device)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (to be compatible with the postional encodings)
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(
                idx_cond
            )  # this is the nn.Module way of calling self.foward(x)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get the probabilities, apply to each batch's embedding of length 'vocab_size'
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.concat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
