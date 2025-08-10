import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = Path(__file__).parent.parent.parent

class Transformer(nn.Module):

    def __init__(self, vocab_size= int, block_size= int, n_embd= 32):
        super().__init__()
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd) 
        self.positional_embedding_table = nn.Embedding(block_size, self.n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, batch= torch.Tensor, targets=None):
        B, T = batch.shape # B = batch_size, T = block_size
        C = self.n_embd
        # batch and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(batch) # replaces each token (int) in each sequence with an embedding vector such that tok_emb is (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # Note torch uses broadcasting to convert pos_emb (T,C) to pos_emb (B,T,C) for this calculation (stacks B copies aling dimension 1)
        logits = self.lm_head(x) # (B,T,vocab_size) if L = nn.Linear(input_dim, output_dim), W = (output_dim, input_dim) and b = (output_dim)
