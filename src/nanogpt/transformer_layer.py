import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    ''' An implementation of a simple feed-forward network. '''


    def __init__(self, n_embed, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(n_embed, 4 * n_embed),
                    nn.ReLU(),
                    nn.Linear(4 * n_embed, n_embed), # projection back into the residual pathway
                    nn.Dropout(dropout) # implementing dropout to reduce overfitting
        )

    def forward(self, x):
        return self.net(x)
    
class Head(nn.Module):
    ''' One head of self-attention as found in the 'Attention is All You Need' paper. '''

    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.query(x) # (B,T,C)
        q = self.key(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        # compute the attention pattern (QK^T / sqrt(head_size))
        A = q @ k.transpose(-2, -1) * C**-0.5 #(B,T,C) x (B,C,T) -> (B,T,T)
        # prevent future logits from attending to past logits
        A = A.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        # softmax 
        A = F.softamx(A, dim=-1) # (B,T,T) softmaxing along the last dimension means each row of the (T,T) matrix is softmaxed
        # implement dropout to reduce overfitting
        A = self.dropout(A)
        # matrix multiply the softmaxed tensor with V, softmax(QK^T/sqrt(head_size)) @ V
        output = A @ v # (B,T,T) x (B,T,C) -> (B,T,C)
        return output
    
class MultiHeadAttention(nn.Module):
    ''' An implementation of multi-headed attention as found in the 'Attention is All You Need' paper. '''

    def __init__(self, num_heads: int, n_embed: int, block_size: int, dropout: int):
        super().__init__()

        if n_embed % num_heads != 0: # this must be true, as we concatenate all outputs of all heads to rebuild a (B,T,'C') tensor
            raise ValueError(
                f"n_embed ({n_embed}) is not divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.n_embed = n_embed
        self.head_size = self.n_embed // self.num_heads
        self.heads = nn.ModuleList(Head(head_size=self.head_size, n_embed=n_embed, block_size=block_size) for _ in range(num_heads))
        self.out_proj = nn.Linear(self.n_embed, self.n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.out_proj(x) # projection back into the residual pathway
        x = self.dropout(x) # implement dropout to reduce overfitting
        return x

class TransformerLayer(nn.Module):
    ''' An implementation of a transformer layer form the 'Attention is All You Need' paper. '''

    def __init__(self,
                n_embed: int,
                block_size: int,
                num_heads: int,
                dropout: float
                ):
        
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.sa = MultiHeadAttention(num_heads= self.num_heads,
                                     n_embed= self.n_embed, 
                                     block_size= self.block_size,
                                     dropout= self.dropout
                                     )
        self.ffwd = FeedForward(n_embed=self.n_embed, dropout= self.dropout)
        self.ln1 = nn.LayerNorm(self.n_embed) 
        self.ln2 = nn.LayerNorm(self.n_embed)  
        
    def forward(self, x):
        x =  x + self.sa(self.ln1(x)) # implemented residual connection and layer normalisation
        x = x + self.ffwd(self.ln2(x)) # implemented residual connection and layer normalisation
        return x
    