import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Setting up GELU function
def GELU(x):
    return (0.5*x)*(1+math.tanh((2*math.pi()**-1)**0.5) * (x + (0.044715 * x**3)))

# Setting up classes (Layers)
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout_rate):
        super().__init__()
        self.n_embd = n_embd
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        
        wei = Q @ K.transpose(-2, -1) * (self.n_embd**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads])
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout_rate):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout_rate)
        self.ffwd = FeedForward(n_embd, dropout_rate)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        