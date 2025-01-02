import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import random

class EloisHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout = 0.1):
        super().__init__()
        self.head_size = head_size
        # self.mask_proj = nn.Linear(n_embd, head_size, bias=False)
        self.values_proj = nn.Linear(n_embd, head_size, bias=False)
        self.cov = nn.Parameter(torch.randn(head_size, head_size) * head_size ** -0.5)
        # self.loading = nn.Parameter(torch.randn(head_size, head_size) * head_size ** -0.5)
        self.bias = nn.Parameter(torch.randn(head_size))

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        values = self.values_proj(x[:,0,:])  # (B, hs)
        mask = self.values_proj(x[:,1,:]) # (B, hs)
        weighted_avg_coefficients = torch.diag_embed(mask) @ self.cov # (B, hs, hs)
        
        weighted_avg_coefficients = F.softmax(weighted_avg_coefficients, dim=-1)
        # weighted_avg_coefficients = self.dropout(weighted_avg_coefficients)
        
        # Add dimension to values using unsqueeze
        values = values.unsqueeze(1)  # (B, 1, hs)
        out = values @ weighted_avg_coefficients + self.bias  # (B, 1, hs)
        
        return out[:,0,:]
    

class MultiHeadLayer(nn.Module):
    """Multiple EloisHead in parallel."""

    def __init__(self, n_embd, n_head, head_size, dropout = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([EloisHead(n_embd=n_embd, head_size=head_size, dropout=dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mh = MultiHeadLayer(n_embd=n_embd, n_head=n_head, head_size=head_size)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        out = self.mh(x)
        out = out + self.ffwd(out)
        out = torch.cat([out.unsqueeze(1), x[:,1,:].unsqueeze(1)], dim=1)

        return out


class FinancialDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training:
            return x
            
        # x shape is (batch_size, 2, n_embd)
        values = x[:, 0, :]  # (batch_size, n_embd)
        masks = x[:, 1, :]   # (batch_size, n_embd)
        
        # Create random dropout mask
        random_mask = (torch.rand_like(values) > self.drop_prob).float()
        
        # Apply dropout to values and masks
        new_values = values * random_mask
        new_masks = masks * random_mask
        
        # Recombine into original format
        return torch.stack([new_values, new_masks], dim=1)

class EloisNet(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, output_size, dropout_prob=0.4):
        super().__init__()
        self.financial_dropout = FinancialDropout(dropout_prob)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.lm_head = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, output_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, x, targets=None):
        x = self.financial_dropout(x)  # Apply financial dropout during training
        x = self.blocks(x)  # (B,2,n_emb)
        logits = self.lm_head(x[:,0,:])  # (B, output_size)

        if targets is None:
            return logits, None
            
        loss = F.l1_loss(logits, targets)
        return logits, loss

