import torch
from torch import nn

import sys
sys.path.append("./")

from src.blocks.Attention import Attention
from src.blocks.MLP import MLP
from src.blocks.Norm import Norm
from xformers.ops.swiglu_op import SwiGLU



class Transformer_Block(nn.Module):
    def __init__(self, dim, c_dim, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", causal=False, layer_idx=None):
        super().__init__()
        
        # MLP and attention blocks
        # self.MLP = MLP(dim, hidden_scale)
        self.MLP = SwiGLU(dim, int(dim*hidden_scale), dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_type=attn_type, causal=causal, layer_idx=layer_idx)
        
        # Two layer norms
        self.norm1 = Norm(dim, c_dim)
        self.norm2 = Norm(dim, c_dim)

        # Scale params
        self.scale1 = nn.Linear(c_dim, dim)
        self.scale2 = nn.Linear(c_dim, dim)

        
    def forward(self, X, y=None):
        # Attn layer
        X = (self.attn(self.norm1(X, y)) * self.scale1(y)[:, None, :]) + X
        
        # MLP layer
        return (self.MLP(self.norm2(X, y)) * self.scale2(y)[:, None, :]) + X