import torch
from torch import nn

import sys
sys.path.append("./")

from src.blocks.Attention import Attention
from src.blocks.MLP import MLP
from src.blocks.Norm import Norm
from xformers.ops.swiglu_op import SwiGLU



class Transformer_Block_Dual(nn.Module):
    def __init__(self, dim, c_dim, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", MLP_type = "gelu", causal=False, positional_encoding="absolute", RoPE_Scale=1, kv_merge_attn=False, qk_half_dim=False, checkpoint_MLP=True, checkpoint_attn=True, layer_idx=None, last=False):
        super().__init__()

        self.checkpoint_MLP = checkpoint_MLP
        self.checkpoint_attn = checkpoint_attn

        # On the last block, we don't worry about the c stream
        self.last = last

        # y input projection
        self.y_proj = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.SiLU()
        )
        
        # MLP and attention blocks
        if MLP_type == "swiglu_old":
            self.MLP_x = SwiGLU(dim, int(dim*hidden_scale), dim)
            if not self.last:
                self.MLP_c = SwiGLU(dim, int(dim*hidden_scale), dim)
        else:
            self.MLP_x = MLP(dim, hidden_scale, act=MLP_type)
            if not self.last:
                self.MLP_c = MLP(dim, hidden_scale, act=MLP_type)
        self.attn = Attention(dim, num_heads=num_heads, attn_type=attn_type, causal=causal, positional_encoding=positional_encoding, RoPE_Scale=RoPE_Scale, kv_merge_attn=kv_merge_attn, qk_half_dim=qk_half_dim, layer_idx=layer_idx, dual=True, last=last)
        
        # Two layer norms
        self.norm1_x = Norm(dim, c_dim)
        self.norm2_x = Norm(dim, c_dim)
        self.norm1_c = Norm(dim, c_dim)
        if not self.last:
            self.norm2_c = Norm(dim, c_dim)

        # GLU gates
        self.scale1_x = nn.Linear(c_dim, dim, bias=False)
        self.scale2_x = nn.Linear(c_dim, dim, bias=False)
        if not self.last:
            self.scale1_c = nn.Linear(c_dim, dim, bias=False)
            self.scale2_c = nn.Linear(c_dim, dim, bias=False)

        
    def forward(self, X, c, y, orig_shape):
        y = self.y_proj(y)

        # Attn layer
        if self.checkpoint_attn:
            X_, c_ = torch.utils.checkpoint.checkpoint(self.attn, self.norm1_x(X, y), self.norm1_c(c, y), orig_shape)
        else:
            X_, c_ = self.attn(self.norm1_x(X, y), self.norm1_c(c, y), orig_shape)
        X = (X_ * self.scale1_x(y)[:, None, :]) + X
        if not self.last:
            c = (c_ * self.scale1_c(y)[:, None, :]) + c
        
        # MLP layer
        if self.checkpoint_MLP:
            X = (torch.utils.checkpoint.checkpoint(self.MLP_x, self.norm2_x(X, y)) * self.scale2_x(y)[:, None, :]) + X
            if not self.last:
                c = (torch.utils.checkpoint.checkpoint(self.MLP_c, self.norm2_c(c, y)) * self.scale2_c(y)[:, None, :]) + c
        else:
            X = (self.MLP_x(self.norm2_x(X, y)) * self.scale2_x(y)[:, None, :]) + X
            if not self.last:
                c = (self.MLP_c(self.norm2_c(c, y)) * self.scale2_c(y)[:, None, :]) + c

        return X, c