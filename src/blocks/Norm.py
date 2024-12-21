import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, dim, c_dim):
        super().__init__()
        
        # Normalize the data. Only the class information is used to shift and scale the data.
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # Class shift and scale
        self.c_shift = nn.Linear(c_dim, dim)
        self.c_scale = nn.Linear(c_dim, dim)
        
    def forward(self, X, y=None):
        # Layer norm
        X = self.norm(X)
        
        # Class conditioning
        if type(y) != type(None):
            X = (X * (1 + self.c_scale(y)[:, None, :])) + self.c_shift(y)[:, None, :]
            
        return X