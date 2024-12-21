import torch
from torch import nn



class MLP(nn.Module):
    def __init__(self, dim, hidden_scale=4.0, act="swiglu"):
        super().__init__()
        
        self.proj_size = int(dim*hidden_scale)
        
        self.act_ = act

        if act == "swiglu":
            self.lin_up = nn.Linear(dim, int(self.proj_size*2))
            self.lin_down = nn.Linear(self.proj_size, dim)
            self.act = nn.functional.silu
        elif act == "gelu":
            self.lin_up = nn.Linear(dim, self.proj_size)
            self.lin_down = nn.Linear(self.proj_size, dim)
            self.act = nn.functional.gelu
        
    def forward(self, X):
        if self.act_ == "swiglu":
            # Up projection
            up_proj, gate = self.lin_up(X).split(self.proj_size, dim=-1)
            
            # Gate
            up_proj = up_proj * self.act(gate)
        else:
            # Up projection
            up_proj = self.lin_up(X)
            up_proj = self.act(up_proj)
        
        # Output projection
        return self.lin_down(up_proj)