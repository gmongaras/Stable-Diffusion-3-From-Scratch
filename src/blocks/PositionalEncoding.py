import torch
from torch import nn
from colorama import Fore




class PositionalEncoding(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim

        # Calculate the denominator of the position encodings
        # as this value is constant
        self.denom = (torch.tensor(10000.0)**\
            ((2*torch.arange(self.dim))/self.dim)).to(dtype=torch.float, device=device)

    # Convert time steps to embedding tensors
    # Inputs:
    #   time - Time values of shape (N)
    # Outputs:
    #   embedded time values of shape (N, dim)
    def forward(self, time):
        # Compute the current timestep embeddings
        embeddings = time[:, None] / self.denom[None, :].to(time.device)

        # Sin/Cos transformation for even, odd indices
        embeddings = torch.cat((embeddings[:, ::2].sin(), embeddings[:, 1::2].cos()), dim=1)

        return embeddings
