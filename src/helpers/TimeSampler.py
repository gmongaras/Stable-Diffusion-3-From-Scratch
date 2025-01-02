import torch
from torch import nn



class TimeSampler:
    def __init__(self, weighted=True, m=0.0, s=1.0):
        self.weighted = weighted
        self.m = m
        self.s = s

    def __call__(self, n):
        return self.sample(n)

    def sample(self, n):
        if self.weighted:
            # Sample n time points from a normal distribution
            u = torch.randn(n) * self.s + self.m

            # Map the samples to the range [0, 1] using the logistic function
            return torch.sigmoid(u)
        else:
            return torch.rand(n)