import torch
import random

class GaussianNoiseAug(torch.nn.Module):
    def __init__(self, mean=0.0, std_range=(0.001, 0.01)):
        super().__init__()
        self.mean = mean
        self.std_range = std_range

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3  # (B, C, T)
        B, _, _ = x.size()
        stds = torch.empty(B).uniform_(*self.std_range).to(x.device).view(B, 1, 1)
        noise = torch.randn_like(x) * stds + self.mean
        return x + noise