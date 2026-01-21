import torch
import torch.nn as nn


class ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return self.linear(x) * self.scale
