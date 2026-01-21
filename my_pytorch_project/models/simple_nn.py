import torch
import torch.nn as nn

from .base_model import BaseModel


class SimpleNN(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
