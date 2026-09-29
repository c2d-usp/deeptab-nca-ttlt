"""
models.py : MLP architecture used for global training
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 num_layers=2, hidden_units=100,
                 task="classification"):
        super().__init__()

        layers = []
        last_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_units))
            layers.append(nn.Tanh())
            last_dim = hidden_units

        layers.append(nn.Linear(last_dim, output_dim))

        if task == "classification":
            layers.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)