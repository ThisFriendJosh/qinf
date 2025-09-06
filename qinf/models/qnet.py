from __future__ import annotations

import torch
import torch.nn as nn

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

import torch, torch.nn as nn

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int] = [256,256]):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.V = nn.Linear(hidden[1], 1)
        self.A = nn.Linear(hidden[1], n_actions)

    def forward(self, x):
        h = self.body(x)
        v = self.V(h)
        a = self.A(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
