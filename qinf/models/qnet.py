from __future__ import annotations
import torch
import torch.nn as nn


class DuelingQNet(nn.Module):
    """Minimal dueling architecture Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int] | None = None) -> None:
        super().__init__()
        hidden = hidden or [256, 256]
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.V = nn.Linear(hidden[1], 1)
        self.A = nn.Linear(hidden[1], n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        v = self.V(h)
        a = self.A(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
