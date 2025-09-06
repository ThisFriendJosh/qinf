from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

A_UP, A_DOWN, A_LEFT, A_RIGHT = 0, 1, 2, 3

class GridToy:
    def __init__(self, size=8, stochasticity=0.05, reward_step=-0.01, reward_goal=1.0):
        self.size = size
        self.p = stochasticity
        self.r_step = reward_step
        self.r_goal = reward_goal
        self.action_space = 4
        self.observation_dim = size * size
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent = np.array([0, 0])
        self.key = np.array([self.size // 2, self.size // 2])
        self.door = np.array([self.size - 2, self.size - 2])
        self.goal = np.array([self.size - 1, self.size - 1])
        self.has_key = False
        return self._obs(), {}

    def step(self, a: int):
        a = int(a)
        if self._rng.random() < self.p:
            a = self._rng.integers(0, 4)
        if a == A_UP:
            self.agent[0] = max(0, self.agent[0] - 1)
        if a == A_DOWN:
            self.agent[0] = min(self.size - 1, self.agent[0] + 1)
        if a == A_LEFT:
            self.agent[1] = max(0, self.agent[1] - 1)
        if a == A_RIGHT:
            self.agent[1] = min(self.size - 1, self.agent[1] + 1)
        r = self.r_step
        if (self.agent == self.key).all():
            self.has_key = True
        terminated = False
        if (self.agent == self.goal).all() and self.has_key:
            r += self.r_goal
            terminated = True
        truncated = False
        return self._obs(), float(r), terminated, truncated, {}

    def _obs(self):
        flat = np.zeros((self.observation_dim,), dtype=np.float32)
        idx = self.agent[0] * self.size + self.agent[1]
        flat[idx] = 1.0
        obs = {
            "flat": flat,
            "needs_key": not self.has_key,
            "at_key": bool((self.agent == self.key).all()),
            "at_goal": bool((self.agent == self.goal).all()),
            "suggested_action_to_key": int(np.random.randint(0, 4)),
        }
        return obs


def make_env(**cfg):
    return GridToy(**cfg)
