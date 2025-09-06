"""Intrinsic reward functions implemented without third-party dependencies."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


class IntrinsicReward:
    """Base class for intrinsic reward calculators."""

    def compute(self, obs, next_obs, action, extras) -> float:  # pragma: no cover - interface
        return 0.0


class CuriosityReward(IntrinsicReward):
    """Random Network Distillation implemented with pure Python.

    A fixed random linear projection (the "target" network) is compared to a
    trainable linear predictor.  The prediction error after an update serves as
    a curiosity bonus.
    """

    def __init__(self, beta: float = 0.2, obs_dim: int | None = None,
                 feat_dim: int = 16, lr: float = 0.01):
        self.beta = beta
        self.obs_dim = obs_dim
        self.feat_dim = feat_dim
        self.lr = lr
        self.target_w: List[List[float]] | None = None
        self.pred_w: List[List[float]] | None = None
        if obs_dim is not None:
            self._init_weights(obs_dim)

    def _init_weights(self, obs_dim: int) -> None:
        rnd = random.Random(0)
        self.target_w = [[rnd.uniform(-1, 1) for _ in range(self.feat_dim)] for _ in range(obs_dim)]
        self.pred_w = [[0.0 for _ in range(self.feat_dim)] for _ in range(obs_dim)]

    def _ensure_weights(self, next_obs) -> None:
        if self.target_w is None or self.pred_w is None:
            flat = next_obs["flat"]
            self.obs_dim = len(flat)
            self._init_weights(self.obs_dim)

    def _forward(self, weights: List[List[float]], x: List[float]) -> List[float]:
        return [sum(x[i] * weights[i][j] for i in range(self.obs_dim)) for j in range(self.feat_dim)]

    def compute(self, obs, next_obs, action, extras) -> float:
        if self.beta == 0.0:
            return 0.0
        self._ensure_weights(next_obs)
        x = next_obs["flat"]
        phi = self._forward(self.target_w, x)  # type: ignore[arg-type]
        pred = self._forward(self.pred_w, x)  # type: ignore[arg-type]
        diff = [pred[j] - phi[j] for j in range(self.feat_dim)]
        mse = sum(d * d for d in diff) / self.feat_dim
        # gradient descent update for predictor weights
        for i in range(self.obs_dim):
            for j in range(self.feat_dim):
                grad = 2.0 * diff[j] * x[i] / self.feat_dim
                self.pred_w[i][j] -= self.lr * grad  # type: ignore[index]
        return self.beta * mse


class CompressionGainReward(IntrinsicReward):
    """Information gain based on simple frequency counts."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.counts: Dict[Tuple[int, ...], int] = {}
        self.total = 0

    def _key(self, obs) -> Tuple[int, ...]:
        return tuple(int(v) for v in obs["flat"])

    def compute(self, obs, next_obs, action, extras) -> float:
        if self.beta == 0.0:
            return 0.0
        key = self._key(next_obs)
        old_count = self.counts.get(key, 0)
        # Information gain of updating count from old_count to old_count+1.
        gain = 0.0 if old_count == 0 else math.log(old_count + 1) - math.log(old_count)
        self.counts[key] = old_count + 1
        self.total += 1
        return self.beta * gain
from __future__ import annotations

class IntrinsicReward:
    def compute(self, obs, next_obs, action, extras: dict) -> float:
        return 0.0

class CuriosityReward(IntrinsicReward):
    def __init__(self, beta: float = 0.2):
        self.beta = beta


    def compute(self, obs, next_obs, action, extras):
        # Placeholder: random small bonus to stimulate exploration
        return 0.0

class CompressionGainReward(IntrinsicReward):
    def __init__(self, beta: float = 0.1):
        self.beta = beta




    def compute(self, obs, next_obs, action, extras):
        return 0.0