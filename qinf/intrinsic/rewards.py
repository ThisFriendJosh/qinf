from __future__ import annotations


class IntrinsicReward:
    def compute(self, obs, next_obs, action, extras: dict) -> float:
        return 0.0


class CuriosityReward(IntrinsicReward):
    def __init__(self, beta: float = 0.2) -> None:
        self.beta = beta

    def compute(self, obs, next_obs, action, extras: dict) -> float:
        # Placeholder: no intrinsic reward by default
        return 0.0


class CompressionGainReward(IntrinsicReward):
    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute(self, obs, next_obs, action, extras: dict) -> float:
        return 0.0
