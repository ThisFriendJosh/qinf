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
