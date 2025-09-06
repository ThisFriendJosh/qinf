from __future__ import annotations
from typing import Protocol, Any, Dict, Tuple

class Env(Protocol):
    def reset(self, *, seed: int | None = None) -> Tuple[Any, Dict]: ...
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]: ...
    @property
    def action_space(self) -> int: ...
    @property
    def observation_space(self) -> Tuple[int, ...]: ...

class Task:
    """Env + goal spec; placeholder for curriculum support."""
    def __init__(self, env: Env, goal: dict | None = None):
        self.env, self.goal = env, (goal or {})
