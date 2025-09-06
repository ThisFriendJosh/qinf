from __future__ import annotations
from abc import ABC, abstractmethod


class Option(ABC):
    @abstractmethod
    def should_start(self, obs) -> bool:
        ...

    @abstractmethod
    def policy(self, obs) -> int:
        ...

    @abstractmethod
    def should_terminate(self, obs, step_count: int) -> bool:
        ...


class OptionScheduler(ABC):
    @abstractmethod
    def select(self, obs, available_options: list[Option]):
        ...


class GoToKey(Option):
    def should_start(self, obs) -> bool:
        return obs.get("needs_key", False)

    def policy(self, obs) -> int:
        return int(obs.get("suggested_action_to_key", 0))

    def should_terminate(self, obs, t: int) -> bool:
        return obs.get("at_key", False) or t > 50
