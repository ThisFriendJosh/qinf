from __future__ import annotations
from abc import ABC, abstractmethod


class Option(ABC):
    """Abstract base for an option in the hierarchy."""

    @abstractmethod
    def should_start(self, obs) -> bool:
        """Return True if the option should be initiated given current observation."""
        raise NotImplementedError

    @abstractmethod
    def policy(self, obs) -> int:
        """Return the primitive action proposed by this option."""
        raise NotImplementedError

    @abstractmethod
    def should_terminate(self, obs, step_count: int) -> bool:
        """Return True if option should terminate after step_count steps."""
        raise NotImplementedError


class OptionScheduler(ABC):
    """Interface for option schedulers."""

    @abstractmethod
    def select(self, obs, available_options: list[Option]):
        raise NotImplementedError


class GoToKey(Option):
    """Simple heuristic option to navigate toward the key."""

    def should_start(self, obs):
        return obs.get("needs_key", False)

    def policy(self, obs):
        return int(obs.get("suggested_action_to_key", 0))

    def should_terminate(self, obs, t):
        return obs.get("at_key", False) or t > 50


class GoToDoor(Option):
    """Navigate to the door once the agent has picked up the key."""

    def should_start(self, obs):
        # Start after the key is collected and before reaching the goal.
        has_key = not obs.get("needs_key", True)
        return has_key and not obs.get("at_goal", False)

    def policy(self, obs):
        # Environment may provide a suggested action toward the door; default to 0.
        return int(obs.get("suggested_action_to_door", 0))

    def should_terminate(self, obs, t):
        # Stop when standing on the door or after too many steps.
        return obs.get("at_door", False) or t > 50


class GoToGoal(Option):
    """Final option to move from the door to the goal."""

    def should_start(self, obs):
        # Start once the agent has reached the door.
        return obs.get("at_door", False)

    def policy(self, obs):
        return int(obs.get("suggested_action_to_goal", 0))

    def should_terminate(self, obs, t):
        return obs.get("at_goal", False) or t > 50
