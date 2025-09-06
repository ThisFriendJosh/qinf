from __future__ import annotations
from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """Abstract key-value store used for persisting run data."""

    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        """Store a value under ``key``."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: bytes) -> bytes | None:
        """Retrieve a value for ``key`` if present."""
        raise NotImplementedError

    @abstractmethod
    def commit_snapshot(self, manifest: dict) -> str:
        """Persist a snapshot manifest and return its identifier."""
        raise NotImplementedError
class MemoryStore(ABC):
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None: ...


    @abstractmethod
    def get(self, key: bytes) -> bytes | None: ...


    @abstractmethod
    def get(self, key: bytes) -> bytes | None: ...

    @abstractmethod
    def commit_snapshot(self, manifest: dict) -> str: ...
