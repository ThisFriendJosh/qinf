from __future__ import annotations
from abc import ABC, abstractmethod

class MemoryStore(ABC):
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None: ...

    @abstractmethod
    def get(self, key: bytes) -> bytes | None: ...

    @abstractmethod
    def commit_snapshot(self, manifest: dict) -> str: ...
