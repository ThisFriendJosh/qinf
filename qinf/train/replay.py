from __future__ import annotations
import random


class Replay:
    """Basic FIFO replay buffer."""

    def __init__(self, capacity: int = 200_000, warmup: int = 10_000) -> None:
        self.capacity = capacity
        self.warmup = warmup
        self.buf: list[tuple] = []
        self.idx = 0

    def add(self, s, a, r, s2, done) -> None:
        item = (s["flat"], a, r, s2["flat"], done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.idx] = item
            self.idx = (self.idx + 1) % self.capacity

    def ready(self) -> bool:
        return len(self.buf) >= self.warmup

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)
