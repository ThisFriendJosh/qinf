from __future__ import annotations
import random

class Replay:
    def __init__(self, capacity=200000, warmup=10000):
        self.capacity = capacity
        self.warmup = warmup
        self.buf = []
        self.idx = 0
    def add(self, s, a, r, s2, done):
        item = (s["flat"], a, r, s2["flat"], done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.idx] = item
            self.idx = (self.idx + 1) % self.capacity
    def ready(self):
        return len(self.buf) >= self.warmup
    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)
