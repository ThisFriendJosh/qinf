from __future__ import annotations
import random
import numpy as np

class Replay:
    def __init__(self, capacity=200000, warmup=10000, alpha: float = 0.6, beta0: float = 0.4):
        self.capacity = capacity
        self.warmup = warmup
        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.buf: list[tuple] = []
        self.priorities: list[float] = []
        self.idx = 0
        self._max_priority = 1.0

    def add(self, s, a, r, s2, done):
        item = (s["flat"], a, r, s2["flat"], done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
            self.priorities.append(self._max_priority)
        else:
            self.buf[self.idx] = item
            self.priorities[self.idx] = self._max_priority
            self.idx = (self.idx + 1) % self.capacity

    def ready(self):
        return len(self.buf) >= self.warmup

    def sample(self, batch_size: int):
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        idxs = random.choices(range(len(self.buf)), weights=probs.tolist(), k=batch_size)
        batch = [self.buf[i] for i in idxs]
        weights = (len(self.buf) * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        return batch, idxs, weights

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            p = abs(float(p)) + 1e-6
            self.priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = p

    def anneal_beta(self, frac: float):
        self.beta = min(1.0, self.beta0 + (1.0 - self.beta0) * frac)
