import random
from typing import List, Tuple

Vector = List[float]
Matrix = List[List[float]]


def matvec(m: Matrix, v: Vector) -> Vector:
    return [sum(w * x for w, x in zip(row, v)) for row in m]


def matadd(m1: Matrix, m2: Matrix) -> Matrix:
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(m1, m2)]


def matmul(m1: Matrix, m2: Matrix) -> Matrix:
    out_rows = len(m1)
    out_cols = len(m2[0])
    return [[sum(m1[i][k] * m2[k][j] for k in range(len(m2))) for j in range(out_cols)] for i in range(out_rows)]


class QNet:
    """Simple linear Q-network with optional LoRA adaptation."""
    def __init__(self, obs_dim: int, n_actions: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.weight: Matrix = [[random.uniform(-0.1, 0.1) for _ in range(obs_dim)]
                               for _ in range(n_actions)]
        self.bias: Vector = [0.0 for _ in range(n_actions)]

    def forward(self, x: Vector, lora: Tuple[Matrix, Matrix] | None = None) -> Vector:
        weight = self.weight
        if lora is not None:
            A, B = lora  # A: (r, obs_dim), B: (n_actions, r)
            rank = len(A)
            BA = matmul(B, A)
            delta = [[BA[i][j] / rank for j in range(self.obs_dim)] for i in range(self.n_actions)]
            weight = matadd(weight, delta)
        return [sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(weight, self.bias)]
