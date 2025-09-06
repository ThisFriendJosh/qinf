import math
import random
from typing import List, Tuple

State = List[float]
Trajectory = List[Tuple[State, int, float, State]]
Vector = List[float]
Matrix = List[List[float]]


def matvec(m: Matrix, v: Vector) -> Vector:
    return [sum(w * x for w, x in zip(row, v)) for row in m]


class TrajectoryEncoder:
    """Encode a trajectory into a context vector."""
    def __init__(self, obs_dim: int, context_dim: int):
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.weight: Matrix = [[random.uniform(-0.1, 0.1) for _ in range(obs_dim)]
                               for _ in range(context_dim)]
        self.bias: Vector = [0.0 for _ in range(context_dim)]

    def __call__(self, traj: Trajectory) -> Vector:
        T = len(traj)
        mean = [sum(step[0][j] for step in traj) / T for j in range(self.obs_dim)]
        return [math.tanh(sum(w * m for w, m in zip(row, mean)) + b)
                for row, b in zip(self.weight, self.bias)]


class LoRAAdapter:
    """Generate LoRA weights conditioned on context."""
    def __init__(self, obs_dim: int, n_actions: int, rank: int, context_dim: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rank = rank
        self.WA: Matrix = [[random.uniform(-0.1, 0.1) for _ in range(context_dim)]
                           for _ in range(rank * obs_dim)]
        self.bA: Vector = [0.0 for _ in range(rank * obs_dim)]
        self.WB: Matrix = [[random.uniform(-0.1, 0.1) for _ in range(context_dim)]
                           for _ in range(n_actions * rank)]
        self.bB: Vector = [0.0 for _ in range(n_actions * rank)]

    def __call__(self, ctx: Vector) -> Tuple[Matrix, Matrix]:
        A_vec = matvec(self.WA, ctx)
        A_vec = [a + b for a, b in zip(A_vec, self.bA)]
        B_vec = matvec(self.WB, ctx)
        B_vec = [b + bb for b, bb in zip(B_vec, self.bB)]
        A = [A_vec[i * self.obs_dim:(i + 1) * self.obs_dim] for i in range(self.rank)]
        B = [B_vec[i * self.rank:(i + 1) * self.rank] for i in range(self.n_actions)]
        return A, B


class MetaLearner:
    """Meta-learner producing LoRA weights via trajectory context."""
    def __init__(self, obs_dim: int, n_actions: int, context_dim: int = 16, rank: int = 4):
        self.encoder = TrajectoryEncoder(obs_dim, context_dim)
        self.adapter = LoRAAdapter(obs_dim, n_actions, rank, context_dim)

    def encode(self, trajectory: Trajectory) -> Vector:
        return self.encoder(trajectory)

    def adapt(self, trajectory: Trajectory) -> Tuple[Matrix, Matrix]:
        ctx = self.encode(trajectory)
        return self.adapter(ctx)

    def q_values(self, qnet, x: Vector, trajectory: Trajectory) -> Vector:
        lora = self.adapt(trajectory)
        return qnet.forward(x, lora)
