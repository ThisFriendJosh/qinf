import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from qinf.models.qnet import QNet
from qinf.meta.meta_learner import MetaLearner


def make_traj(value: float, obs_dim: int):
    s = [value for _ in range(obs_dim)]
    return [(s, 0, 0.0, s)]


def diff(v1, v2):
    return any(abs(a - b) > 1e-9 for a, b in zip(v1, v2))


def test_adaptation_changes_q_values():
    random.seed(0)
    obs_dim, n_actions = 4, 2
    q = QNet(obs_dim, n_actions)
    meta = MetaLearner(obs_dim, n_actions, context_dim=8, rank=2)

    x = [0.5 for _ in range(obs_dim)]
    traj = make_traj(1.0, obs_dim)
    base = q.forward(x)
    adapted = meta.q_values(q, x, traj)

    assert diff(base, adapted)


def test_different_contexts_produce_different_q_values():
    random.seed(0)
    obs_dim, n_actions = 4, 2
    q = QNet(obs_dim, n_actions)
    meta = MetaLearner(obs_dim, n_actions, context_dim=8, rank=2)

    x = [0.5 for _ in range(obs_dim)]
    traj1 = make_traj(1.0, obs_dim)
    traj2 = make_traj(0.0, obs_dim)

    q1 = meta.q_values(q, x, traj1)
    q2 = meta.q_values(q, x, traj2)

    assert diff(q1, q2)
