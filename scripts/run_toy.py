from __future__ import annotations
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from qinf.envs.gridtoy import make_env
from qinf.models.qnet import DuelingQNet
from qinf.train.replay import Replay
from qinf.train.learner import QLearner
from qinf.hierarchy.options import GoToKey
from qinf.intrinsic.rewards import CuriosityReward, CompressionGainReward
from qinf.runtime.logging import Logger

class EpsilonSoftScheduler:
    def __init__(self, eps_start=1.0, eps_end=0.05, steps=50000):
        self.eps_start, self.eps_end, self.steps = eps_start, eps_end, steps
        self.t = 0

    def eps(self):
        k = max(0.0, (self.steps - self.t) / self.steps)
        return self.eps_end + (self.eps_start - self.eps_end) * k

    def select(self, obs, opts):
        self.t += 1
        use_option = random.random() > self.eps()
        return random.choice(opts) if (use_option and opts) else None

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def main(cfg_path="configs/default.yml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["experiment"]["seed"])
    logger = Logger(Path("./runs"), cfg["experiment"]["id"])

    env = make_env(size=cfg["env"]["size"], stochasticity=cfg["env"]["stochasticity"],
                   reward_step=cfg["env"]["reward_step"], reward_goal=cfg["env"]["reward_goal"])
    obs, _ = env.reset()
    obs_dim, n_actions = env.observation_dim, env.action_space

    q = DuelingQNet(obs_dim, n_actions)
    tgt = DuelingQNet(obs_dim, n_actions)
    tgt.load_state_dict(q.state_dict())
    optim = torch.optim.Adam(q.parameters(), lr=cfg["optim"]["lr"])
    replay_cfg = cfg["optim"]["replay"]
    replay = Replay(
        capacity=replay_cfg["capacity"],
        warmup=replay_cfg["warmup"],
        alpha=replay_cfg.get("alpha", 0.6),
        beta0=replay_cfg.get("beta0", 0.4),
    )
    learner = QLearner(
        q,
        tgt,
        optim,
        replay,
        gamma=cfg["optim"]["gamma"],
        n_step=cfg["optim"]["n_step"],
        double_q=cfg["model"]["double_q"],
    )

    options = [GoToKey()]  # Add more later
    scheduler = EpsilonSoftScheduler(**cfg["hierarchy"]["scheduler"])
    r_cur = (
        CuriosityReward(**cfg["intrinsic"]["curiosity"])
        if cfg["intrinsic"]["curiosity"]["enabled"]
        else CuriosityReward(0.0)
    )
    r_cmp = (
        CompressionGainReward(**cfg["intrinsic"]["compression_gain"])
        if cfg["intrinsic"]["compression_gain"]["enabled"]
        else CompressionGainReward(0.0)
    )

    episodic_return, step = 0.0, 0
    obs, _ = env.reset()
    total_steps = cfg["experiment"]["total_steps"]
    while step < total_steps:
        available = [o for o in options if o.should_start(obs)]
        opt = scheduler.select(obs, available)
        if opt is not None:
            t = 0
            while True:
                a = opt.policy(obs)
                next_obs, r_ext, terminated, truncated, extras = env.step(a)
                r_int = r_cur.compute(obs, next_obs, a, extras) + r_cmp.compute(obs, next_obs, a, extras)
                replay.add(obs, a, r_ext + r_int, next_obs, terminated)
                obs = next_obs
                episodic_return += r_ext
                step += 1
                t += 1
                if terminated or truncated or opt.should_terminate(obs, t):
                    break
        else:
            x = torch.tensor(obs["flat"]).float().unsqueeze(0)
            if random.random() < scheduler.eps():
                a = random.randrange(n_actions)
            else:
                a = int(torch.argmax(q(x), dim=1))
            next_obs, r_ext, terminated, truncated, extras = env.step(a)
            r_int = r_cur.compute(obs, next_obs, a, extras) + r_cmp.compute(obs, next_obs, a, extras)
            replay.add(obs, a, r_ext + r_int, next_obs, terminated)
            obs = next_obs
            episodic_return += r_ext
            step += 1

        if replay.ready():
            replay.anneal_beta(step / total_steps)
            batch = replay.sample(cfg["optim"]["batch_size"])
            logs = learner.step(batch)
            if step % cfg["experiment"]["log_every"] == 0:
                logger.log(step=step, loss=logs["loss"], ret=episodic_return)
                print(step, logs)
            if step % cfg["model"]["target_update_interval"] == 0:
                tgt.load_state_dict(q.state_dict())

        if terminated or truncated:
            obs, _ = env.reset()
            episodic_return = 0.0

if __name__ == "__main__":
    main()
