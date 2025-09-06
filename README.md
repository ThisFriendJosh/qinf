# Q∞ (qinf)

Minimal scaffold for hierarchical + meta-RL with intrinsic rewards and optional persistent memory/ledger. Start with `python scripts/run_toy.py` after installing deps. TODOs are marked in code.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)]()

Q∞ is a minimalist scaffold for hierarchical and meta reinforcement learning with intrinsic rewards
and optional persistent memory. The repository ships with a toy grid-world example and is meant to
be extended into more capable agents.

## Quickstart

Clone the repository and install it in editable mode:

```bash
pip install -e .
python scripts/run_toy.py
```

The toy script launches a small grid-world environment with a dueling DQN, a single option, and
simple curiosity and compression-gain rewards.

## Key modules

The stack is organized into several top-level packages:

- `qinf.envs.gridtoy`: lightweight grid-world for experimentation.
- `qinf.models.qnet`: Dueling Q-network and related helpers.
- `qinf.train.replay`: replay buffer with optional prioritization.
- `qinf.train.learner`: training loop and target network updates.
- `qinf.hierarchy.options`: starter option classes such as `GoToKey`.
- `qinf.intrinsic.rewards`: intrinsic reward signals (curiosity, compression gain).
- `qinf.runtime.logging`: minimal logger for metrics and checkpoints.

## Configuration

All experiment settings live in `configs/default.yml`. Notable sections include:

- **experiment** – run id, RNG seed, total steps, logging and evaluation cadence.
- **env** – grid size, stochasticity, reward shaping and curriculum stages.
- **model** – Q-network architecture, target update interval and double-Q toggle.
- **hierarchy** – available options and epsilon-soft scheduler settings.
- **meta** – optional LoRA-style adapter and update frequency.
- **intrinsic** – curiosity and compression-gain coefficients.
- **memory** – LMDB-backed persistent storage and ledger mode.
- **optim** – learning rate, batch size, discount factor and replay buffer settings.

Adjust the YAML file or provide your own to explore different setups.

## Development roadmap

- Swap in a stronger curiosity model (e.g. forward/inverse dynamics).
- Add additional options like `GoToDoor` and `GoToGoal` and compare against flat DQN.
- Enable `ledger_mode` once long-term persistence is needed.
- Extend prioritized replay and experiment with LoRA-style meta adapters.
- Add continuous integration, unit tests and richer examples.

## References

- [Deep Q-Networks](https://www.nature.com/articles/nature14236) – Mnih et al., 2015.
- [Option-Critic](https://proceedings.mlr.press/v70/bacon17a.html) – Bacon et al., 2017.
- [Intrinsic Curiosity Module](https://arxiv.org/abs/1705.05363) – Pathak et al., 2017.
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) – Hu et al., 2021.

## License

The project is distributed under the MIT License. See `LICENSE` for details.
