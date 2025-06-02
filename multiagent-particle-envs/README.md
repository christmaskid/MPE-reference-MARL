# Multi-Agent Particle Environment

This is a modified version of [openai/multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs) environment, which is a simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

## Code structure

- `make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./multiagent/environment-conti.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.).  Here we use a **continuous**-action version instead of the original discrete one. We import this in `./make_env.py` instead of original `./multiagent/environment.py`.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored.

## List of environments


| Env name in code (name in paper) |  Communication? | Competitive? | Notes |
| --- | --- | --- | --- |
| `multiple_reference_broadcast.py` | Y | N | N agents, N landmarks. Each agent is assigned a target landmark, but only another agent knows which landmark is their target. Agents must communicate via a broadcast channel to help each other reach their goals. Reward is based on reaching the correct landmark. |
| `multiple_reference_direct.py` | Y | N | N agents, N landmarks. Each agent is assigned a target landmark, and agents communicate directly (not via broadcast) to inform each other of their goals. Reward is based on reaching the correct landmark. |

## Reference

Environments in this repo:
<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>

Original particle world environment:
<pre>
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
</pre>
