# MPE-reference-MARL
**Multi-agent Cooperative Communication in Speaking-Listening Environment**
Deep Reinforcement learning, Spring 2025. Group 20 Final project.

A clone from the OpenAI [MPE](https://github.com/openai/multiagent-particle-envs) environment.

## Environment
```bash
pip install -r requirements.txt
```
## Training Scripts

This repository provides several training scripts for multi-agent reinforcement learning algorithms:

- **train_maddpg.py**: Trains agents using the MADDPG algorithm on the MPE environments.
- **train_maddpg_modified.py**: A variant of MADDPG with communication loss modifications.
- **train_SAC.py**: Implements Multi-Agent Soft Actor-Critic (MASAC) training with communication loss.
- **train_ppo.py**: Trains agents using Proximal Policy Optimization (PPO) with direct communication environment.
- **train_ppo_broadcast.py**: PPO training with broadcast communication environment.

Each script supports configurable hyperparameters and saves model checkpoints and training curves. See the script headers and code comments for usage details.

Example usage:
```bash
python train_maddpg.py
```