# MPE-reference-MARL
**Multi-agent Cooperative Communication in Speaking-Listening Environment**
Deep Reinforcement learning, Spring 2025. Group 20 Final project.

A clone from the OpenAI [MPE](https://github.com/openai/multiagent-particle-envs) environment.

## Environment
```bash
conda create -n mpe_marl python=3.10
conda activate mpe_marl
pip install -r requirements.txt
```

## Training & Testing Scripts

This repository provides several scripts for training and evaluating multi-agent reinforcement learning algorithms on MPE environments. All scripts use `argparse` for flexible command-line configuration, supporting similar arguments for consistency.

### Common Arguments

- `--env_name`: Name of the environment (required)
- `--n_agents`: Number of agents (required)
- `--episodes`: Number of training episodes (default: 3000)
- `--save_dir`: Directory to load/save models and results (optional)
- `--render`: Render environment and save GIFs (optional, for testing scripts)
- `--vanilla`: Use vanilla MADDPG without communication (optional, MADDPG only)

Run any script with `-h` to see all available options.

### Training Scripts

- **train_maddpg_vanilla.py**: Train agents using the MADDPG algorithm.
- **train_maddpg.py**: MADDPG variant with communication loss.
- **train_masac.py**: Multi-Agent Soft Actor-Critic (MASAC) with communication loss.
- **train_mappo.py**: Proximal Policy Optimization (PPO).

Example:
```bash
python train_maddpg.py --env_name multiple_reference_broadcast --n_agents 3 --episodes 3000 --save_dir results/
```

### Testing Scripts

- **test_maddpg.py**: Evaluate MADDPG or its communication-loss variant.
- **test_sac.py**: Evaluate MASAC models.
- **test_mappo.py**: Evaluate MAPPO models.

Example:
```bash
python test_maddpg.py --env_name multiple_reference_broadcast --n_agents 3 --episodes 3000 --render
```

Each script saves model checkpoints and training curves as appropriate. See script headers and code comments for further usage details.
