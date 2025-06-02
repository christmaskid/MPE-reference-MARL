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

## Testing Scripts

This repository also provides scripts for evaluating trained models:

- **test_maddpg.py**: Test MADDPG or its communication-loss variant.  
    Usage example:
    ```bash
    python test_maddpg.py --env_name multiple_reference_broadcast --n_agents 3 --episodes 3000 --render
    ```
    Key arguments:
    - `--env_name`: Name of the environment (required)
    - `--n_agents`: Number of agents (required)
    - `--episodes`: Number of episodes (default: 3000)
    - `--save_dir`: Directory for loading models/results (optional)
    - `--render`: Render environment and save GIFs (optional)
    - `--vanilla`: Use vanilla MADDPG without communication (optional)

- **test_sac.py**: Test MASAC (Multi-Agent SAC) models.  
    Usage example:
    ```bash
    python test_sac.py --env_name multiple_reference_broadcast --n_agents 3 --episodes 3000 --render
    ```
    Key arguments:
    - `--env_name`, `--n_agents`, `--episodes`, `--save_dir`, `--render` (same as above)

- **test_mappo.py**: Test MAPPO models.  
    Usage example:
    ```bash
    python test_mappo.py --env_name multiple_reference_broadcast --n_agents 3 --episodes 3000 --render
    ```
    Key arguments:
    - `--env_name`, `--n_agents`, `--episodes`, `--save_dir`, `--render` (same as above)

All testing scripts use `argparse` for flexible command-line configuration. See each script's help (`-h`) for details.
