import numpy as np
import torch
import os

os.environ['SUPPRESS_MA_PROMPT']='1'
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env

from train_maddpg import MADDPG
import imageio

ENV_NAME = 'simple_reference'
N_AGENTS = 2
STEPS_PER_EPISODE = 40
SAVE_DIR = 'models/'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS)
    env.reset()

    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    act_u_dim = 4
    print("obs_dims:", obs_dims, "act_dims:", act_dims, flush=True)

    multi_agent = MADDPG(obs_dims, act_dims, act_u_dim, N_AGENTS, device)
    multi_agent.load_models(SAVE_DIR)

    returns = []
    obs = env.reset()
    obs = np.array(obs)
    total_reward = np.zeros((N_AGENTS,))
    frames = []

    for step in range(STEPS_PER_EPISODE):
        env_actions, actions = multi_agent.select_action(obs)
        next_obs, rewards, dones, info = env.step(env_actions)
        
        total_reward += rewards
        actions = np.array(actions)
        next_obs = np.array(next_obs)
        # rewards = 1 + np.clip(np.array(rewards), -10, 0) / 10
        rewards = np.array(rewards)
        dones = np.array(dones)

        multi_agent.add(obs, next_obs, actions, rewards, dones)
        actor_loss, critic_loss = multi_agent.train()
        obs = next_obs

        returns.append(total_reward.mean())

        frame = np.array(env.render(mode='rgb_array'))
        frames.append(frame[0])
        
        if dones.all():
            break
    
    # Save frames as gif
    save_path = 'maddpg_test_{}_{}.gif'.format(ENV_NAME, N_AGENTS)
    imageio.mimsave(save_path, frames, duration=0.05)
    print("Saved episode as {}".format(save_path))

if __name__ == "__main__":
    main()
