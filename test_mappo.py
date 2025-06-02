import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env
from collections import deque
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
os.environ['SUPPRESS_MA_PROMPT']='1'
from train_ppo_share_weight_broadcast_big import PPONet
import time

N_AGENTS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = make_env("multiple_reference", n_agents=N_AGENTS)
env = make_env("multiple_reference_broadcast", n_agents=N_AGENTS)
env.reset()
np.random.seed(0)
obs_dims = env.observation_space[0].shape[0]
act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])

class PPOAgent:
    def __init__(self, env, device):
        self.n_agents = N_AGENTS
        self.device = device

        self.obs_dims = env.observation_space[0].shape[0]
        self.act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])

        # print(f"Observation space dimension: {self.obs_dims}")
        # print(f"Action space dimension: {self.act_dims}")
        
        self.model = [PPONet(self.obs_dims, self.act_dims).to(self.device) for i in range(self.n_agents)]

        for i in range(self.n_agents):
            # self.model[i].load_state_dict(torch.load(f"PPO_agent_{i}.pth"))
            # self.model[i].load_state_dict(torch.load(f"PPO_agent_{N_AGENTS}_agent_2.pth"))
            self.model[i].load_state_dict(torch.load(f"PPO_agent_{N_AGENTS}_agent_broadcast_1.pth"))
            self.model[i].eval()

device = torch.device("cuda")
agent = PPOAgent(env ,device)
agents_total_reward = 0

frames = []
for i in range(10):
    obs = env.reset()
    obs = np.array(obs)

    total_reward = np.zeros(len(env.agents))
    done = False
    step = 0
    STEPS_PER_EPISODE = 25
    while not done:
        # actions_env, actions = agent.select_action(obs, noise_scale=0)
        with torch.no_grad():
            obs = [torch.tensor(obs[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
            outputs = [agent.model[i].get_act(obs[i]) for i in range(N_AGENTS)]
            move, comm, actions_n, log_prob_n = zip(*outputs)
            action_input = [[move[i].cpu().numpy(), comm[i].cpu().numpy()] for i in range(N_AGENTS)]

        next_obs, rewards, dones, info = env.step(action_input)
        
        if i == 0:
            frames.append(Image.fromarray(np.array(env.render(mode='rgb_array')[0])))
            time.sleep(0.1)

        total_reward += rewards
        next_obs = np.array(next_obs)
        dones = np.array(dones)
        done = dones.all()
        if step > STEPS_PER_EPISODE:
            done = True
        obs = next_obs
        step += 1

    agents_total_reward += total_reward.sum()

# for i in range(N_AGENTS):
#     ga = env.world.agents[i].goal_a.state.p_pos
#     gb = env.world.agents[i].goal_b.state.p_pos
#     c = env.agents[i].state.c
#     print(f'agent {i}:\n{obs[i]}')
#     print(f'{ga} to {gb} (mse:{((ga-gb)**2).mean()})')
#     print(f'{c}\n')

print("Total reward: ", agents_total_reward / (10 * N_AGENTS))
# frames[0].save(f"test_{N_AGENTS}_agent.gif", save_all=True, append_images=frames[1:], optimize=True, loop=0)
frames[0].save(f"test_{N_AGENTS}_agent_broadcast.gif", save_all=True, append_images=frames[1:], optimize=True, loop=0)