import numpy as np
import torch
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env
import os
from PIL import Image
os.environ['SUPPRESS_MA_PROMPT']='1'
from train_mappo import PPONet
import time
import argparse
import imageio

class PPOAgent:
    def __init__(self, env, n_agents, device):
        self.n_agents = n_agents
        self.device = device

        self.obs_dims = env.observation_space[0].shape[0]
        self.act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])

        print(f"Observation space dimension: {self.obs_dims}")
        print(f"Action space dimension: {self.act_dims}")
        
        self.model = [PPONet(self.obs_dims, self.act_dims, self.n_agents).to(self.device) for i in range(self.n_agents)]

    def load_models(self, save_dir):
        for i in range(self.n_agents):
            self.model[i].load_state_dict(torch.load(os.path.join(save_dir, f"PPO_agent_{self.n_agents}.pth"), map_location=self.device))
            self.model[i].eval()


def main():    
    parser = argparse.ArgumentParser(description="Test MAPPO on multi-agent particle environments.")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--n_agents", type=int, required=True, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to load/save models and results")
    parser.add_argument("--render", action="store_true", help="Render environment and save gifs")
    args = parser.parse_args()

    ENV_NAME = args.env_name
    N_AGENTS = args.n_agents
    EPISODES = args.episodes
    if args.save_dir is not None:
        SAVE_DIR = args.save_dir
    else:
        if ENV_NAME.split("_")[-1] == "reference":
            SAVE_DIR = "results/mappo_direct" + "_" + str(N_AGENTS) + "agents_" + str(EPISODES)
        else:
            SAVE_DIR = "results/mappo_" + (ENV_NAME.split("_")[-1]) + "_" + str(N_AGENTS) + "agents_" + str(EPISODES)
        
    env = make_env(ENV_NAME, n_agents=N_AGENTS)
    env.reset()
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(env, N_AGENTS, device)
    agent.load_models(SAVE_DIR)
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
            
            if args.render:
                frame = np.array(env.render(mode='rgb_array'))
                frames.append(frame[0])

            total_reward += rewards
            next_obs = np.array(next_obs)
            dones = np.array(dones)
            done = dones.all()
            if step > STEPS_PER_EPISODE:
                done = True
            obs = next_obs
            step += 1

        agents_total_reward += total_reward.sum()

    print("Total reward: ", agents_total_reward / (10 * N_AGENTS))
    if args.render:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        imageio.mimsave(os.path.join(SAVE_DIR, f"test_{N_AGENTS}_agent_broadcast.gif"), frames, duration=0.01)

    
if __name__=="__main__":
    main()