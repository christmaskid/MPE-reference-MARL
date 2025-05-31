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
os.environ['SUPPRESS_MA_PROMPT']='1'

# ==== Hyperparameters ====
ENV_NAME = "multiple_reference_broadcast"
N_AGENTS = 3
EPISODES = 3000
SAVE_DIR = "maddpg_"+(ENV_NAME.split("_")[-1])+"_"+str(N_AGENTS)+"agents_"+str(EPISODES)

LR_ACTOR = 1e-5
LR_CRITIC = 1e-5
LR_ALPHA = 1e-5
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 1024
BUFFER_SIZE = int(3e4)
STEPS_PER_EPISODE = 30
WARMUP_EP = 100
COMM_DIM = 10
NOISE_SCALE = 0.1

# ==== Actor ====
class Actor(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential( 
                nn.Linear(obs_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, act_dim))
        
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_mean = (self.min_action + self.max_action) / 2.0
        self.action_std = (self.max_action - self.min_action) / 2.0

    def forward(self, x):
        return self.fc(x)
    
    def sample(self, obs, noise_scale=NOISE_SCALE):
        action = self(obs)
        action = torch.clamp(action + noise_scale * torch.randn_like(action), self.min_action, self.max_action)
        action = (action - self.action_mean) / self.action_std
        return action

# ==== Critic ====
class Critic(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, 1))

    # s(n_agents, B, obs+comm), a(n_agents, B, act)
    def forward(self, s, a, idx):
        B = s.shape[1]
        s = s[:,:,:-COMM_DIM]
        a = a[:,:,:-COMM_DIM]
        # index 0 is the one it critics
        if idx != 0:
            s_c = s.clone()
            a_c = a.clone()
            s_c[0] = s[idx]
            s_c[idx] = s[0]
            a_c[0] = a[idx]
            a_c[idx] = a[0]
            s=s_c
            a=a_c
        # (B, n_agents * obs) <- (n_agents, B, obs)
        s = s.transpose(0, 1).reshape(B, self.obs_dim)
        # (B, n_agents * act) <- (n_agents, B, act)
        a = a.transpose(0, 1).reshape(B, self.act_dim)
        # (B, n_agents*(obs+act)) <- (B, n_agents*obs), (B, n_agents*act)
        x = torch.cat((s,a), -1)
        return self.fc(x)


# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, cap, n_agents, obs_dim, act_dim):
        self.cap = cap
        self.size = 0
        self.state = np.empty((cap, n_agents, obs_dim), dtype=np.float32)
        self.action = np.empty((cap, n_agents, act_dim), dtype=np.float32)
        self.reward = np.empty((cap, n_agents, 1), dtype=np.float32)
        self.next_state = np.empty((cap, n_agents, obs_dim), dtype=np.float32)
        self.done = np.empty((cap, 1), dtype=np.float32)
        self.pos = 0

    def add(self, state, next_state, action, reward, done):
        self.state[self.pos] = state
        self.next_state[self.pos] = next_state
        self.action[self.pos] = action
        self.reward[self.pos] = reward[:, None]
        self.done[self.pos, 0] = done
        self.last_pos = self.pos


        self.pos = (self.pos+1)%self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = np.random.choice(self.size, n, replace=False)
        states = self.state[idx]
        next_states = self.next_state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        dones = self.done[idx]
        return states, next_states, actions, rewards, dones


# ==== MADDPG Agent ====
class MADDPG:
    def __init__(self, obs_dim, act_dim, comm_dim, n_agents, device):
        self.n_agents = n_agents
        self.device = device

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.comm_dim = comm_dim
        self.buffer = ReplayBuffer(cap=BUFFER_SIZE, n_agents=n_agents, obs_dim=self.obs_dim, act_dim=self.act_dim)

        self.actor = Actor(obs_dim=obs_dim,
                             hid_dim=512,
                             act_dim=act_dim).to(device)
        self.target_actor = Actor(obs_dim=obs_dim,
                             hid_dim=512,
                             act_dim=act_dim).to(device)

        self.critic = Critic(obs_dim=(obs_dim-COMM_DIM) * n_agents,
                                hid_dim=512,
                                act_dim=(act_dim-COMM_DIM) * n_agents).to(device)

        self.target_critic = Critic(obs_dim=(obs_dim-COMM_DIM) * n_agents,
                                       hid_dim=512,
                                       act_dim=(act_dim-COMM_DIM) * n_agents).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.clip_norm = 1


    def select_action(self, obs_all, noise_scale=0.1):
        env_actions = []
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_all[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actor.sample(obs, noise_scale=noise_scale).detach().cpu().numpy().squeeze()
            u, c = action[:-COMM_DIM], action[-COMM_DIM:]
            action_list = []
            if len(u)>0:
                action_list.append(u)
            if len(c)>0:
                action_list.append(c)
            env_actions.append(action_list)
            
            actions.append(action)
        
        return env_actions, actions

    def train(self):
        if self.buffer.size < BATCH_SIZE:
            return 0.0, 0.0

        obs, next_obs, actions, rewards, dones = self.buffer.sample(BATCH_SIZE)

        dones = torch.tensor(dones.astype(np.float32), device=self.device)

        # (n_agents, B, obs) <- ...
        obs_all = torch.tensor(obs.swapaxes(0,1), device=self.device)
        actions_all = torch.tensor(actions.swapaxes(0,1), device=self.device)
        rewards_all = torch.tensor(rewards.swapaxes(0,1), device=self.device)
        next_obs_all = torch.tensor(next_obs.swapaxes(0,1), device=self.device)

        total_actor_loss = 0
        total_critic_loss = 0

        next_actions_all = []

        for i in range(self.n_agents):
            next_obs_i = next_obs_all[i]
            with torch.no_grad():
                next_actions = self.target_actor.sample(next_obs_i)
                next_actions_all.append(next_actions)

        # (n_agents, B, act)
        next_actions_all = torch.stack(next_actions_all, dim=0)

        for i in range(self.n_agents):
            next_obs_i = next_obs_all[i]
            rewards_i = rewards_all[i]


            with torch.no_grad():
                target_Q = self.target_critic(next_obs_all, next_actions_all, i)
                y = rewards_i + GAMMA * target_Q * (1 - dones)

            current_Q = self.critic(obs_all, actions_all, i)

            delta = y - current_Q
            loss_q = (delta**2).mean()
            self.critic_optim.zero_grad()
            loss_q.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
            self.critic_optim.step()
            total_critic_loss += loss_q.item()

            sample_actions_all = []
            for j in range(self.n_agents):
                sample_actions = self.actor.sample(obs_all[j])
                if i != j:
                    sample_actions = sample_actions.detach()
                sample_actions_all.append(sample_actions)
            sample_actions_all = torch.stack(sample_actions_all, dim=0)

            comm_sample_actions_all = []
            comm_next_obs_all = []
            for j in range(self.n_agents):
                comm_next_obs_all.append(torch.cat((next_obs_all[j,:,:-self.comm_dim], sample_actions_all[i,:,-self.comm_dim:]), dim=-1))
            comm_next_obs_all = torch.stack(comm_next_obs_all, dim=0)

            for j in range(self.n_agents):
                sample_actions = self.actor.sample(comm_next_obs_all[j])
                comm_sample_actions_all.append(sample_actions)

            comm_sample_actions_all = torch.stack(comm_sample_actions_all, dim=0)
            q_com = torch.zeros_like(y)
            for j in range(self.n_agents):
                q_com += self.critic(comm_next_obs_all, comm_sample_actions_all, j)

            self.actor_optim.zero_grad()
            loss_pi = (-q_com / self.n_agents).mean()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)

            self.actor_optim.step()
            total_actor_loss += loss_pi.item()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents

    def save_models(self, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        (actor, critic) = (self.actor, self.critic)
        actor_path = os.path.join(save_dir, f"agent_actor.pth")
        critic_path = os.path.join(save_dir, f"agent_critic.pth")

        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)

    def load_models(self, save_dir='checkpoints'):
        actor_path = os.path.join(save_dir, f"agent_actor.pth")
        critic_path = os.path.join(save_dir, f"agent_critic.pth")
        
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"Loaded actor model from {actor_path}")
        else:
            print(f"Actor model not found at {actor_path}, using random initialization.")
        if os.path.exists(critic_path): 
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print(f"Loaded critic model from {critic_path}")
        else:
            print(f"Critic model not found at {critic_path}, using random initialization.")

# ==== Training Loop ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS)
    env.reset()
    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    agent = MADDPG(obs_dims, act_dims, COMM_DIM, N_AGENTS, device)
    ep_actor_losses = []
    ep_critic_losses = []

    returns = []
    pbar = tqdm(range(EPISODES))
    for episode in pbar:
        obs = env.reset()
        obs = np.array(obs)
        actor_losses = []
        critic_losses = []

        total_reward = np.zeros(len(env.agents))
        _, last_reward, _, _ = env.step([[np.zeros(2), np.zeros(10)] for _ in range(N_AGENTS)])
        last_reward = np.array(last_reward)
        done = False
        step = 0
        while not done:
            env_actions, actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(env_actions)
            total_reward += rewards
            next_obs = np.array(next_obs)
            rewards = np.array(rewards)
            if step > STEPS_PER_EPISODE:
                done = True
            agent.buffer.add(obs, next_obs, actions, rewards-last_reward, done)
            last_reward = rewards
            if episode > WARMUP_EP:
                actor_loss, critic_loss = agent.train()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            obs = next_obs
            step += 1
        if episode > WARMUP_EP:
            returns.append(np.sum(total_reward))
            ep_actor_losses.append(np.mean(actor_losses))
            ep_critic_losses.append(np.mean(critic_losses))
        pbar.set_description(f'reward:{sum(total_reward):.5f}')
        if episode % 10 == 0:
            print(f"Episode {episode}: return {np.mean(returns[-100:]):.2f}, actor loss {np.mean(ep_actor_losses[-100:]):.4f}, critic loss {np.mean(ep_critic_losses[-100:]):.4f}", flush=True)
            draw_result(returns, ep_actor_losses, ep_critic_losses)

        if episode % 100 == 0:  # adjust frequency
            agent.save_models(save_dir=SAVE_DIR)

def draw_result(returns, actor_losses, critic_losses):
    episodes = range(1, len(returns) + 1)

    plt.figure(figsize=(15, 4))

    # Reward
    plt.subplot(1, 3, 1)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(episodes, returns, label='reward')
    plt.title("Average Return")
    plt.grid(True)

    # Actor loss
    plt.subplot(1, 3, 2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(episodes, actor_losses, label='Actor Loss', color='orange')
    plt.title("Actor Loss")
    plt.grid(True)

    # Critic loss
    plt.subplot(1, 3, 3)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(episodes, critic_losses, label='Critic Loss', color='green')
    plt.title("Critic Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(SAVE_DIR+"/training_curves.png")  # or plt.show() if you prefer
    plt.close()

if __name__ == '__main__':
    main()
