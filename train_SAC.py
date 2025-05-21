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
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
GAMMA = 0.0
TAU = 0.01
BATCH_SIZE = 1024
BUFFER_SIZE = int(1e6)
EPISODES = 100000
STEPS_PER_EPISODE = 100
N_AGENTS = 3

# ==== Actor ====
class Actor(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(obs_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                )
        self.h1 = nn.Sequential(
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, act_dim))
        self.h2 = nn.Sequential(
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, act_dim))

    def forward(self, x):
        x = self.fc(x)
        return self.h1(x), torch.clamp(self.h2(x), -20, 2).exp()


# ==== Critic ====
class Critic(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, hid_dim), nn.SiLU(),
                nn.Linear(hid_dim, 1))

    def forward(self, x):
        return self.fc(x)


class Critics(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.Q1 = Critic(obs_dim, hid_dim, act_dim)
        self.Q2 = Critic(obs_dim, hid_dim, act_dim)

    # s, a (B, n_agents, obs, act)
    def forward(self, s, a, idx):
        B = s.shape[0]
        # index 0 is the one it critics
        if idx != 0:
            s_c = s.clone()
            a_c = a.clone()
            s_c[[0, idx]] = s[[idx, 0]]
            a_c[[0, idx]] = a[[idx, 0]]
            s=s_c
            a=a_c
        s = s.reshape(B, self.obs_dim)
        a = a.reshape(B, self.act_dim)
        x = torch.cat((s,a), -1)
        return self.Q1(x), self.Q2(x)


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
        self.reward[self.pos,:, 0] = reward
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


# ==== MASAC Agent ====
class MASAC:
    def __init__(self, obs_dims, act_dims, n_agents, device):
        self.n_agents = n_agents
        self.device = device

        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.buffer = ReplayBuffer(cap=BUFFER_SIZE, n_agents=n_agents, obs_dim=self.obs_dims, act_dim=self.act_dims)

        self.actor = Actor(obs_dim=obs_dims,
                             hid_dim=512,
                             act_dim=act_dims).to(device)

        self.critic = Critics(obs_dim=obs_dims * n_agents,
                                hid_dim=512,
                                act_dim=act_dims * n_agents).to(device)

        self.target_critic = Critics(obs_dim=obs_dims * n_agents,
                                       hid_dim=512,
                                       act_dim=act_dims * n_agents).to(device)

        # for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
        #     actor_path = os.path.join('models', f"agent_{i}_actor.pth")
        #     critic_path = os.path.join('models', f"agent_{i}_critic.pth")
        #     actor.load_state_dict(torch.load(actor_path))
        #     critic.load_state_dict(torch.load(critic_path))

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.alpha = 0.01
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-5)
        self.clip_norm = 1

        self.mask = torch.ones(12, device=self.device) * 0.5
        self.mask[0] = 0
        self.mask[1] = 0
        self.mask = self.mask[None, :]

    def select_action(self, obs_all, noise_scale=0.1):
        actions_env = []
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_all[i:i + 1].astype(np.float32), device=self.device)
            mu, sigma = self.actor(obs)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z).cpu().numpy()[0]
            # The first two are (x, y) \in [-1, 1]
            # The rest are say_0~say_9 \in [0, 1]
            noise = np.random.normal(size=action.shape) * noise_scale
            action = np.clip(action + noise, -1, 1)
            action[2:] = 0.5 * (action[2:] + 1)
            actions_env.append([action[:2], action[2:]])
            actions.append(action)
        return actions_env, np.array(actions)

    def train(self):
        if self.buffer.size < BATCH_SIZE:
            return 0.0, 0.0

        obs, next_obs, actions, rewards, dones = self.buffer.sample(BATCH_SIZE)

        dones = torch.tensor(dones.astype(np.float32), device=self.device)

        obs_all = torch.tensor(obs, device=self.device)
        actions_all = torch.tensor(actions, device=self.device)
        rewards_all = torch.tensor(rewards, device=self.device)
        next_obs_all = torch.tensor(next_obs, device=self.device)

        total_actor_loss = 0
        total_critic_loss = 0

        next_actions_all = []
        next_log_p_all = []

        for i in range(self.n_agents):
            next_obs_i = next_obs_all[:, i]
            with torch.no_grad():
                mu, sigma = self.actor(next_obs_i)
                dist = Normal(mu, sigma)
                z = dist.sample()
                next_actions = torch.tanh(z)
                next_log_p = (dist.log_prob(z) - torch.log(1 - next_actions.pow(2) + 1e-6)).sum(-1, keepdim=True)
                next_actions_all.append(next_actions)
                next_log_p_all.append(next_log_p)
        next_actions_all = torch.cat(next_actions_all, dim=1)
        next_log_p_all = torch.cat(next_log_p_all, dim=1)

        for i in range(self.n_agents):
            next_obs_i = next_obs_all[:, i]
            rewards_i = rewards_all[:, i]


            with torch.no_grad():
                q1, q2 = self.target_critic(next_obs_all, next_actions_all, i)
                next_q = torch.min(q1, q2)
                next_q = next_q - self.alpha * next_log_p_all[:, i:i+1]
                target_q = rewards_i + GAMMA * next_q * (1 - dones)

            pred_q1, pred_q2 = self.critic(obs_all, actions_all, i)
            delta1 = pred_q1 - target_q
            delta2 = pred_q2 - target_q
            loss_q1 = (delta1**2).mean()
            loss_q2 = (delta2**2).mean()
            loss_q = 0.5 * (loss_q1 + loss_q2)
            self.critic_optim.zero_grad()
            loss_q.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
            self.critic_optim.step()
            total_critic_loss += loss_q.item()

            sample_actions_all = []
            for j in range(self.n_agents):
                mu, sigma = self.actor(obs_all[:,j])
                dist = Normal(mu, sigma)
                z = dist.rsample()
                sample_actions = torch.tanh(z)
                log_p = (dist.log_prob(z) - torch.log(1 - sample_actions.pow(2) + 1e-6)).sum(-1, keepdim=True)
                sample_actions = sample_actions + self.mask * (1 - sample_actions)
                if i != j:
                    sample_actions = sample_actions.detach()
                    log_p = log_p.detach()
                else:
                    log_p_i = log_p
                sample_actions_all.append(sample_actions)
            sample_actions_all = torch.cat(sample_actions_all, dim=1)
            q1, q2 = self.critic(obs_all, sample_actions_all, i)
            q = torch.min(q1, q2)
            loss_pi = (self.alpha * log_p_i - q).mean()
            self.actor_optim.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)
            self.actor_optim.step()
            total_actor_loss += loss_pi.item()

            self.alpha_optim.zero_grad()
            loss_alpha = self.log_alpha * (-log_p_i.mean().item() + self.act_dims)
            loss_alpha.backward()
            self.alpha_optim.step()
            self.alpha = np.exp(self.log_alpha.item())


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

# ==== Training Loop ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env("multiple_reference", n_agents=N_AGENTS)
    env.reset()
    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    agent = MASAC(obs_dims, act_dims, N_AGENTS, device)
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
        done = False
        step = 0
        while not done:
            actions_env, actions = agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(actions_env)
            total_reward += rewards
            next_obs = np.array(next_obs)
            rewards = 1 + np.clip(np.array(rewards), -10, 0) / 10
            dones = np.array(dones)
            done = dones.all()
            if step > STEPS_PER_EPISODE:
                done = True
            agent.buffer.add(obs, next_obs, actions, rewards, done)
            actor_loss, critic_loss = agent.train()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            obs = next_obs
            step += 1

        returns.append(-np.sum(total_reward))
        ep_actor_losses.append(np.mean(actor_losses))
        ep_critic_losses.append(np.mean(critic_losses))
        pbar.set_description(f'reward:{sum(total_reward):.5f}')
        if episode % 10 == 0:
            tqdm.write(f"Episode {episode}: return {np.mean(returns[-100:]):.2f}, actor loss {np.mean(ep_actor_losses[-100:]):.4f}, critic loss {np.mean(ep_critic_losses[-100:]):.4f}")
            draw_result(returns, ep_actor_losses, ep_critic_losses)

        if episode % 100 == 0:  # adjust frequency
            agent.save_models(save_dir='models/')

def draw_result(returns, actor_losses, critic_losses):
    episodes = range(1, len(returns) + 1)

    plt.figure(figsize=(15, 4))

    # Reward
    plt.subplot(1, 3, 1)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.yscale("log")
    plt.plot(episodes, returns, label='-log(reward)')
    plt.title("Average Return")
    plt.grid(True)

    # Actor loss
    plt.subplot(1, 3, 2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.yscale("linear")
    plt.plot(episodes, actor_losses, label='Actor Loss', color='orange')
    plt.title("Actor Loss")
    plt.grid(True)

    # Critic loss
    plt.subplot(1, 3, 3)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.yscale("linear")
    plt.plot(episodes, critic_losses, label='Critic Loss', color='green')
    plt.title("Critic Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")  # or plt.show() if you prefer
    plt.close()

if __name__ == '__main__':
    main()
