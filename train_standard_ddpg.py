# MADDPG for PettingZoo simple_reference_v3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_reference_v3
import supersuit as ss
from collections import deque
import random
import os
import matplotlib.pyplot as plt

# ==== Hyperparameters ====
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.01
BATCH_SIZE = 1024
BUFFER_SIZE = int(1e6)
EPISODES = 3000
STEPS_PER_EPISODE = 25

# ==== Actor ====
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)

# ==== Critic ====
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self):
        samples = random.sample(self.buffer, BATCH_SIZE)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

# ==== MADDPG Agent ====
class MADDPG:
    def __init__(self, env, device):
        self.env = env
        self.n_agents = env.num_agents
        self.device = device

        self.obs_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
        self.act_dims = [env.action_space(agent).shape[0] for agent in env.agents]

        self.actors = [Actor(self.obs_dims[i], self.act_dims[i]).to(device) for i in range(self.n_agents)]
        self.critics = [Critic(sum(self.obs_dims), sum(self.act_dims)).to(device) for _ in range(self.n_agents)]

        self.target_actors = [Actor(self.obs_dims[i], self.act_dims[i]).to(device) for i in range(self.n_agents)]
        self.target_critics = [Critic(sum(self.obs_dims), sum(self.act_dims)).to(device) for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optim = [optim.Adam(self.actors[i].parameters(), lr=LR_ACTOR) for i in range(self.n_agents)]
        self.critic_optim = [optim.Adam(self.critics[i].parameters(), lr=LR_CRITIC) for i in range(self.n_agents)]

        self.buffer = ReplayBuffer()

    def select_action(self, obs_all, noise_scale=0.1):
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_all[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actors[i](obs).detach().cpu().numpy().squeeze()
            action += noise_scale * np.random.randn(*action.shape)
            # actions.append(np.clip(action, -1, 1))
            action = np.tanh(action)
            action = (action + 1) / 2.0 
            actions.append(action)
        return actions

    def train(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0, 0.0

        obs, actions, rewards, next_obs, dones = self.buffer.sample()

        obs = [torch.tensor(np.vstack(obs[:, i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]
        actions = [torch.tensor(np.vstack(actions[:, i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.tensor(np.vstack(next_obs[:, i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]
        rewards = [torch.tensor(rewards[:, i], dtype=torch.float32).to(self.device).unsqueeze(1) for i in range(self.n_agents)]
        dones = [torch.tensor(dones[:, i], dtype=torch.float32).to(self.device).unsqueeze(1) for i in range(self.n_agents)]

        obs_all = torch.cat(obs, dim=1)
        actions_all = torch.cat(actions, dim=1)
        next_obs_all = torch.cat(next_obs, dim=1)

        total_actor_loss = 0
        total_critic_loss = 0

        with torch.no_grad():
            next_actions = [self.target_actors[i](next_obs[i]) for i in range(self.n_agents)]
            next_actions_all = torch.cat(next_actions, dim=1)

        for i in range(self.n_agents):
            # critic loss
            with torch.no_grad():
                target_Q = self.target_critics[i](next_obs_all, next_actions_all)
                y = rewards[i] + GAMMA * (1 - dones[i]) * target_Q

            current_Q = self.critics[i](obs_all, actions_all)
            critic_loss = nn.MSELoss()(current_Q, y)
            self.critic_optim[i].zero_grad()
            critic_loss.backward()
            self.critic_optim[i].step()

            # actor loss
            curr_action = [self.actors[j](obs[j]) if j == i else actions[j].detach() for j in range(self.n_agents)]
            curr_action_all = torch.cat(curr_action, dim=1)
            actor_loss = -self.critics[i](obs_all, curr_action_all).mean()
            self.actor_optim[i].zero_grad()
            actor_loss.backward()
            self.actor_optim[i].step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

            # soft update
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents
    
    def save_models(self, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(save_dir, f"agent_{i}_actor.pth")
            critic_path = os.path.join(save_dir, f"agent_{i}_critic.pth")

            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)

# ==== Training Loop ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_reference_v3.parallel_env(continuous_actions=True)
    env.reset()
    agent = MADDPG(env, device)
    ep_actor_losses = []
    ep_critic_losses = []

    returns = []

    for episode in range(EPISODES):
        obs_dict, _ = env.reset()
        obs_n = [obs_dict[agent_name] for agent_name in env.agents]
        # [self_vel_x, self_vel_y , (landmark_rel_position_x, landmark_rel_position_y) * 3, goal_id(dim=3), communication(dim=10)]
        actor_losses = []
        critic_losses = []

        total_reward = np.zeros(len(env.agents))

        for step in range(STEPS_PER_EPISODE):

            actions_n = agent.select_action(obs_n)
            actions_dict = {agent_name: action for agent_name, action in zip(env.agents, actions_n)}
            next_obs_dict, rewards_dict, terminations, truncations, _ = env.step(actions_dict)

            next_obs_n = [next_obs_dict[agent_name] for agent_name in env.agents]
            rewards_n = [rewards_dict[agent_name] for agent_name in env.agents]
            dones_n = [terminations[agent_name] or truncations[agent_name] for agent_name in env.agents]
            
            if rewards_n:
                agent.buffer.add((obs_n, actions_n, rewards_n, next_obs_n, dones_n))
                total_reward += rewards_n
            
            actor_loss, critic_loss = agent.train()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            obs_n = next_obs_n
            if all(dones_n):
                break

        returns.append(total_reward.mean())
        ep_actor_losses.append(np.mean(actor_losses))
        ep_critic_losses.append(np.mean(critic_losses))

        if episode % 50 == 0:
            print(f"Episode {episode}: return {returns[-1]:.2f}, actor loss {np.mean(actor_losses):.4f}, critic loss {np.mean(critic_losses):.4f}")
            draw_result(returns, ep_actor_losses, ep_critic_losses)

        if episode % 100 == 0:  # adjust frequency
            agent.save_models(save_dir='models/')

def draw_result(returns, actor_losses, critic_losses):
    episodes = range(1, len(returns) + 1)

    plt.figure(figsize=(15, 4))

    # Reward
    plt.subplot(1, 3, 1)
    plt.plot(episodes, returns, label='Average Return')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Average Return")
    plt.grid(True)

    # Actor loss
    plt.subplot(1, 3, 2)
    plt.plot(episodes, actor_losses, label='Actor Loss', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.grid(True)

    # Critic loss
    plt.subplot(1, 3, 3)
    plt.plot(episodes, critic_losses, label='Critic Loss', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")  # or plt.show() if you prefer
    plt.close()

if __name__ == '__main__':
    main()
