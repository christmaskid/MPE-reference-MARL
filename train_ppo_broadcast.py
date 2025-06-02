import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
# from pettingzoo.mpe import simple_reference_v3
import os
import csv
import matplotlib.pyplot as plt
# from gymnasium.spaces import Box

import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env

os.environ['SUPPRESS_MA_PROMPT']='1'

N_AGENTS = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'done', 'log_prob', 'value'])

class PPONet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512), 
            nn.Tanh(),
            nn.Linear(512, 512), 
            nn.Tanh(),
            nn.Linear(512, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim*N_AGENTS, 512),
            nn.Tanh(),
            nn.Linear(512, 512), 
            nn.Tanh(),
            nn.Linear(512, 512), 
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        # self.actor = nn.Sequential(
        #     nn.Linear(obs_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 128), 
        #     nn.Tanh(),
        #     nn.Linear(128, act_dim)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(obs_dim*N_AGENTS, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 128), 
        #     nn.Tanh(),
        #     nn.Linear(128, 1)
        # )

        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.epsilon = 1e-6

    def forward(self, state):
        # action distribution
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(mu, std)
        
        return distribution

    def get_act(self, state):
        distribution = self.forward(state)
        
        raw_action = distribution.rsample()

        # Use Sigmoid directory map to (0, 1)
        # action = torch.sigmoid(raw_action)
        # log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        # log_prob -= torch.log(action * (1 - action) + self.epsilon).sum(dim=-1)  # log_det_jacobian
        
        # Use tabh map to (-1, 1), then rescale to (0, 1)
        action = torch.tanh(raw_action)
        # action = (tanh_action + 1) / 2  # map to (0, 1)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon).sum(dim=-1)

        return action[:2], action[2:], action, log_prob
    
    # all_obs (batch ,n_agnet, obs_shape)
    def evaluate(self, all_obs, action, agent):
        b = all_obs.shape[0]
        input = torch.reshape(all_obs, shape=(b, -1)) # input shape (batch, n_agnet*obs_shape)
        value = self.critic(input).squeeze(-1)
        distribution = self.forward(all_obs[:, agent])

        # raw_action = torch.log(action + self.epsilon) - torch.log(1 - action + self.epsilon)
        # log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        # log_prob -= torch.log(action * (1 - action) + self.epsilon).sum(dim=-1)
        
        # x = action * 2 -1
        raw_action = 0.5 * torch.log((1 + action + self.epsilon) / (1 - action + self.epsilon))
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon).sum(dim=-1)

        entropy = distribution.entropy().sum(dim=-1)
        
        return value, log_prob, entropy

class PPOAgent:
    def __init__(self, env, device):
        self.n_agents = N_AGENTS
        self.device = device

        self.obs_dims = env.observation_space[0].shape[0]
        self.act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])

        print(f"Observation space dimension: {self.obs_dims}")
        print(f"Action space dimension: {self.act_dims}")
        # action all in range (-1, 1)
        
        # self.model = [PPONet(self.obs_dims, self.act_dims).to(self.device) for i in range(self.n_agents)]
        self.model = PPONet(self.obs_dims, self.act_dims).to(self.device)
        
        self.ppo_epoches = 10
        self.buffer = []
        self.batch_size = 64

        self.gamma = 0.99
        self.lam = 0.95
        ep = 0.2
        self.min_ratio = 1-ep
        self.max_ratio = 1+ep

        # self.optimizer = [optim.Adam(self.model[i].parameters(), lr = 0.00025) for i in range(self.n_agents)]
        # self.optimizer = optim.Adam(self.model.parameters(), lr = 7e-4)

        actor_params = list(self.model.actor.parameters()) + [self.model.log_std]
        critic_params = self.model.critic.parameters()

        self.actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = optim.Adam(critic_params, lr=1e-3)

        self.c1 = 0.5
        self.c2 = 0.01

    def collect(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append(Transition(state, action, reward, next_state, done, log_prob, value))

    def compute_gae(self, rews, values, dones, next_value):
        rewards = []
        gae = 0
        values = list(values)
        values = values + [next_value]
        
        for t in reversed(range(len(rews))):
            delta = rews[t] + self.gamma * values[t+1] * (1 - dones[t].float()) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t].float()) * gae
            rewards.insert(0, gae + values[t])
        return rewards

    def get_experience(self):
        exp = Transition(*zip(*self.buffer))
        obs, actions, rewards, next_obs, dones, log_probs, values = exp

        obs = [torch.stack([o[i] for o in obs]) for i in range(self.n_agents)]
        actions = [torch.stack([a[i] for a in actions]) for i in range(self.n_agents)]
        rewards = [torch.stack([r[i] for r in rewards]) for i in range(self.n_agents)]
        next_obs = [torch.stack([o[i] for o in next_obs]) for i in range(self.n_agents)]
        dones = [torch.stack([d[i] for d in dones]) for i in range(self.n_agents)]
        old_log_probs = [torch.stack([p[i] for p in log_probs]) for i in range(self.n_agents)]
        values = [torch.stack([v[i] for v in values]) for i in range(self.n_agents)]

        returns = [torch.tensor(self.compute_gae(rews=rewards[i], values = values[i], dones = dones[i], next_value=0), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]

        advantages = [returns[i] - values[i] for i in range(self.n_agents)]
        advantages = [(adv - adv.mean()) / (adv.std() + 1e-8) for adv in advantages]

        return obs, actions, next_obs, old_log_probs, advantages, returns

    def update(self):
        obs, actions, next_obs, old_log_probs, advantages, returns = self.get_experience()

        actor_losses = [[] for _ in range(self.n_agents)]
        critic_losses = [[] for _ in range(self.n_agents)]
        entropy_losses = [[] for _ in range(self.n_agents)]
        total_losses = [[] for _ in range(self.n_agents)]

        all_obs = torch.stack([obs[agent] for agent in range(self.n_agents)], dim = 1)

        for _ in range(self.ppo_epoches):

            # for i in range(0, len(obs[0]), self.batch_size):
            #     idx = range(i, min(i + self.batch_size, len(obs[0])))

            #     all_obs_batch = []
            #     for agent in range(self.n_agents):
            #         all_obs_batch.append(obs[agent][idx])
            #     all_obs_batch = torch.stack(all_obs_batch, dim=1)

            for agent in range(self.n_agents):

                # V_phi(s) , logP_theta(s, a), H_theta(s)
                value, log_prob, entropy = self.model.evaluate(all_obs, actions[agent], agent)

                policy_ratio = torch.exp(log_prob - old_log_probs[agent])

                surr1 = policy_ratio * advantages[agent]
                surr2 = torch.clamp(policy_ratio, self.min_ratio, self.max_ratio) * advantages[agent]

                clipped_surrogate_objective = torch.min(surr1, surr2).mean()

                actor_loss = -clipped_surrogate_objective

                critic_loss = F.mse_loss(value, returns[agent])

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy.mean()
                actor_losses[agent].append(actor_loss.item())
                critic_losses[agent].append(self.c1 * critic_loss.item())
                entropy_losses[agent].append(-self.c2 * entropy.mean().item())
                total_losses[agent].append(loss.item())

                # self.optimizer.zero_grad()
                # loss.backward()
                # # torch.nn.utils.clip_grad_norm_(self.model[agent].parameters(), max_norm=0.5)
                # self.optimizer.step()

                actor_all_loss = actor_loss - self.c2 * entropy.mean()
                self.actor_optimizer.zero_grad()
                actor_all_loss.backward()
                self.actor_optimizer.step()

                critic_all_loss = self.c1 * critic_loss
                self.critic_optimizer.zero_grad()
                critic_all_loss.backward()
                self.critic_optimizer.step()

        mean_actor_losses = [sum(agent_loss) / len(agent_loss) for agent_loss in actor_losses]
        mean_critic_losses = [sum(agent_loss) / len(agent_loss) for agent_loss in critic_losses]
        mean_entropy_losses = [sum(agent_loss) / len(agent_loss) for agent_loss in entropy_losses]
        mean_total_losses = [sum(agent_loss) / len(agent_loss) for agent_loss in total_losses]

        return mean_total_losses, mean_actor_losses, mean_critic_losses, mean_entropy_losses


def train(env, epoches):
    env.reset()
    agent = PPOAgent(env, device)

    returns = []
    total_losses = []
    actor_losses = []
    critic_losses = []
    entropy_losses = []

    log_path = "training_log.csv"
    log_header = [
        "epoch", "avg_reward",
        "total_loss", "actor_loss", "critic_loss", "entropy_loss"
    ]

    # Create the log file and write header if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
    
    max_step_per_game = 25
    max_steps = 32 * max_step_per_game
    max_returns = np.array([-np.inf, -np.inf])
    check_peroid = 10

    for epoch in tqdm(range(epoches)):
        obs_n = env.reset()

        agent.buffer = [] # empty the buffer

        episode_rewards = []
        total_reward = np.zeros(N_AGENTS)
        prev_rewards_n = [torch.tensor(0)] * N_AGENTS

        step = 0
        for _ in range(max_steps):
            with torch.no_grad():
                # self_pose(2), self_velocity(2), other's landmark_pose(2), comm(10)
                obs_n = [torch.tensor(obs_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
         
                outputs = [agent.model.get_act(obs_n[i]) for i in range(N_AGENTS)]
                move, comm, actions_n, log_prob_n = zip(*outputs)

                obs_input = torch.stack(obs_n, dim=0).unsqueeze(0)
                value_n = [torch.tensor(agent.model.evaluate(obs_input, actions_n[i], i)[0].item()).to(device) for i in range(N_AGENTS)]

                action_input = [[move[i].cpu().numpy(), comm[i].cpu().numpy()] for i in range(N_AGENTS)]

            next_obs_n, rewards_n, dones_n, _  = env.step(action_input)
            # rewards_n = [r / 10 for r in rewards_n]

            if step == max_step_per_game -1:
                dones_n = [True for i in range(N_AGENTS)]

            next_obs_n = [torch.tensor(next_obs_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
            rewards_n = [torch.tensor(rewards_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
            dones_n = [torch.tensor(dones_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]

            incremental_rewards = [rewards_n[i]-prev_rewards_n[i] for i in range(N_AGENTS)]
            prev_rewards_n = rewards_n

            agent.collect(obs_n, actions_n, incremental_rewards, next_obs_n, dones_n, log_prob_n, value_n)
            for i in range(N_AGENTS):
                total_reward[i] += rewards_n[i].cpu().numpy()
            
            obs_n = next_obs_n
            step += 1

            if all(dones_n):
                obs_n = env.reset()
                
                episode_rewards.append(total_reward)
                total_reward = np.zeros(N_AGENTS)
                step = 0
                prev_rewards_n =  [torch.tensor(0)] * N_AGENTS

        if not all(dones_n):
            episode_rewards.append(total_reward)

        mean_total_losses, mean_actor_losses, mean_critic_losses, mean_entropy_losses = agent.update()
        
        if epoch % check_peroid == 0:
            avg_rew = np.mean(episode_rewards, axis = 0)
            # tqdm.write(f"Epoch {epoch}, Average Reward: {avg_rew[0]:.2f}, {avg_rew[1]:.2f}, Loss: {total_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")
            
            tqdm.write(f"Epoch {epoch}:")
            for i in range(len(mean_total_losses)):
                tqdm.write(
                    f"Agent {i} -- Total: {mean_total_losses[i]:.4f}, "
                    f"Average Reward: {avg_rew[0]:.2f}, {avg_rew[1]:.2f}, "
                    f"Actor: {mean_actor_losses[i]:.4f}, "
                    f"Critic: {mean_critic_losses[i]:.4f}, "
                    f"Entropy: {mean_entropy_losses[i]:.4f}"
                )
            
            # save model
            eval_returns = eval(env, agent, max_step_per_game)
            tqdm.write(f"{eval_returns}")
            # if eval_returns[0] > max_returns[0] and eval_returns[1] > max_returns[1]:
            if sum(eval_returns) > sum(max_returns):
                tqdm.write(f"Save model with evaluation returns: {eval_returns}")
                max_returns = eval_returns
                # for i in range(agent.n_agents):
                torch.save(agent.model.state_dict(), f"PPO_agent_broadcast.pth")

            # average reward for two agents
            returns.append(avg_rew)

            # average loss for all agents
            total_losses.append(mean_total_losses)
            actor_losses.append(mean_actor_losses)
            critic_losses.append(mean_critic_losses)
            entropy_losses.append(mean_entropy_losses)

            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    avg_rew.mean(),
                    np.mean(mean_total_losses),
                    np.mean(mean_actor_losses),
                    np.mean(mean_critic_losses),
                    np.mean(mean_entropy_losses)
                ])

            plot(returns, actor_losses, critic_losses, entropy_losses, check_peroid)

def eval(env, agent, max_step_per_game):
    returns = []

    for epoch in range(10):
        obs_n= env.reset()
        total_reward = np.zeros(N_AGENTS)
        dones_n = [torch.tensor(False, dtype=torch.float32).to(device) for i in range(N_AGENTS)]

        for _ in range(max_step_per_game):
            with torch.no_grad():
                obs_n = [torch.tensor(obs_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
         
                outputs = [agent.model.get_act(obs_n[i]) for i in range(N_AGENTS)]
                move, comm, _, _ = zip(*outputs)

                action_input = [[move[i].cpu().numpy(), comm[i].cpu().numpy()] for i in range(N_AGENTS)]

            next_obs_n, rewards_n, dones_n, _  = env.step(action_input)
            # rewards_n = np.clip(rewards_n, -10, 0)

            next_obs_n = [torch.tensor(next_obs_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
            rewards_n = [torch.tensor(rewards_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]
            dones_n = [torch.tensor(dones_n[i], dtype=torch.float32).to(device) for i in range(N_AGENTS)]

            for i in range(N_AGENTS):
                total_reward[i] += rewards_n[i].cpu().numpy()

            if all(dones_n):
                break

            obs_n = next_obs_n

        returns.append(total_reward)

    return np.mean(np.array(returns), axis = 0)


def plot(returns, actor_losses, critic_losses, entropy_losses, check_peroid):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 column

    returns_plot = np.array(returns)
    num_agents = returns_plot.shape[1]

    # Subplot 1: Average Reward
    for agent_idx in range(num_agents):
        x_vals = [i * check_peroid for i in range(len(returns))]
        y_vals = [r[agent_idx] for r in returns]
        axes[0, 0].plot(x_vals, y_vals, label=f'Agent {agent_idx}')
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()
    axes[0, 0].set_title("Average Return")
    
    # Convert to numpy for easier indexing
    actor_losses_np = np.array(actor_losses)        # shape: (epochs, n_agents)
    critic_losses_np = np.array(critic_losses)
    entropy_losses_np = np.array(entropy_losses)

    x_vals_loss = [i * check_peroid for i in range(len(actor_losses))]

    # Plot actor losses
    for agent_idx in range(num_agents):
        axes[0, 1].plot(x_vals_loss, actor_losses_np[:, agent_idx], label=f'Agent {agent_idx}')
    axes[0, 1].set_title("Actor Loss per Agent")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    # Plot critic losses
    for agent_idx in range(num_agents):
        axes[1, 0].plot(x_vals_loss, critic_losses_np[:, agent_idx], label=f'Agent {agent_idx}')
    axes[1, 0].set_title("Critic Loss per Agent")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Plot entropy losses
    for agent_idx in range(num_agents):
        axes[1, 1].plot(x_vals_loss, entropy_losses_np[:, agent_idx], label=f'Agent {agent_idx}')
    axes[1, 1].set_title("Entropy Loss per Agent")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("training_summary.png")
    plt.close()


def main():
    # env = simple_reference_v3.parallel_env(continuous_actions=True)
    env = make_env("multiple_reference_broadcast", n_agents=N_AGENTS)
    epoches = 3000
    train(env, epoches)

if __name__ == "__main__":
    main()