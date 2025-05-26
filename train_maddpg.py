# MADDPG for our simple reference env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from buffer import ReplayBuffer

os.environ['SUPPRESS_MA_PROMPT']='1'
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env

ENV_NAME = "multiple_reference"
N_AGENTS = 3
DIM_C = 10
SHARED_REWARD = 1
act_u_dim = 2

SAVE_DIR = f"maddpg_{ENV_NAME}_{N_AGENTS}_{DIM_C}_{SHARED_REWARD}_{act_u_dim}" #'models/'

# ==== Hyperparameters ====
HIDDEN_DIM = 128 #32
LR_ACTOR = 3e-4 # 1e-3 #
LR_CRITIC = 3e-4 # 1e-3 #
GAMMA = 0.95
TAU = 0.005
BATCH_SIZE = 256
WARMUP_SIZE = 1024
BUFFER_SIZE = int(1e6)
EPISODES = 30000
STEPS_PER_EPISODE = 100 #00
CLIP_NORM = 1
NOISE_SCALE = 0.1


# ==== Actor ====
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)

# ==== Critic ====
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)

# ==== MADDPG Agent ====
class Agent:
    def __init__(self, obs_dim, act_dim, agent_id, n_agents, device='cpu'):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim * self.n_agents, act_dim * self.n_agents).to(device)
        self.target_actor = Actor(obs_dim, act_dim).to(device)
        self.target_critic = Critic(obs_dim * self.n_agents, act_dim * self.n_agents).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save(self, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_dir, f'maddpg_agent{self.agent_id}.pth'))

    def load(self, save_dir='checkpoints'):
        state_dict = torch.load(os.path.join(save_dir, f'maddpg_agent{self.agent_id}.pth'), map_location=self.device)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.target_actor.load_state_dict(state_dict['target_actor'])
        self.target_critic.load_state_dict(state_dict['target_critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])

    def learn_actor(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), CLIP_NORM)
        self.actor_optimizer.step()
        
    def learn_critic(self, critic_loss):
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), CLIP_NORM)
        self.critic_optimizer.step()

    def update_targets(self):
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, from_network, to_network):
        for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
            to_p.data.copy_(TAU * from_p.data + (1.0 - TAU) * to_p.data)


class MADDPG:
    def __init__(self, obs_dim, act_dim, act_u_dim, n_agents, device):
        self.n_agents = n_agents
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_u_dim = act_u_dim

        self.clip_norm = True

        self.agents = {
            agent_id: Agent(obs_dim, act_dim, agent_id=agent_id, 
                                n_agents=self.n_agents, device=self.device)
            for agent_id in range(n_agents)
        }
        
        # Create a separate buffer for each agent
        self.buffer = {
            i: ReplayBuffer(cap=BUFFER_SIZE, obs_dim=self.obs_dim, act_dim=self.act_dim)
            for i in range(self.n_agents)
        }

    def select_action(self, obs_all, noise_scale=NOISE_SCALE):
        env_actions = [] # for env to step (u, c)
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_all[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.agents[i].actor(obs).detach().cpu().numpy().squeeze()
            action += noise_scale * np.random.randn(*action.shape)
            action = np.tanh(action)
            action = (action + 0) / 1.0

            u, c = action[:self.act_u_dim], action[self.act_u_dim:]
            action_list = []
            if len(u)>0:
                action_list.append(u)
            if len(c)>0:
                action_list.append(c)
            env_actions.append(action_list)
            
            actions.append(action)
        
        return env_actions, actions
    
    def add(self, obs, next_obs, actions, rewards, dones):
        for i in range(self.n_agents):
            self.buffer[i].add(obs[i], next_obs[i], actions[i], rewards[i], dones[i])

    def train(self):
        if len(self.buffer[0]) < WARMUP_SIZE:
            return 0.0, 0.0, 0.0

        obs, next_obs, actions, rewards, dones = [], [], [], [], []
        for i in range(self.n_agents):
            obs_, next_obs_, action, reward, done = self.buffer[i].sample(BATCH_SIZE)
            obs.append(obs_) # (B, obs_dim)
            next_obs.append(next_obs_) # (B, obs_dim)
            actions.append(action) # (B, act_dim)
            rewards.append(reward) # (B, 1)
            dones.append(done) # (B, 1)
        
        obs = [torch.tensor(np.vstack(obs[i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)] # 
        actions = [torch.tensor(np.vstack(actions[i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.tensor(np.vstack(next_obs[i]), dtype=torch.float32).to(self.device) for i in range(self.n_agents)]
        rewards = [torch.tensor(rewards[i], dtype=torch.float32).to(self.device).to(self.device) for i in range(self.n_agents)]
        dones = [torch.tensor(dones[i], dtype=torch.float32).to(self.device).to(self.device) for i in range(self.n_agents)]
        # (n_agents, B, obs_dim), (n_agents, B, act_dim), (n_agents, B, obs_dim), (n_agents, B, 1), (n_agents, B, 1)

        obs_all = torch.cat(obs, dim=1)
        actions_all = torch.cat(actions, dim=1)
        next_obs_all = torch.cat(next_obs, dim=1)
        # (B, n_agents * obs_dim), (B, n_agents * act_dim), (B, n_agents * obs_dim)

        total_actor_loss = 0
        total_critic_loss = 0
        total_q_value = 0

        with torch.no_grad():
            next_actions = [self.agents[i].target_actor(next_obs[i]) for i in range(self.n_agents)] # [(B, act_dim)] * n_agents
            next_actions_all = torch.cat(next_actions, dim=1) # (B, n_agents * act_dim)

        for i in range(self.n_agents):
            with torch.no_grad():
                # A centric Q value
                target_Q = self.agents[i].target_critic(next_obs_all, next_actions_all) # (B, 1)
                y = rewards[i] + GAMMA * (1 - dones[i]) * target_Q # (B, 1)
            
            # critic loss
            current_Q = self.agents[i].critic(obs_all, actions_all)
            critic_loss = nn.MSELoss()(current_Q, y)
            self.agents[i].learn_critic(critic_loss)

            # actor loss
            curr_actions = [self.agents[j].actor(obs[j]) if j == i 
                            else actions[j].detach() 
                            for j in range(self.n_agents)] # [(B, act_dim)] * n_agents
            curr_actions_all = torch.cat(curr_actions, dim=1) # (B, n_agents * act_dim)
            actor_loss = -self.agents[i].critic(obs_all, curr_actions_all).mean()
            self.agents[i].learn_actor(actor_loss)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_q_value += y.mean().item()

            self.agents[i].update_targets()

        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents, total_q_value / self.n_agents

    def save_models(self, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        for agent in self.agents.values():
            agent.save(save_dir=save_dir)
        
    def load_models(self, save_dir='checkpoints'):
        for agent in self.agents.values():
            agent.load(save_dir=save_dir)

# ==== Training Loop ====
def main():
    print("Training MADDPG for {} agents in {} environment...".format(N_AGENTS, ENV_NAME), flush=True)
    print("DIM_C: ", DIM_C, flush=True)
    print("SHARED_REWARD: ", SHARED_REWARD, flush=True)
    print("========================================", flush=True)
    print("Hyperparameters: ", flush=True) 
    print("- LR_ACTOR: ", LR_ACTOR, flush=True)
    print("- LR_CRITIC: ", LR_CRITIC, flush=True)
    print("- GAMMA: ", GAMMA, flush=True)
    print("- TAU: ", TAU, flush=True)
    print("- BATCH_SIZE: ", BATCH_SIZE, flush=True)
    print("- BUFFER_SIZE: ", BUFFER_SIZE, flush=True)
    print("- EPISODES: ", EPISODES, flush=True)
    print("- STEPS_PER_EPISODE: ", STEPS_PER_EPISODE, flush=True)
    print("- CLIP_NORM: ", CLIP_NORM, flush=True)
    print("- NOISE_SCALE: ", NOISE_SCALE, flush=True)
    print("- WARMUP_SIZE: ", WARMUP_SIZE, flush=True)
    print("- HIDDEN_DIM: ", HIDDEN_DIM, flush=True)
    print("========================================", flush=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS, n_landmarks=N_AGENTS,
                   shared_reward=SHARED_REWARD, dim_c=DIM_C)
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset()
    os.makedirs(SAVE_DIR, exist_ok=True)

    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    print("obs_dims:", obs_dims, "act_dims:", act_dims, flush=True)

    multi_agent = MADDPG(obs_dims, act_dims, act_u_dim, N_AGENTS, device)
    ep_actor_losses = []
    ep_critic_losses = []
    returns = []
    return_dict = {i: [] for i in range(N_AGENTS)}
    ep_q_values = []

    for episode in tqdm(range(EPISODES)):
        obs = env.reset()
        obs = np.array(obs)
        actor_losses = []
        critic_losses = []
        q_values = []

        total_reward = np.zeros((N_AGENTS,))
        done = False
        prev_rewards = None  # Initialize previous rewards to a large value

        for step in range(STEPS_PER_EPISODE):
            env_actions, actions = multi_agent.select_action(obs)
            next_obs, rewards, dones, info = env.step(env_actions)
            
            # print(rewards, flush=True)
            # input()
            total_reward += rewards
            actions = np.array(actions)
            next_obs = np.array(next_obs)
            rewards = np.array(rewards)
            # rewards = 1 + np.clip(np.array(rewards), -10, 0) / 10
            incremental_rewards = np.array(rewards) - np.array(prev_rewards) if prev_rewards is not None else np.zeros_like(rewards)
            prev_rewards = rewards
            dones = np.array(dones)

            # multi_agent.add(obs, next_obs, actions, rewards, dones)
            multi_agent.add(obs, next_obs, actions, incremental_rewards, dones)
            actor_loss, critic_loss, q_value = multi_agent.train()
            obs = next_obs

            if actor_loss != 0.0 or critic_loss != 0.0:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                q_values.append(q_value)

            if dones.all():
                break

        returns.append(total_reward.mean())
        for i in range(N_AGENTS):
            return_dict[i].append(total_reward[i])
        ep_actor_losses.append(np.mean(actor_losses))
        ep_critic_losses.append(np.mean(critic_losses))
        ep_q_values.append(np.mean(q_values))

        if episode % 1 == 0:
            print(f"Episode {episode}: return {returns[-1]:.2f}, ",
                  f"actor loss {np.mean(actor_losses):.4f}, ",
                  f"critic loss {np.mean(critic_losses):.4f}", 
                  f"q value {np.mean(q_values):.4f}", flush=True)

        if episode % 10 == 0:  # adjust frequency
            draw_result(returns, return_dict, ep_actor_losses, ep_critic_losses, ep_q_values, save_dir=SAVE_DIR, print_single=(SHARED_REWARD==0))
            multi_agent.save_models(save_dir=SAVE_DIR)

def draw_result(returns, return_dict,actor_losses, critic_losses, q_values, save_dir='checkpoints', print_single=False):
    episodes = range(1, len(returns) + 1)

    plt.figure(figsize=(15, 4))
    smooth_interval = 100

    def smooth(data, interval):
        return [np.mean(data[i-min(i, interval):i]) for i in range(len(data))]
    
    # Reward
    plt.subplot(1, 4, 1)
    if print_single:
        plt.plot(episodes, returns, label='Average Return')
        plt.plot(episodes, smooth(returns, smooth_interval), label='Smoothed Return', color='red')
    else:
        for i in range(N_AGENTS):
            plt.plot(episodes, return_dict[i], label='Agent {}'.format(i), alpha=0.5)
        for i in range(N_AGENTS):
            plt.plot(episodes, smooth(return_dict[i], smooth_interval), label='Agent {} smoothed'.format(i))
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Average Return")
    plt.grid(True)

    # Actor loss
    plt.subplot(1, 4, 2)
    plt.plot(episodes, actor_losses, label='Actor Loss', color='orange')
    # plt.plot(episodes, smooth(actor_losses, smooth_interval), label='Smoothed Return', color='red')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.grid(True)

    # Critic loss
    plt.subplot(1, 4, 3)
    plt.plot(episodes, critic_losses, label='Critic Loss', color='green')
    # plt.plot(episodes, smooth(critic_losses, smooth_interval), label='Smoothed Return', color='red')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.grid(True)

    # Q value
    plt.subplot(1, 4, 4)
    plt.plot(episodes, q_values, label='Q Value', color='purple')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    plt.title("Q Value")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_ddpg_{}_{}.png".format(ENV_NAME, N_AGENTS)))  # or plt.show() if you prefer
    plt.close()

if __name__ == '__main__':
    main()
