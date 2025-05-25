import numpy as np
import torch
import os

os.environ['SUPPRESS_MA_PROMPT']='1'
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env

from train_maddpg import MADDPG
import imageio

ENV_NAME = 'speaking' #'simple_reference_alpha' #"simple_reference_no_pos" #"multiple_reference_no_pos" #"multiple_reference" # # 
N_AGENTS = 2 #3 #2 # 
STEPS_PER_EPISODE = 1 #00
SAVE_DIR = f"maddpg_{ENV_NAME}_{N_AGENTS}" #'models/'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS)
    env.reset()

    print("Testing environment: ", ENV_NAME)
    print("Number of agents: ", N_AGENTS)
    print("Model directory: ", SAVE_DIR)
    print("Number of steps per episode: ", STEPS_PER_EPISODE)
    
    for (i, agent) in enumerate(env.world.agents):
        print("Agent #%d: go to landmark #%d; shall tell agent #%d to go to landmark #%d." 
              % (i, agent.self_goal.id, agent.goal_a.id, agent.goal_b.id), flush=True)

    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    act_u_dim = 4
    print("obs_dims:", obs_dims, "act_dims:", act_dims, flush=True)

    multi_agent = MADDPG(obs_dims, act_dims, act_u_dim, N_AGENTS, device)
    if os.path.exists(SAVE_DIR):
        multi_agent.load_models(SAVE_DIR)
        print("Loaded models from", SAVE_DIR, flush=True)
    else:
        print("Model directory does not exist. Use random initialized model.", flush=True)

    returns = []
    obs = env.reset()
    obs = np.array(obs)
    total_reward = np.zeros((N_AGENTS,))
    frames = []

    for step in range(STEPS_PER_EPISODE):
        env_actions, actions = multi_agent.select_action(obs)
        next_obs, rewards, dones, info = env.step(env_actions)

        broadcast_agent = env.world.agents[env.world.steps % len(env.agents)]
        print(f"Step {step}, agent {broadcast_agent.id} broadcasts: {broadcast_agent.state.c}", flush=True)
        
        total_reward += rewards
        actions = np.array(actions)
        next_obs = np.array(next_obs)
        # rewards = 1 + np.clip(np.array(rewards), -10, 0) / 10
        rewards = np.array(rewards)
        dones = np.array(dones)
        obs = next_obs
        print(rewards, flush=True)

        returns.append(total_reward.mean())

        frame = np.array(env.render(mode='rgb_array'))
        frames.append(frame[0])

        if ENV_NAME == 'speaking':
            print("Step {}".format(step), flush=True)
            for i in range(N_AGENTS):
                agent = env.world.agents[i]
                print(f"Agent {i} target direction: {agent.self_goal.state.p_pos - agent.state.p_pos}, ",
                      f"communication: {agent.state.c}", flush=True)
        
        if dones.all():
            break
    
    # Save frames as gif
    save_path = 'maddpg_test_{}_{}.gif'.format(ENV_NAME, N_AGENTS)
    imageio.mimsave(save_path, frames, duration=0.01)
    print("Saved episode as {}".format(save_path))

if __name__ == "__main__":
    main()
