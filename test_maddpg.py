import numpy as np
import torch
import os

os.environ['SUPPRESS_MA_PROMPT']='1'
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env
import imageio


from train_maddpg_mut_critic import MADDPG
ENV_NAME = "multiple_reference_broadcast"
N_AGENTS = 3
DIM_C = 10
SHARED_REWARD = 0
act_u_dim = 2
REWARD_ALPHA = 0 # the weight of learning communication vs. movement

SAVE_DIR = f"maddpg_{ENV_NAME}_{N_AGENTS}_{DIM_C}_{SHARED_REWARD}_{REWARD_ALPHA}_{act_u_dim}_mut_broadcast" #'models/'

STEPS_PER_EPISODE = 30

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS, n_landmarks=N_AGENTS,
                   shared_reward=SHARED_REWARD, dim_c=DIM_C, reward_alpha=REWARD_ALPHA, training=False)
    env.reset()

    print("Testing environment: ", ENV_NAME)
    print("Number of agents: ", N_AGENTS)
    print("Model directory: ", SAVE_DIR)
    print("Number of steps per episode: ", STEPS_PER_EPISODE)
    
    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    print("obs_dims:", obs_dims, "act_dims:", act_dims, flush=True)

    multi_agent = MADDPG(obs_dims, act_dims, act_u_dim, N_AGENTS, device)
    if os.path.exists(SAVE_DIR):
        multi_agent.load_models(SAVE_DIR)
        print("Loaded models from", SAVE_DIR, flush=True)
    else:
        print("Model directory does not exist. Use random initialized model.", flush=True)

    avg_total_reward = 0

    for ep in range(10):  # Run 10 episodes
        returns = []
        obs = env.reset()
        obs = np.array(obs)
        total_reward = np.zeros((N_AGENTS,))
        frames = []
        
        for (i, agent) in enumerate(env.world.agents):
            print("Agent #%d: go to landmark #%d; shall tell agent #%d to go to landmark #%d." 
                % (i, agent.my_goal.id, agent.goal_a.id, agent.goal_b.id), flush=True)

        for step in range(STEPS_PER_EPISODE):
            env_actions, actions = multi_agent.select_action(obs, noise_scale=0)#.1)
            next_obs, rewards, dones, info = env.step(env_actions)

            # broadcast_agent = env.world.agents[env.world.steps % len(env.agents)]
            # print(f"Step {step}, agent {broadcast_agent.id} broadcasts: {broadcast_agent.state.c}", flush=True)
            # for i in range(len(env.world.agents)):
            #     print(f"Step {step}, agent {i} action: {actions[i]}", flush=True)
            
            total_reward += rewards
            actions = np.array(actions)
            next_obs = np.array(next_obs)
            rewards = np.array(rewards)
            dones = np.array(dones)
            obs = next_obs

            frame = np.array(env.render(mode='rgb_array'))
            frames.append(frame[0])

            if dones.all():
                break

        print("Total reward for episode {}: {}".format(ep, total_reward), flush=True)
        
        # Save frames as gif
        save_path = os.path.join(SAVE_DIR, 'maddpg_test_{}_{}_out{}.gif'.format(ENV_NAME, N_AGENTS, ep))
        imageio.mimsave(save_path, frames, duration=0.01)
        print("Saved episode as {}".format(save_path))
        print("Total reward:", total_reward.sum(), flush=True)
        avg_total_reward += total_reward.sum()
    
    print("Average total reward over 10 episodes:", avg_total_reward / 10, flush=True)
    env.close()

if __name__ == "__main__":
    main()
