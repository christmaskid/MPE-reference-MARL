import numpy as np
import torch
import os

os.environ['SUPPRESS_MA_PROMPT']='1'
import sys
sys.path.insert(0,'multiagent-particle-envs')
from make_env import make_env
import imageio
import argparse
DIM_C = 10
SHARED_REWARD = 0
act_u_dim = 2
REWARD_ALPHA = 0 # the weight of learning communication vs. movement

STEPS_PER_EPISODE = 25

def main():
    parser = argparse.ArgumentParser(description="Test MASAC on multi-agent particle environments.")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--n_agents", type=int, required=True, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to load/save models and results")
    parser.add_argument("--render", action="store_true", help="Render environment and save gifs")
    parser.add_argument("--vanilla", action="store_true", help="Use vanilla MADDPG without communication")
    args = parser.parse_args()

    ENV_NAME = args.env_name
    N_AGENTS = args.n_agents
    EPISODES = args.episodes
    if args.save_dir is not None:
        SAVE_DIR = args.save_dir
    else:
        if args.vanilla:
            SAVE_DIR = "results/maddpg-vanilla_" + (ENV_NAME.split("_")[-1]) + "_" + str(N_AGENTS) + "agents_" + str(EPISODES)
        else:
            SAVE_DIR = "results/maddpg-commloss_" + (ENV_NAME.split("_")[-1]) + "_" + str(N_AGENTS) + "agents_" + str(EPISODES)
    
    if args.vanilla:
        from train_maddpg_vanilla import MADDPG
    else:
        from train_maddpg import MADDPG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_NAME, n_agents=N_AGENTS, n_landmarks=N_AGENTS,
                   shared_reward=SHARED_REWARD, dim_c=DIM_C, reward_alpha=REWARD_ALPHA, training=False)
    env.reset()
    np.random.seed(0)
    torch.manual_seed(0)

    print("Testing environment: ", ENV_NAME)
    print("Number of agents: ", N_AGENTS)
    print("Model directory: ", SAVE_DIR)
    print("Number of steps per episode: ", STEPS_PER_EPISODE)
    
    obs_dims = env.observation_space[0].shape[0]
    act_dims = sum([sp.shape[0] for sp in env.action_space[0].spaces])
    print("obs_dims:", obs_dims, "act_dims:", act_dims, flush=True)

    multi_agent = MADDPG(obs_dims, act_dims, act_u_dim, N_AGENTS, device)
    if os.path.exists(SAVE_DIR):
        print("Loaded models from", SAVE_DIR, flush=True)
        multi_agent.load_models(SAVE_DIR)
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

            if args.render:
                frame = np.array(env.render(mode='rgb_array'))
                frames.append(frame[0])

            if dones.all():
                break

        print("Total reward for episode {}: {}".format(ep, total_reward), flush=True)
        
        # Save frames as gif
        if args.render:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            save_path = os.path.join(SAVE_DIR, 'out{}.gif'.format(ep))
            imageio.mimsave(save_path, frames, duration=0.01, save_all=True, append_images=frames[1:], optimize=True, loop=0)
            save_img_path = os.path.join(SAVE_DIR, 'out{}.png'.format(ep))
            imageio.imwrite(save_img_path, frames[-1])
            print("Saved episode as {}".format(save_path))
            
        print(f"Average Total reward for each agent: {total_reward.mean()}", f"{total_reward}", flush=True)
        avg_total_reward += total_reward.mean()
    
    print("Average total reward over 10 episodes for each agent:", avg_total_reward / 10, flush=True)
    env.close()

if __name__ == "__main__":
    main()
