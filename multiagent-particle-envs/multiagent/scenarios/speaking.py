import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_landmarks = 2
        self.reward_alpha = 0.5  # weight of learning communication vs. movement
    
    def custom_init(self, world):
        world.dim_c = world.dim_p  # communication dimension
        for agent in world.agents:
            agent.movable = False

    def reward(self, agent, world):
        def norm(v):
            return np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros(world.dim_c)
        my_vel_gt = norm(agent.self_goal.state.p_pos - agent.state.p_pos)
        my_vel_pred = norm(agent.action.c)

        target_vel_gt = norm(agent.goal_b.state.p_pos - agent.goal_a.state.p_pos)
        target_val_pred = norm(agent.goal_a.action.c)

        dist_self = np.sum(np.square(my_vel_gt - my_vel_pred))
        dist_target = np.sum(np.square(target_vel_gt - target_val_pred))
        return -dist_target * self.reward_alpha - dist_self * (1 - self.reward_alpha)

    def observation(self, agent, world):
        # who am I speaking to? where should they go?
        speakto_pos = agent.goal_a.state.p_pos if agent.goal_a is not None else (0, 0)
        speakto_target_pos = agent.goal_b.state.p_pos
        speakto_direction = speakto_target_pos - speakto_pos

        # the agent that is broadcasting at current step
        comm = world.agents[world.steps % self.n_agents].state.c
        # one hot encoding of the agent that is broadcasting
        comm_one_hot = np.zeros(world.dim_c)
        comm_one_hot[world.steps % self.n_agents] = 1
        
        return np.concatenate([
            speakto_direction,
            comm_one_hot,
            comm
        ])