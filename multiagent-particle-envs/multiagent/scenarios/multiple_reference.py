import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.norm_direction = False  # normalize direction vectors
        self.reward_shaping = False
        self.broadcast = True
        
    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        if not self.training:
            return -dist2

        dist_self = np.sum(np.square(agent.state.p_pos - agent.self_goal.state.p_pos))        
        ret_reward = -dist2 * self.reward_alpha - dist_self * (1 - self.reward_alpha)

        if self.reward_shaping:
            # Don't cluster with other agents
            for other_agent in world.agents:
                if other_agent is not agent and other_agent.state.p_pos is not None:
                    dist_other = np.sum(np.square(agent.state.p_pos - other_agent.state.p_pos))
                    ret_reward -= dist_other * 0.1

            # # Slow down when approaching self goal
            # goal_dist = np.linalg.norm(agent.state.p_pos - agent.self_goal.state.p_pos)
            # if goal_dist < 0.2:
            #     ret_reward -= (0.2 - goal_dist) * 2.0 * np.linalg.norm(agent.state.p_vel)
            
        return ret_reward
    
    def observation(self, agent, world):
        # where am I?
        self_pos = agent.state.p_pos
        self_vel = agent.state.p_vel

        # who am I speaking to? where should they go?
        speakto_dst = agent.goal_b.state.p_pos if agent.goal_b is not None else (0, 0)
                
        # communication of the other agent
        if self.broadcast:
            # If broadcasting, use the communication of the agent that is currently broadcasting
            comm = world.agents[world.steps % self.n_agents].state.c if world.steps % self.n_agents < len(world.agents) else np.zeros(self.dim_c)
        else:
            comm = agent.listen_to.state.c
    
        return np.concatenate([
            # [agent.id],
            self_pos,
            self_vel,
            # [agent.goal_a.id],
            speakto_dst,
            comm
        ])