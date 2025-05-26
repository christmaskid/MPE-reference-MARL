import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.norm_direction = False  # normalize direction vectors
        self.reward_alpha = 1
        self.reward_shaping = False
        
    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        dist_self = np.sum(np.square(agent.state.p_pos - agent.self_goal.state.p_pos))

        # if self.reward_shaping:
        #     SCALE = 4
        #     for landmark in world.landmarks:
        #         if landmark.id != agent.goal_b.id:
        #             dist2 -= np.sum(np.square(agent.goal_a.state.p_pos - landmark.state.p_pos)) / SCALE
        #         if landmark.id != agent.self_goal.id:
        #             dist_self -= np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) / SCALE

        return -dist2 * self.reward_alpha - dist_self * (1 - self.reward_alpha)
    
    def observation(self, agent, world):
        # where am I?
        self_pos = agent.state.p_pos
        self_vel = agent.state.p_vel

        # who am I speaking to? where should they go?
        speakto_dst = agent.goal_b.state.p_pos if agent.goal_b is not None else (0, 0)
                
        # communication of the other agent
        comm = agent.listen_to.state.c
    
        return np.concatenate([
            self_pos,
            self_vel,
            speakto_dst,
            comm
        ])