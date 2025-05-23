import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_landmarks = 2

    def observation(self, agent, world):
        # where am I going?
        direction = np.array(agent.state.p_vel)

        # who am I speaking to? where should they go?
        speakto_id = agent.goal_a.id if agent.goal_a is not None else -1
        speakto_pos = agent.goal_a.state.p_pos if agent.goal_a is not None else (0, 0)
        speakto_target_pos = agent.goal_b.state.p_pos
        speakto_distance = speakto_target_pos - speakto_pos

        # get relative positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                
        # the agent that is broadcasting at current step
        comm = world.agents[world.steps % self.n_agents].state.c
        
        return np.concatenate([
            # [agent.id],
            direction,
            # [speakto_id],
            speakto_distance,
            comm
        ])