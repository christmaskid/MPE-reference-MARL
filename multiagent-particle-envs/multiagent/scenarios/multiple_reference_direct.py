import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys
class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.norm_direction = False  # normalize direction vectors
        self.reward_shaping = False
        
    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
            agent.my_goal = None
            agent.listen_to = None
        # want other agent to go to the goal landmark

        # random permutation of agents and landmarks
        # Randomly permute agents and landmarks
        target_agents = np.random.permutation(self.n_agents)

        for i in range(self.n_agents):
            if i == target_agents[i]:
                other_agents = [j for j in range(self.n_agents) if j != i]
                j = np.random.choice(other_agents)
                target_agents[i], target_agents[j] = target_agents[j], target_agents[i]

        for (i, agent) in enumerate(world.agents):
            agent.id = i
            agent.goal_a = world.agents[target_agents[i]]
            agent.goal_b = world.landmarks[i]
            agent.goal_a.my_goal = agent.goal_b
            agent.goal_a.listen_to = agent
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = colorsys.hsv_to_rgb(i/self.n_agents,1.0,1.0)

        # special colors for goals
        for agent in world.agents:
            if agent.goal_a is not None and agent.goal_b is not None:
                # Set goal_a's color to be the same as goal_b's color for easy identification
                agent.goal_a.color = np.copy(agent.goal_b.color)                      

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.state.p_pos - agent.my_goal.state.p_pos))
        # dist2 += np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2


    def observation(self, agent, world):
        # (x, y, tx, ty, comm)
        my_id=np.zeros(self.n_agents,)
        target_id=np.zeros(self.n_agents)
        my_id[agent.id] = 1
        target_id[agent.goal_a.id] = 1
        return np.concatenate([
            my_id,
            agent.state.p_pos, 
            agent.state.p_vel,
            target_id,
            agent.goal_b.state.p_pos, 
            world.agents[agent.listen_to.id].state.c])
 
            
