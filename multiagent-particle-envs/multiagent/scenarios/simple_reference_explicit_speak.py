import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_landmarks = 2
        self.use_landmark_pos = True
        self.use_agent_id = False

    def custom_init(self, world):
        world.dim_c = 3
        # world.collaborative = False
        
    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None # the agent it is supposed to speak to
            agent.goal_b = None # the landmark goal_a is supposed to go to
            agent.self_goal = None # the landmark it is supposed to go to
            agent.listen_to = None # the agent it is supposed to listen to
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
            # agent knows the id of the agent it is supposed to communicate with: goal_a
            # and the landmark agent[goal_a] is supposed to go to: goal_b
            agent.id = i
            agent.goal_a = world.agents[target_agents[i]]
            agent.goal_b = world.landmarks[i]

            # assign the goal landmark to the agent that is supposed to go there
            agent.goal_a.self_goal = agent.goal_b
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
