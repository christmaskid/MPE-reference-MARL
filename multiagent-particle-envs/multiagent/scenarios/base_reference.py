import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import colorsys

class BaseReferenceScenario(BaseScenario):
    def __init__(self):
        super().__init__()

    def make_world(self, n_agents=2, n_landmarks=None):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks if n_landmarks is not None else self.n_agents

        world = World()
        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(self.n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.n_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
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
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2


    def observation(self, agent, world):
        raise NotImplementedError("This method should be implemented in the derived class.")
            
