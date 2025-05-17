import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, n_agents=2, n_landmarks=3):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks

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
        world.landmarks = [Landmark() for i in range(self.n_landmarks)]
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
        agent_indices = np.random.permutation(len(world.agents))
        landmark_indices = np.random.permutation(len(world.landmarks))
        for i, agent_idx in enumerate(agent_indices):
            agent = world.agents[agent_idx]
            # Assign another agent as goal_a (excluding self)
            other_agents = [a for j, a in enumerate(world.agents) if j != agent_idx]
            agent.goal_a = np.random.choice(other_agents)
            # Assign a random landmark as goal_b
            agent.goal_b = world.landmarks[landmark_indices[i % len(world.landmarks)]]
        
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.random.uniform(0, 1, 3)

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
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
            