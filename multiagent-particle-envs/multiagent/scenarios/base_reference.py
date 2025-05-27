import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import colorsys

class BaseReferenceScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.use_landmark_pos = False
        self.use_agent_id = True
        self.reward_alpha = 1 # the weight of learning communication vs. movement 
        self.norm_direction = False
        self.landmark_movable = False

    def make_world(self, n_agents=2, n_landmarks=None, shared_reward=0.5, dim_c=10, reward_alpha = 1, training=True):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks if n_landmarks is not None else self.n_agents
        self.shared_reward = shared_reward
        self.reward_alpha = reward_alpha
        self.training = training

        world = World()
        # set any world properties first
        world.dim_c = dim_c
        world.shared_reward = shared_reward  # whether agents share rewards

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
            landmark.movable = self.landmark_movable or False
            landmark.id = i
        
        # make initial conditions
        self.custom_init(world)
        self.reset_world(world)
        return world
    
    def custom_init(self, world):
        pass

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
            agent.self_goal = None
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

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        dist_self = np.sum(np.square(agent.state.p_pos - agent.self_goal.state.p_pos))

        # print(f"Reward for agent {agent.id}: ", end="")
        # print(f"agent.goal_a: {agent.goal_a.state.p_pos}, agent.goal_b: {agent.goal_b.state.p_pos}")
        # print(f"agent.state.p_pos: {agent.state.p_pos}, agent.self_goal: {agent.self_goal.state.p_pos}")
        # print(f"dist2: {dist2}, dist_self: {dist_self}")

        return -dist2 * self.reward_alpha - dist_self * (1 - self.reward_alpha)

    def get_broadcast_agent(self, world):
        # Get the agent that is broadcasting at the current step
        return world.agents[world.steps % self.n_agents]

    def observation(self, agent, world):
        # where am I going?
        direction = np.array(agent.state.p_vel)
        if self.norm_direction:
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction

        # who am I speaking to? where should they go?
        speakto_id = agent.goal_a.id if agent.goal_a is not None else -1
        speakto_pos = agent.goal_a.state.p_pos if agent.goal_a is not None else (0, 0)
        speakto_target_pos = agent.goal_b.state.p_pos
        speakto_distance = speakto_target_pos - speakto_pos

        # get relative positions of all entities in this agent's reference frame
        entity_pos = []
        if self.use_landmark_pos:
            for entity in world.landmarks:
                dist = entity.state.p_pos - agent.state.p_pos
                if self.norm_direction:
                    dist = dist / np.linalg.norm(dist) if np.linalg.norm(dist) > 0 else dist
                entity_pos.append(dist)
                
        # the agent that is broadcasting at current step
        comm = world.agents[world.steps % self.n_agents].state.c
        
        if self.use_agent_id:
            return np.concatenate([
                [agent.id],
                direction,
                [speakto_id],
                speakto_distance,
                *entity_pos,
                comm
            ])
        else:
            return np.concatenate([
                # [agent.id],
                direction,
                # [speakto_id],
                speakto_distance,
                *entity_pos,
                comm
            ])