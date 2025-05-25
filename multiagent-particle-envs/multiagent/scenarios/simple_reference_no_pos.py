import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.base_reference import BaseReferenceScenario
import colorsys

class Scenario(BaseReferenceScenario):
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_landmarks = 2
        self.use_landmark_pos = False
        self.use_agent_id = False
