import numpy as np

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, cap, obs_dim, act_dim):
        self.cap = cap
        self.size = 0
        self.state = np.empty((cap, obs_dim), dtype=np.float32)
        self.action = np.empty((cap, act_dim), dtype=np.float32)
        self.reward = np.empty((cap, 1), dtype=np.float32)
        self.next_state = np.empty((cap, obs_dim), dtype=np.float32)
        self.done = np.empty((cap, 1), dtype=np.float32)
        self.pos = 0

    def add(self, state, next_state, action, reward, done):
        # print("state shape:", state.shape, "next_state shape:", next_state.shape, "action shape:", action.shape, "reward shape:", reward.shape, "done shape:", done.shape, flush=True)
        self.state[self.pos] = state
        self.next_state[self.pos] = next_state
        self.action[self.pos] = action
        self.reward[self.pos, 0] = reward
        self.done[self.pos, 0] = done
        self.last_pos = self.pos

        self.pos = (self.pos+1)%self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = np.random.choice(self.size, n, replace=False)
        states = self.state[idx]
        next_states = self.next_state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        dones = self.done[idx]
        return states, next_states, actions, rewards, dones
    
    def __len__(self):
        return self.size