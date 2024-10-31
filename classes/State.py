class State:
    def __init__(self, obs):
        self.obs = obs

    def __eq__(self, other):
        if isinstance(other, State):
            return self.obs == other.obs
        return NotImplemented

    def add(self, other_state):
        for i in range(4):
            self.obs[i] += other_state.obs[i]

    def __hash__(self):
        return hash((self.obs[0], self.obs[1], self.obs[2], self.obs[3]))

    def __str__(self):
        return str(self.obs)