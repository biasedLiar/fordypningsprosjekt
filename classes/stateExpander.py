import numpy as np
import matplotlib.pyplot as plt

class StateExpander:
    def __init__(self, observation_space_size, observation_limits, segments=5, gaussian_width=1):
        self.observation_space_size = observation_space_size

        # These are hardcoded for the environment.
        # 1st and 3rd set of limits are based on ob-space
        # 2nd and 4th set of limits are based on experimentation.
        self.observation_limits = observation_limits
        self.segments = segments
        self.gaussian_width = gaussian_width
        self.delta = 10**-8
        self.add_vector = np.average(self.observation_limits, axis=1)
        self.div_vector = (self.observation_limits[:, 1] - self.observation_limits[:, 0]) / (self.segments-1)

        self.range_matrix = np.outer(np.ones(self.observation_space_size), np.linspace(-(self.segments-1)/2, (self.segments-1)/2, self.segments))


    def create_alteration_matrix(self):
        if self.segments < 2:
            raise Exception("At least 2 segments required")
        base_matrix = np.tile(self.observation_limits[:, 0], (self.segments, 1)).transpose()
        diff_vector = (self.observation_limits[:, 1] - self.observation_limits[:, 0]) / (self.segments-1)
        range_matrix = np.outer(np.ones(self.observation_space_size), np.linspace(0, self.segments-1, self.segments))
        alteration_matrix = base_matrix
        div_vector = (self.observation_limits[:, 1] - self.observation_limits[:, 0]) / (self.segments-1)
        input()
        return alteration_matrix


    def expand(self, state):
        standardized_state = (state - self.add_vector)/self.div_vector
        dist = self.range_matrix.transpose() - standardized_state
        weight = np.exp(-np.square(dist) / self.gaussian_width) + self.delta
        expanded_state = weight.flatten('F')
        return expanded_state

