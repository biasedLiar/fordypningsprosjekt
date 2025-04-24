import numpy as np
import matplotlib.pyplot as plt

class StateTrigExpander:
    def __init__(self, observation_space_size, observation_limits, segments=5, gaussian_width=1):
        self.observation_space_size = observation_space_size

        # These are hardcoded for the environment.
        # 1st and 3rd set of limits are based on ob-space
        # 2nd and 4th set of limits are based on experimentation.
        self.observation_limits = observation_limits
        self.observation_space_size = observation_space_size
        self.freqs = segments
        self.gaussian_width = gaussian_width
        self.delta = 10**-8
        self.add_vector = np.average(self.observation_limits, axis=1)
        self.div_vector = (self.observation_limits[:, 1] - self.observation_limits[:, 0])/np.pi

        self.range_matrix = np.outer(np.ones(self.observation_space_size), np.power(0.5, range(self.freqs)))
        #print(self.range_matrix)

    def expand(self, state):
        standardized_state = (state - self.add_vector)/self.div_vector
        matrix = (np.outer(standardized_state, np.ones(self.freqs)) * self.range_matrix)
        expanded_state = np.append(np.cos(matrix), np.sin(matrix))
        return expanded_state

        out = np.zeros(2*self.observation_space_size*self.freqs)
        for i in range(self.observation_space_size):
            for j in range(self.freqs):
                index = 2*(i*self.freqs + j)
                out[index] = np.cos(standardized_state[i]/np.power(2, j))
                out[index+1] = np.sin(standardized_state[i])
        print(out)
        input()
        return out

        standardized_state = (state - self.add_vector)/self.div_vector
        dist = self.range_matrix.transpose() - standardized_state
        weight = np.exp(-np.square(dist) / self.gaussian_width) + self.delta
        expanded_state = weight.flatten('F')
        return expanded_state

