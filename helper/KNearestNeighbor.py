import numpy as np

K = 3

def k_nearest_distance(terminated_observations_normalized, observation, mean, std):
    num_samples = terminated_observations_normalized.shape[0]
    if num_samples == 0:
        mean_distance = 0.0
    elif num_samples < K*2:
        mean_distance = last_closest_distance = np.min(np.sum(np.square((observation - mean) / std -
                                                terminated_observations_normalized), axis=1))
    else:
        distances = np.sum(np.square((observation - mean) / std - terminated_observations_normalized), axis=1)

        # source: https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        idx = np.argpartition(distances, K)
        mean_distance = np.mean(distances[idx[:K]])
    return mean_distance