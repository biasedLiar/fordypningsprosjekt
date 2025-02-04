import numpy as np


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def rolling_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    rolling_average = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    print(window_width)
    shifted_rolling_avg = np.append([None for i in range(window_width-1)], rolling_average)
    return shifted_rolling_avg

def average_every_n(data, list_of_list=False, n=5):
    out_list = []
    if list_of_list:
        for sublist in data:
            out_list.append(average_every_n(sublist, n=n))
    else:
        data = data[:len(data)-(len(data)%n)]
        out_list = np.mean(data.reshape(-1, n), axis=1)


    return out_list

def average_of_diff_seeds(data):
    np_data = np.asarray(data)
    avg_data = np.mean(np_data, axis=0)
    return avg_data

def get_upper_lower_error_bounds(data, avg_reward):
    np_data = np.asarray(data)
    max_data = np.max(np_data, axis=0) - avg_reward
    min_data = avg_reward - np.min(np_data, axis=0)
    return [min_data, max_data]

def average_max_min_diagrams(data):
    np_data = np.asarray(data)
    avg_data = np.mean(np_data, axis=0)
    max_data = np.max(np_data, axis=0)
    min_data = np.min(np_data, axis=0)
    return avg_data, max_data, min_data





