import numpy as np


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def rolling_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    rolling_average = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    print(window_width)
    shifted_rolling_avg = np.append([None for i in range(window_width-1)], rolling_average)
    return shifted_rolling_avg
