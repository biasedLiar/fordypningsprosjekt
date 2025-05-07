import numpy as np
import matplotlib.pyplot as plt
from helper import fileHelper


color_list = ['blue', 'red', 'green', 'pink', 'orange', 'purple', 'cyan', 'yellow']

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
    avg_data = np.mean(data, axis=0)
    return avg_data

def get_upper_lower_error_bounds(data, avg_reward):
    max_data = np.max(data, axis=0) - avg_reward
    min_data = avg_reward - np.min(data, axis=0)
    return [min_data, max_data]

def average_max_min_diagrams(data):
    avg_data = np.mean(data, axis=0)
    max_data = np.max(data, axis=0)
    min_data = np.min(data, axis=0)
    return avg_data, max_data, min_data


def get_std_deviation(datas):
    sigma1 = datas.std(axis=0)
    return sigma1

def get_rewards(reward_list, gamma):
    num = len(reward_list)
    out_list = np.zeros(num)
    for i in range(num):
        reward = reward_list[num - i -1]
        if i != 0:
            reward += gamma * out_list[num - i]
        out_list[num - i - 1] = reward
    return out_list

def plot_with_max_min_mean_std(datas, plot_name, plot_title):
    avg_data, max_data, min_data = average_max_min_diagrams(datas)
    avg_num = np.round(np.mean(avg_data), 2).item()

    std = datas.std(axis=0)
    plt.plot(max_data, label='max_steps')
    plt.plot(avg_data, label='avg_steps')
    plt.plot(min_data, label='min_steps')
    plt.fill_between(range(len(std)), avg_data - std, avg_data + std, facecolor='orange', alpha=0.5, label='std')
    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="lower right")
    plt.title(plot_title + f", {avg_num=}")
    plt.ylim(0, 525)
    #fileHelper.createDirIfNotExist(plot_name)
    plt.savefig(plot_name)
    plt.show()
    plt.clf()
    print()



def plot_multiple_graph_types(datas_list, labels, plot_name, plot_title, show_std=False):
    if len(labels) <= 1 or len(labels) != len(datas_list) or len(labels) > 8:
        return
    for i in range(len(labels)):
        avg_data, max_data, min_data = average_max_min_diagrams(datas_list[i])
        plt.plot(avg_data, label=labels[i] + ' mean', color=color_list[i])
        if show_std:
            std = datas_list[i].std(axis=0)
            plt.fill_between(range(len(std)), avg_data - std, avg_data + std, facecolor=color_list[i], alpha=0.5,
                             label=labels[i] + ' std')
    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="lower right")
    plt.title(plot_title)
    plt.ylim(0, 525)
    plt.savefig(plot_name)
    plt.show()
    plt.clf()
    print()