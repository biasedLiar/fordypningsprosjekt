import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

import helper.plotHelper as plotHelper
import helper.KNearestNeighbor as kNearest
import helper.fileHelper as fileHelper


############### Constants ###################
CUTOFF = True
CUTOFFPOINT = 100

STOP_AFTER_CONSEC_500S = False

ANGLE = 0.2095
# ANGLE = 1.0

SHOW_GAMES = False

MANUAL_ROLLING_AVERAGE = -1

K_START = 1
K_END = 21
K_STEP = 2

########### End constants #################

path = f"plots\\kNearest\\{ANGLE}_angle\\{CUTOFFPOINT}_gens"
fileHelper.createDirIfNotExist(path)

if SHOW_GAMES:
    render_mode = "human"  # Set to None to run without graphics
else:
    render_mode = None

if MANUAL_ROLLING_AVERAGE == -1:
    window_width = round(CUTOFFPOINT/26.67)
else:
    window_width = MANUAL_ROLLING_AVERAGE

def run_k_nearest(k=-1, show_results=True, save_results=True, window_width=5):
    if k > 0:
        kNearest.K = k
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(0)
    np.random.seed(0)

    observation, info = env.reset(seed=0)

    steps_alive = 0
    terminated_observations = np.zeros((0, 4))
    terminated_observations_normalized = np.zeros((0, 4))
    mean = np.zeros((4))
    std = np.ones((4))
    last_closest_distance_change = None
    action = 0
    last_it_succeeded = False
    its_before_finished = 0

    data = np.array([], dtype=int)

    while its_before_finished < CUTOFFPOINT:
        last_closest_distance = kNearest.k_nearest_distance(terminated_observations_normalized, observation, mean, std)
        observation, reward, terminated, truncated, info = env.step(action)

        closest_distance = kNearest.k_nearest_distance(terminated_observations_normalized, observation, mean, std)

        closest_distance_change = closest_distance - last_closest_distance

        if closest_distance_change >= 0.0 or (last_closest_distance_change != None and closest_distance_change > last_closest_distance_change):
            pass
        else:
            action = (action + 1) % 2

        last_closest_distance_change = closest_distance_change

        steps_alive += 1

        if np.abs(observation[2]) > ANGLE or np.abs(observation[0]) > 2.4 or truncated:
            if not truncated:
                terminated_observations = np.concatenate((terminated_observations, [observation]), axis=0)
                mean = terminated_observations.mean(axis=0)
                std = terminated_observations.std(axis=0)
                std = np.where(std != 0., std, 1.)  # To avoid division by zero
                terminated_observations_normalized = (terminated_observations - mean) / std
                last_it_succeeded = False
            else:
                if last_it_succeeded and STOP_AFTER_CONSEC_500S:
                    print(f"Steps alive: {steps_alive}")
                    break
                else:
                    last_it_succeeded = True

            print(f"{its_before_finished}: Steps alive: {steps_alive}")
            data = np.append(data, [steps_alive])
            steps_alive = 0
            last_closest_distance_change = None
            observation, info = env.reset()
            its_before_finished += 1

    print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{its_before_finished}")
    rolling_avg = plotHelper.rolling_average(data, window_width)

    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label="10-step avg")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor Classification, k={kNearest.K}, angle={ANGLE}")

    plot_name = path + f"\\K{kNearest.K}__plot.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()
    return data


def run_k_nearest_new_plotting(k=-1, show_results=True, save_results=True, window_width=5):
    if k > 0:
        kNearest.K = k
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(0)
    np.random.seed(0)

    observation, info = env.reset(seed=0)

    steps_alive = 0
    terminated_observations = np.zeros((0, 4))
    terminated_observations_normalized = np.zeros((0, 4))
    mean = np.zeros((4))
    std = np.ones((4))
    last_closest_distance_change = None
    action = 0
    last_it_succeeded = False
    its_before_finished = 0

    data = np.array([], dtype=int)

    while its_before_finished < CUTOFFPOINT:
        last_closest_distance = kNearest.k_nearest_distance(terminated_observations_normalized, observation, mean, std)
        observation, reward, terminated, truncated, info = env.step(action)

        closest_distance = kNearest.k_nearest_distance(terminated_observations_normalized, observation, mean, std)

        closest_distance_change = closest_distance - last_closest_distance

        if closest_distance_change >= 0.0 or (last_closest_distance_change != None and closest_distance_change > last_closest_distance_change):
            pass
        else:
            action = (action + 1) % 2

        last_closest_distance_change = closest_distance_change

        steps_alive += 1

        if np.abs(observation[2]) > ANGLE or np.abs(observation[0]) > 2.4 or truncated:
            if not truncated:
                terminated_observations = np.concatenate((terminated_observations, [observation]), axis=0)
                mean = terminated_observations.mean(axis=0)
                std = terminated_observations.std(axis=0)
                std = np.where(std != 0., std, 1.)  # To avoid division by zero
                terminated_observations_normalized = (terminated_observations - mean) / std
                last_it_succeeded = False
            else:
                if last_it_succeeded and STOP_AFTER_CONSEC_500S:
                    print(f"Steps alive: {steps_alive}")
                    break
                else:
                    last_it_succeeded = True

            print(f"{its_before_finished}: Steps alive: {steps_alive}")
            data = np.append(data, [steps_alive])
            steps_alive = 0
            last_closest_distance_change = None
            observation, info = env.reset()
            its_before_finished += 1

    print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{its_before_finished}")
    rolling_avg = plotHelper.rolling_average(data, window_width)

    avg = np.sum(data)/len(data)
    avg_plot = np.full((CUTOFFPOINT), avg)
    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{window_width}-step avg")
    plt.plot(avg_plot, label=f"avg score")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor, k={kNearest.K}, avg score={avg}")

    plot_name = path + f"\\K{kNearest.K}__plot.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()

    avg = np.sum(data) / len(data)

    rolling_avg = plotHelper.rolling_average(data, 100)
    avg_plot = np.full((CUTOFFPOINT), 475)
    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{100}-step avg")
    plt.plot(avg_plot, label=f"CartPole-v1 threshold")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor, k={kNearest.K}")

    plot_name = path + f"\\K{kNearest.K}__plot-v1.png"
    if False:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()



    data = np.array([(200 if datum > 200 else datum) for datum in data])

    avg_plot = np.full((CUTOFFPOINT), 195)
    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{100}-step avg")
    plt.plot(avg_plot, label=f"CartPole-v0 threshold")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor Classification, k={kNearest.K}")

    plot_name = path + f"\\K{kNearest.K}__plot-v0.png"
    if False:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()


    return data


def run_and_compare_range_k_nearest(bottom, top, window_width, step=1):
    survival_stats = []
    for i in range(bottom, top+1, step):
        survival_stat = run_k_nearest(k=i, show_results=False, save_results=True)
        survival_stats.append(survival_stat)

    for i in range(len(survival_stats)):
        plt.plot(survival_stats[i], label=f"k={i*step + bottom}")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor Classification, k={range(bottom, top + 1, step)}, angle={ANGLE}")
    plot_name = path +  f"\\K{bottom}-K{top}__group__plot.png"
    plt.savefig(plot_name)
    plt.clf()

    for i in range(len(survival_stats)):
        rolling_avg = plotHelper.rolling_average(survival_stats[i], window_width)
        plt.plot(rolling_avg, label=f"k={step*i + bottom}")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor, Rolling avg of {window_width}, k={range(bottom, top + 1, step)}, angle={ANGLE}")
    plot_name = path + f"\\K{bottom}-K{top}__group__rolling-avg-{window_width}__plot.png"
    plt.savefig(plot_name)
    plt.clf()


def run_standard_iterated():
    survival_stats = []
    fileHelper.createDirIfNotExist(path)
    neighbors_list = [3, 5, 7, 10, 15, 20]
    for neighbors in neighbors_list:


        print("\n\n\n-----------------------------")
        print(f"Starting {neighbors=}")
        print("------------------------------\n\n\n")
        inner_start = time()
        survival_stat = run_k_nearest_new_plotting(k=neighbors, show_results=False, save_results=True)
        survival_stats.append(survival_stat)
        inner_stop = time()
        print("----------------------------")
        print(f"Finished {neighbors=}, time= {round(inner_stop - inner_start)} seconds")
        print("------------------------------")

# shift file saving

if __name__ == '__main__':
    # run_k_nearest()
    #run_and_compare_range_k_nearest(K_START, K_END, window_width, K_STEP)
    run_standard_iterated()