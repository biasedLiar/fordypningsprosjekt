import random

import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time


import helper.plotHelper as plotHelper
import helper.KNearestNeighbor as kNearest
from helper.strategy_names import *
import helper.fileHelper as fileHelper
from classes.TreeV3 import *


############### Constants ###################
CUTOFFPOINT = 250
SHOW_GAMES = False
START_STRATEGY = EXPLORE
LAYERS_CHECKED = 3
NEIGHBORS = 15
GAMMA = 0.8




# Set to -1 for automatic rolling average generation.
MANUAL_ROLLING_AVERAGE = 100
CUTOFF = False
STOP_AFTER_CONSEC_500S = False
ANGLE = 0.2095

K_START = 1
K_END = 1
K_STEP = 1

DETERMINISTIC = False
SEMI_DETERMINISTIC = 10
########### End constants #################

if GAMMA != 0.8:
    sigma_path = f"{str(GAMMA)}-sigma\\"
else:
    sigma_path = ""
path = f"plots\\treeV3\\{sigma_path}{str(LAYERS_CHECKED)}-layer\\{CUTOFFPOINT}_gens"
fileHelper.createDirIfNotExist(path)

if SHOW_GAMES:
    render_mode = "human"  # Set to None to run without graphics
else:
    render_mode = None

if MANUAL_ROLLING_AVERAGE == -1:
    window_width = round(CUTOFFPOINT/26.67)
else:
    window_width = MANUAL_ROLLING_AVERAGE


if SEMI_DETERMINISTIC > 1:
    seeds = list(range(SEMI_DETERMINISTIC))
else:
    seeds = [5]

if DETERMINISTIC:
    seeds = [1]



use_multiple_neighbors = False
if NEIGHBORS > 1:
    use_multiple_neighbors = True


def run_standard(show_results=True, save_results=True, neighbors=NEIGHBORS, layers_checked=LAYERS_CHECKED, local_path=path):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(0)
    np.random.seed(0)
    random.seed(0)

    current_seed = seeds[0]
    observation, info = env.reset(seed=0)
    steps_alive = 0
    actionstring = ""
    iterations = 0
    data = np.array([], dtype=int)

    tree = TreeV3(observation, num_nodes_checked=neighbors, layers_checked=layers_checked, gamma=GAMMA)
    action = tree.pick_action()
    actionstring += str(action)

    while iterations < CUTOFFPOINT:
        observation, reward, terminated, truncated, info = env.step(action)

        if not terminated:
            tree.update_result(observation, terminated)
            action = tree.pick_action()
            actionstring += str(action)

            steps_alive += 1

        if terminated or truncated:
            tree.finished_round(not truncated)

            if DETERMINISTIC:
                random.shuffle(seeds)
                current_seed = seeds[0]
                observation, info = env.reset(seed=current_seed)
            else:
                observation, info = env.reset()

            print(f"{iterations}: Steps alive: {steps_alive}, Nodes: {tree.get_num_nodes()}, String: {actionstring}")

            tree.start_round(observation)
            action = tree.pick_action()
            actionstring = str(action)


            data = np.append(data, [steps_alive])
            steps_alive = 0
            iterations += 1

    print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{iterations}")
    rolling_avg = plotHelper.rolling_average(data, window_width)


    avg = np.sum(data)/len(data)

    avg_plot = np.full((CUTOFFPOINT), 475)
    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{window_width}-step avg")
    plt.plot(avg_plot, label=f"CartPole-v1 threshold")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"{layers_checked}-layer {neighbors}-Neighbor {GAMMA}-gamma: v1 threshold")

    plot_name = local_path + f"\\{layers_checked}L-{neighbors}N-plot-v1.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()

    data = np.array([(200 if datum >200 else datum) for datum in data])

    rolling_avg = plotHelper.rolling_average(data, window_width)

    avg_plot = np.full((CUTOFFPOINT), 195)
    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{window_width}-step avg")
    plt.plot(avg_plot, label=f"CartPole-v0 threshold")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"{layers_checked}-layer {neighbors}-Neighbor {GAMMA}-gamma V0 threshold")

    plot_name = local_path + f"\\{layers_checked}L-{neighbors}N-plot-v0.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()

    return data

# shift file saving


def run_standard_iterated():
    layers_list = [1, 2, 3, 4]
    neighbors_list = [3, 5, 7, 10, 15, 20]
    for layers in layers_list:
        outer_start = time()
        for neighbors in neighbors_list:

            if GAMMA != 0.8:
                sigma_path = f"{str(GAMMA)}-sigma\\"
            else:
                sigma_path = ""
            path = f"plots\\treeV3\\{sigma_path}{str(layers)}-layer\\{CUTOFFPOINT}_gens"
            fileHelper.createDirIfNotExist(path)

            print("\n\n\n-----------------------------")
            print(f"Starting {layers=}, {neighbors=}")
            print("------------------------------\n\n\n")
            inner_start = time()
            run_standard(show_results=False, layers_checked=layers, neighbors=neighbors, local_path=path)
            inner_stop = time()
            print("----------------------------")
            print(f"Finished {layers=}, {neighbors=}, time= {round(inner_stop - inner_start)} seconds")
            print("------------------------------")

        outer_stop = time()
        print("\n#\n#\n#\n#\n#\n#")
        print(f"Finished {layers=}, time= {round((outer_stop - outer_start)/60)} minutes")
        print("\n#\n#\n#\n#\n#\n#")

if __name__ == '__main__':
    #run_standard_iterated()

    run_standard(show_results=False)

