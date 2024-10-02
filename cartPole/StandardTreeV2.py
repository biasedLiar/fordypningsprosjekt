import random

import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import helper.plotHelper as plotHelper
import helper.KNearestNeighbor as kNearest
from helper.strategy_names import *
import helper.fileHelper as fileHelper
from classes.TreeV2 import *


############### Constants ###################
CUTOFF = True
CUTOFFPOINT = 6000

STOP_AFTER_CONSEC_500S = False

ANGLE = 0.2095

SHOW_GAMES = False
STRATEGY = BALANCED

# Set to -1 for automatic rolling average generation.
MANUAL_ROLLING_AVERAGE = -1

K_START = 1
K_END = 1
K_STEP = 1

STEPS_PER_NODE = 5
DETERMINISTIC = True
SEMI_DETERMINISTIC = 4
START_STRATEGY = EXPLORE
########### End constants #################

path = f"plots\\standard_tree\\{'deterministic' if DETERMINISTIC else 'non-deterministic'}\\{STEPS_PER_NODE}-discrete\\{CUTOFFPOINT}_gens"
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
    seeds= list(range(SEMI_DETERMINISTIC))
else:
    seeds = [5]

if DETERMINISTIC:
    seeds = [1]
def run_k_nearest(k=-1, show_results=True, save_results=True):
    if k > 0:
        kNearest.K = k
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(0)
    np.random.seed(0)
    random.seed(0)

    current_seed = seeds[0]
    observation, info = env.reset(seed=current_seed)
    steps_alive = 0
    action = 0
    its_before_finished = 0
    data = np.array([], dtype=int)
    current_node = DCartPoleTreeNode(observation, 0, START_STRATEGY, "-")

    while its_before_finished < CUTOFFPOINT:

        observation, reward, terminated, truncated, info = env.step(action)

        if steps_alive%STEPS_PER_NODE == 0:
            print("test1")

        steps_alive += 1

        if (np.abs(observation[2]) > ANGLE or np.abs(observation[0]) > 2.4 or truncated):
            if not truncated:
                current_node.mark_final()
                for node in reversed(visited_nodes):
                    node.update()
                last_it_succeeded = False
            else:
                if last_it_succeeded and STOP_AFTER_CONSEC_500S:
                    print(f"Steps alive: {steps_alive}")
                    break
                else:
                    last_it_succeeded = True



            visited_nodes = []
            visited_observations = []


            if DETERMINISTIC:
                random.shuffle(seeds)
                current_seed = seeds[0]
                observation, info = env.reset(seed=current_seed)
            else:
                observation, info = env.reset()

            print(f"{its_before_finished}: Steps alive: {steps_alive}")

            steps_alive = 0
            its_before_finished += 1

    print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{its_before_finished}")
    rolling_avg = plotHelper.rolling_average(data, window_width)

    plt.plot(data, label="Steps")
    plt.plot(rolling_avg, label=f"{window_width}-step avg")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"{'Deterministic' if DETERMINISTIC else 'Non-deterministic'} {STEPS_PER_NODE}-discrete {name_of_strategy(START_STRATEGY)}-search")

    plot_name = path + f"\\{name_of_strategy(START_STRATEGY)}_plot.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()
    return data

# shift file saving

if __name__ == '__main__':
    run_k_nearest(show_results=False)
    #run_and_compare_range_k_nearest(K_START, K_END, window_width, K_STEP)
    print("Finished with new")