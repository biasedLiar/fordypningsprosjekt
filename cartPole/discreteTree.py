import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import helper.plotHelper as plotHelper
import helper.KNearestNeighbor as kNearest
from helper.strategy_names import *
import helper.fileHelper as fileHelper
from classes.cartPoleTreeNode import *


############### Constants ###################
CUTOFF = True
CUTOFFPOINT = 6000

STOP_AFTER_CONSEC_500S = False

ANGLE = 0.2095
# ANGLE = 1.0

SHOW_GAMES = False

# Set to -1 for automatic rolling average generation.
MANUAL_ROLLING_AVERAGE = -1

K_START = 1
K_END = 1
K_STEP = 1

STEPS_PER_NODE = 3
DETERMINISTIC = False
START_STRATEGY = BALANCED
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

def run_k_nearest(k=-1, show_results=True, save_results=True, window_width=5):
    if k > 0:
        kNearest.K = k
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(0)
    np.random.seed(0)
    random.seed(0)

    observation, info = env.reset(seed=0)

    steps_alive = 0
    terminated_observations = np.zeros((0, 4))
    terminated_observations_normalized = np.zeros((0, 4))
    mean = np.zeros((4))
    std = np.ones((4))
    action = 0
    last_it_succeeded = False
    its_before_finished = 0

    all_nodes = {}
    data = np.array([], dtype=int)
    root_node = CartPoleTreeNode(observation, 0, START_STRATEGY, "-")
    visited_nodes = []
    current_node = root_node

    #TODO update this file so we can make it run

    while its_before_finished < CUTOFFPOINT:
        if steps_alive%STEPS_PER_NODE == 0:
            action = current_node.pick_action()

        observation, reward, terminated, truncated, info = env.step(action)

        if steps_alive%STEPS_PER_NODE == 0:
            visited_nodes.append(current_node)
            current_node = current_node.register_move(observation, steps_alive, action)


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

            current_node = root_node
            root_node.set_root_strategy(START_STRATEGY)
            visited_nodes = []
            if DETERMINISTIC:
                observation, info = env.reset(seed=0)
            else:
                observation, info = env.reset()

            print(f"{its_before_finished}: Steps alive: {steps_alive}")
            #print(root_node.visualize_tree())
            #print(root_node.show_selected_path())
            data = np.append(data, [steps_alive])
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

def is_game_over(observation):
    game_over = np.abs(observation[2]) > ANGLE or np.abs(observation[0]) > 2.4
    if game_over:
        print("hmmm")
    return game_over


def run_and_compare_range_k_nearest(bottom, top, window_width, step=1):
    survival_stats = []
    print("Update plots...")
    input()
    input()
    input()
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



# shift file saving

if __name__ == '__main__':
    run_k_nearest(show_results=False)
    #run_and_compare_range_k_nearest(K_START, K_END, window_width, K_STEP)
    print("Finished with new")