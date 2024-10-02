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
from classes.discreteCartPoleTreeNode import *


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
    visited_observations = []
    all_observations = np.zeros((0, 4))
    obs_mapped = []
    observations_normalized = np.zeros((0, 4))
    mean = np.zeros((4))
    std = np.ones((4))
    action = 0
    last_it_succeeded = False
    its_before_finished = 0

    all_nodes = {}
    data = np.array([], dtype=int)
    root_node = DCartPoleTreeNode(observation, 0, START_STRATEGY, "-")
    visited_nodes = []
    current_node = root_node
    all_nodes[current_node.get_state_bucket()] = current_node
    action = current_node.pick_action()

    root_list = {}
    root_list[current_seed] = root_node

    while its_before_finished < CUTOFFPOINT:

        observation, reward, terminated, truncated, info = env.step(action)

        if steps_alive%STEPS_PER_NODE == 0:
            visited_nodes.append(current_node)
            visited_observations.append(observation)
            new_node_bucket = current_node.calc_state_bucket(observation)
            action = current_node.pick_action()

            if not new_node_bucket in all_nodes.keys():
                current_node = current_node.register_move(observation, steps_alive, action)
                all_nodes[current_node.get_state_bucket()] = current_node
            else:
                current_node = current_node.register_existing(current_node, steps_alive, action)
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



            if current_seed == 0 and its_before_finished > 2000:
                test = current_seed #TODO why is bad stuff happening

            visited_nodes = []
            visited_observations = []


            if DETERMINISTIC:
                random.shuffle(seeds)
                current_seed = seeds[0]
                observation, info = env.reset(seed=current_seed)
            else:
                observation, info = env.reset()

            new_node_bucket = current_node.calc_state_bucket(observation)
            #To normalize or not to normalize
            if not current_seed in root_list.keys():
                current_node = DCartPoleTreeNode(observation, 0, START_STRATEGY, "-")
                all_nodes[new_node_bucket] = current_node
                root_list[current_seed] = current_node
                if not current_seed in root_list:
                    test =  all_nodes[new_node_bucket]
                    test2 = 1
            else:
                current_node = all_nodes[new_node_bucket]
                if new_node_bucket != root_list[current_seed].get_state_bucket():
                    print("States not matching...")
                    input()

            current_node.set_root_strategy(START_STRATEGY)



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



def add_if_not_present(observation, obs_mapped, all_observations):
    if (str(observation) not in obs_mapped):
        obs_mapped.append(str(observation))
        all_observations = np.concatenate((all_observations, [observation]), axis=0)
    return obs_mapped, all_observations

def normalize_all_observations(observations):
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std = np.where(std != 0., std, 1.)  # To avoid division by zero
    normalized_observations = (observations - mean) / std
    return normalized_observations, mean, std

def normalize_one_observations(observation, mean, std):
    return (observation - mean) / std

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