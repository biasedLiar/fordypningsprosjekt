import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import helper.plotHelper as plotHelper
import helper.KNearestNeighbor as kNearest
import helper.fileHelper as fileHelper


############### Constants ###################
CUTOFF = True
CUTOFFPOINT = 200

STOP_AFTER_CONSEC_500S = False

ANGLE = 0.2095
# ANGLE = 1.0

SHOW_GAMES = False

MANUAL_ROLLING_AVERAGE = -1

K_START = 2
K_END = 6
K_STEP = 1

SHORT_TERM_MEM_LENGTH = 7
########### End constants #################

path = f"plots\\kNearestMem\\{ANGLE}_angle\\{SHORT_TERM_MEM_LENGTH}_mem_len\\{CUTOFFPOINT}_gens"
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
    short_term_memory = np.zeros((0, 4))


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

        short_term_memory = update_short_term_memory(short_term_memory, observation)
        steps_alive += 1

        if np.abs(observation[2]) > ANGLE or np.abs(observation[0]) > 2.4 or truncated:
            if not truncated:
                terminated_observations = add_short_term_mem_to_observations(terminated_observations, short_term_memory)
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
    plt.title(f"K-Nearest Neighbor Classification, short-term-mem={SHORT_TERM_MEM_LENGTH}, angle={ANGLE}")

    plot_name = path + f"\\K{kNearest.K}__plot.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    plt.clf()
    return data


def update_short_term_memory(short_term_mem, datapoint):
    if len(short_term_mem) == SHORT_TERM_MEM_LENGTH:
        short_term_mem = np.concatenate((short_term_mem[1:], [datapoint]), axis=0)
    else:
        short_term_mem = np.concatenate((short_term_mem, [datapoint]), axis=0)
    return short_term_mem


def add_short_term_mem_to_observations(terminated_observations, short_term_memory):
    return np.concatenate((terminated_observations, short_term_memory))




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
    plt.title(f"K-Nearest Neighbor Classification, short-term-mem={SHORT_TERM_MEM_LENGTH}, angle={ANGLE}")
    plot_name = path +  f"\\K{bottom}-K{top}__group__plot.png"
    plt.savefig(plot_name)
    plt.clf()

    for i in range(len(survival_stats)):
        rolling_avg = plotHelper.rolling_average(survival_stats[i], window_width)
        plt.plot(rolling_avg, label=f"k={step*i + bottom}")

    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="upper left")
    plt.title(f"K-Nearest Neighbor, Rolling avg of {window_width}, short-term-mem={SHORT_TERM_MEM_LENGTH}, angle={ANGLE}")
    plot_name = path + f"\\K{bottom}-K{top}__group__rolling-avg-{window_width}__plot.png"
    plt.savefig(plot_name)
    plt.clf()



# shift file saving

if __name__ == '__main__':
    # run_k_nearest()
    run_and_compare_range_k_nearest(K_START, K_END, window_width, K_STEP)
    print("Finished")