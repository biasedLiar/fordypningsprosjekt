import gymnasium as gym
import pygame
import numpy as np
import helper.KNearestNeighbor as kNearest
import matplotlib.pyplot as plt

CUTOFF = True
CUTOFFPOINT = 200


# ANGLE = 0.2095
ANGLE = 1.0
render_mode = "human"  # Set to None to run without graphics

def run_k_nearest(k=-1, show_results=True, save_results=True):
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

    training_survival_stats = []

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

        if np.abs(observation[2]) > 1.0 or np.abs(observation[0]) > 2.4 or truncated:
            if not truncated:
                terminated_observations = np.concatenate((terminated_observations, [observation]), axis=0)
                mean = terminated_observations.mean(axis=0)
                std = terminated_observations.std(axis=0)
                std = np.where(std != 0., std, 1.)  # To avoid division by zero
                terminated_observations_normalized = (terminated_observations - mean) / std
                last_it_succeeded = False
            else:
                print("truncated")
                if last_it_succeeded:
                    print(f"Steps alive: {steps_alive}")
                    break
                else:
                    last_it_succeeded = True

            print(f"{its_before_finished}: Steps alive: {steps_alive}")
            training_survival_stats.append(steps_alive)
            steps_alive = 0
            last_closest_distance_change = None
            observation, info = env.reset()
            its_before_finished += 1

    print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{its_before_finished}")
    plt.plot(training_survival_stats)
    plt.xlabel("iterations")
    plt.ylabel("steps")
    plt.title(f"k nearest neighbor classification, k={kNearest.K}, angle={ANGLE}")

    plot_name = f"plots\\kNearest\\K_{kNearest.K}--angle_{ANGLE}--plot.png"
    if save_results:
        plt.savefig(plot_name)
    if show_results:
        plt.show()
    return training_survival_stats


def run_and_compare_range_k_nearest(bottom, top, step=1):
    survival_stats = []
    for i in range(bottom, top+1, step):
        survival_stats.append(run_k_nearest(k=i, show_results=False, save_results=True))




if __name__ == '__main__':
    run_k_nearest()
    print("Finished")