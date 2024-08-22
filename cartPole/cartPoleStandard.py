import gymnasium as gym
import pygame
import numpy as np

ANGLE = 0.2095

render_mode = "human"  # Set to None to run without graphics

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
while True:
    last_closest_distance = np.min(np.sum(np.square((observation - mean) / std -
                                                    terminated_observations_normalized), axis=1)) if terminated_observations_normalized.shape[0] > 0 else 0.0
    observation, reward, terminated, truncated, info = env.step(action)

    closest_distance = np.min(np.sum(np.square((observation - mean) / std -
                                               terminated_observations_normalized), axis=1)) if terminated_observations.shape[0] > 0 else 0.0

    closest_distance_change = closest_distance - last_closest_distance

    if closest_distance_change >= 0.0 or (last_closest_distance_change != None and closest_distance_change > last_closest_distance_change):
        pass
    else:
        action = (action + 1) % 2

    last_closest_distance_change = closest_distance_change

    steps_alive += 1

    if np.abs(observation[2]) > 0.2095 or np.abs(observation[0]) > 2.4 or truncated:
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

        print(f"{its_before_finished=}, Steps alive: {steps_alive}")
        steps_alive = 0
        last_closest_distance_change = None
        observation, info = env.reset()
        its_before_finished += 1

print(f"\n\nFinished. Iterations before two successful iterations in a row:\n{its_before_finished}")
#Basic version got two in a row after 40 iterations