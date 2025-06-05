import gymnasium
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt

from classes.genericBasic import *
from classes.stateExpander import *
import helper.fileHelper as fileHelper
import helper.plotHelper as plotHelper

SEED = 1
SEED_COUNT = 30
RENDER_MODE = None  # Set to None to run without graphics
GAME_MODE = "CartPole-v1"

DISCOUNT_FACTOR = 0.9  # Low discount penalize longer episodes.
# For example, shorter paths to the goal will receive higher reward than longer paths,
# even though the rewards from the environment are the same.
# Not necessarily applicable to every environment, such as cart pole, where the goal is to stay alive as long as possible,
# unless the reward is also exponential and thus cancels out the discount factor.

# Hyperparameters
GAUSSIAN_WIDTH = 0.3  # Sets the width of the Gaussian function that controls how much far away states should influence the action choice
EXPLORATION_RATE = 0.1  # Controls when actions with little data should be chosen, 0: never, 1: always

SEGMENTS = 5
K_MEANS_K = 20

LEARNING_LENGTH = 100
SLEEPING_LENGTH = 100
TSNE = False


path = f"mplots\\generic\\{GAME_MODE}\\single\\{GAUSSIAN_WIDTH}g"


OBSERVATION_LIMITS = np.asarray([[-2.4, 2.4],
                                 [-0.9, 0.9],
                                 [-0.2095, 0.2095],
                                 [-0.3, 0.3]])

def run_program(seed=SEED, discount_factor=DISCOUNT_FACTOR, gaussian_width=GAUSSIAN_WIDTH,
                exploration_rate=EXPLORATION_RATE, standard_episodes=LEARNING_LENGTH,
                kmeans_episodes=SLEEPING_LENGTH, weighted_kmeans=True, render_mode=RENDER_MODE,
                game_mode=GAME_MODE, k=K_MEANS_K, ignore_kmeans=False, learn=True, do_standardize=True, write_logs=True, segments=SEGMENTS,
                expander_gaussian=1, use_search_tree=False,
                search_tree_depth=-1, save_midway=False, weighted_sigmoid=False):

    env = gymnasium.make(game_mode, render_mode=render_mode)
    env.action_space.seed(seed)
    np.random.seed(seed)

    model = GenericModel(env.action_space.n, env.observation_space.shape[0]*segments, gaussian_width, exploration_rate, K=k, weighted_kmeans=weighted_kmeans,
                         do_standardize=do_standardize, use_search_tree=use_search_tree, search_tree_depth=search_tree_depth, weighted_sigmoid=weighted_sigmoid)

    rewards = 0.  # Accumulative episode rewards
    actions = []  # Episode actions
    states = []  # Episode states
    expander = StateExpander(env.observation_space.shape[0], OBSERVATION_LIMITS, segments=segments, gaussian_width=expander_gaussian)

    state, info = env.reset(seed=seed)
    expanded_state = expander.expand(state)
    states.append(expanded_state)

    episodes = 0
    data = []
    action_string = ""
    path = []
    reward_list = [0.0]
    end_kmeans = time.time()
    start_kmeans = end_kmeans
    start_post_kmeans = end_kmeans
    end_post_kmeans = end_kmeans

    total_sleeping_steps = 0

    while True:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                    env.close()
                    exit()




        if episodes < standard_episodes or (ignore_kmeans and not use_search_tree):
            action = model.get_action_without_kmeans(expanded_state)
        elif use_search_tree:
            action = model.get_action_search_tree(expanded_state, ignore_kmeans=ignore_kmeans)
        else:
            action = model.get_action_kmeans(expanded_state)





        action_string += str(action)


        if episodes >= standard_episodes:
            total_sleeping_steps += 1

        actions.append(action)
        old_state = expanded_state

        state, reward, terminated, truncated, info = env.step(action)
        expanded_state = expander.expand(state)

        states.append(expanded_state)
        path.append(expanded_state)

        rewards += float(reward)
        reward_list.append(float(reward))

        if terminated or truncated:
            if write_logs:
                print(f"{episodes}: {rewards}  {action_string}")
            #print(f"{seed=}, {episodes=}, rewards: {rewards}")
            if episodes >= standard_episodes:
                data.append(rewards)
            if learn or episodes < standard_episodes:
                reward_new = plotHelper.get_rewards(reward_list, DISCOUNT_FACTOR)
                for i, state in enumerate(states):
                    model.states = np.vstack((model.states, state))
                    model.rewards = np.hstack((model.rewards, reward_new[i]))
                    if i > 0:
                        model.state_action_transitions[actions[i - 1]].append((len(model.states) - 2, len(model.states) - 1))
                        model.state_action_transitions_from[actions[i - 1]].append(len(model.states) - 2)
                        model.state_action_transitions_to[actions[i - 1]].append(len(model.states) - 1)

            rewards = 0.
            reward_list = [0]
            action_string = ""
            path = []
            actions.clear()
            states.clear()
            state, info = env.reset()
            expanded_state = expander.expand(state)

            states.append(expanded_state)
            episodes += 1
            if episodes == standard_episodes:
                start_kmeans = time.time()
                model.midway = True
                if save_midway:
                    model.calc_search_tree_state_vectors(ignore_kmeans=ignore_kmeans)
                if not ignore_kmeans:
                    if use_search_tree:
                        model.calc_search_tree_kmeans(write_logs=write_logs, run_tsne=TSNE)
                    else:
                        model.calc_standard_kmeans(write_logs=write_logs, run_tsne=TSNE)
                end_kmeans = time.time()
                start_post_kmeans = time.time()

            if episodes == kmeans_episodes + standard_episodes:
                end_post_kmeans = time.time()
                break
    print(f"Finished seed {seed}")

    kmeans_time = end_kmeans - start_kmeans
    post_kmeans_time = end_post_kmeans - start_post_kmeans
    return (data, kmeans_time, post_kmeans_time, total_sleeping_steps)



if __name__ == '__main__':
    fileHelper.createDirIfNotExist(path)
    #run_program_with_different_seeds(seed_count=SEED_COUNT)
    run_program(3)