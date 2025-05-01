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
#game_mode = "LunarLander-v2"

DISCOUNT_FACTOR = 0.99999  # Low discount penalize longer episodes.
# For example, shorter paths to the goal will receive higher reward than longer paths,
# even though the rewards from the environment are the same.
# Not necessarily applicable to every environment, such as cart pole, where the goal is to stay alive as long as possible,
# unless the reward is also exponential and thus cancels out the discount factor.

# Hyperparameters
GAUSSIAN_WIDTH = 0.3  # Sets the width of the Gaussian function that controls how much far away states should influence the action choice
EXPLORATION_RATE = 0.1  # Controls when actions with little data should be chosen, 0: never, 1: always

SEGMENTS = 5
K_MEANS_K = 20

STANDARD_RUNNING_LENGTH = 50
KMEANS_RUNNING_LENGTH = 100
KMEANS_TYPE = STANDARD
TSNE = False


path = f"mplots\\generic\\{GAME_MODE}\\single\\{GAUSSIAN_WIDTH}g"


OBSERVATION_LIMITS = np.asarray([[-2.4, 2.4],
                                 [-0.9, 0.9],
                                 [-0.2095, 0.2095],
                                 [-0.3, 0.3]])

def run_program(seed=SEED, discount_factor=DISCOUNT_FACTOR, gaussian_width=GAUSSIAN_WIDTH,
                exploration_rate=EXPLORATION_RATE, standard_episodes=STANDARD_RUNNING_LENGTH,
                kmeans_episodes=KMEANS_RUNNING_LENGTH, weighted_kmeans=True, render_mode=RENDER_MODE,
                game_mode=GAME_MODE, k=K_MEANS_K, save_plot=True, ignore_kmeans=False, use_vectors=False, learn=True,
                vector_type=1, do_standardize=True, use_special_kmeans=False, write_logs=True, segments=SEGMENTS,
                expander_gaussian=1, use_cosine_similarity=False):

    env = gymnasium.make(game_mode, render_mode=render_mode)
    env.action_space.seed(seed)
    np.random.seed(seed)

    model = GenericModel(env.action_space.n, env.observation_space.shape[0]*segments, gaussian_width, exploration_rate, K=k, weighted_kmeans=weighted_kmeans,
                         use_vectors=use_vectors, vector_type=vector_type, do_standardize=do_standardize,
                         use_special_kmeans=use_special_kmeans, use_cosine_similarity=use_cosine_similarity)

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
    while True:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                    env.close()
                    exit()

        if episodes < standard_episodes or ignore_kmeans:
            action = model.get_action_without_kmeans(expanded_state)
        elif use_vectors:
            action = model.get_action_with_vector(expanded_state)
        else:
            #print("REached")
            action = model.get_action_kmeans(expanded_state)
        action_string += str(action)


        actions.append(action)
        old_state = expanded_state

        state, reward, terminated, truncated, info = env.step(action)
        expanded_state = expander.expand(state)

        states.append(expanded_state)
        path.append(expanded_state)
        if use_vectors and episodes >= standard_episodes and False:
            model.check_vector(old_state, expanded_state)

        rewards += float(reward)

        if terminated or truncated:
            if write_logs:
                print(f"{episodes}: {rewards}  {action_string}")
            #print(f"{seed=}, {episodes=}, rewards: {rewards}")
            if episodes >= standard_episodes:
                data.append(rewards)
                if write_logs:
                    #print(f"{episodes}: {rewards}  {action_string}")
                    path = np.asarray(path)
                    if TSNE:
                        model.tsne_of_path(path)
            if learn or episodes < standard_episodes:
                for i, state in enumerate(states):
                    model.states = np.vstack((model.states, state))
                    model.rewards = np.hstack((model.rewards, np.power(discount_factor, len(states) - 1 - i) * rewards))
                    if i > 0:
                        model.state_action_transitions[actions[i - 1]].append((len(model.states) - 2, len(model.states) - 1))
                        model.state_action_transitions_from[actions[i - 1]].append(len(model.states) - 2)
                        model.state_action_transitions_to[actions[i - 1]].append(len(model.states) - 1)

            rewards = 0.
            action_string = ""
            path = []
            actions.clear()
            states.clear()
            state, info = env.reset()
            expanded_state = expander.expand(state)

            states.append(expanded_state)
            episodes += 1
            if episodes == standard_episodes and not ignore_kmeans:
                if write_logs:
                    print("Calculating kmeans centers...")
                model.calc_standard_kmeans(write_logs=write_logs, run_tsne=TSNE)
                if TSNE:
                    model.tsne_based_on_reward()
                if write_logs:
                    print(f"{model.states.shape=}")
                    print("Finished calculating kmeans centers")
            if episodes == kmeans_episodes + standard_episodes:
                break

    if save_plot:
        window_width = 5
        rolling_avg = plotHelper.rolling_average(data, window_width)

        avg = np.sum(data)/len(data)
        #avg_plot = np.full((KMEANS_RUNNING_LENGTH), 195)
        plt.plot(data, label="Steps")
        plt.plot(rolling_avg, label=f"{window_width}-step avg")
        #plt.plot(avg_plot, label=f"CartPole-v0 threshold")

        plt.xlabel("Iterations")
        plt.ylabel("Steps")
        plt.legend(loc="upper left")
        plt.title(f"{20}K-{gaussian_width}G avg:{avg} V0 threshold")

        plot_name = path + f"\\{STANDARD_RUNNING_LENGTH}_then_{KMEANS_RUNNING_LENGTH}_plot.png"

        plt.savefig(plot_name)
        plt.clf()
    if use_special_kmeans:
        print(f"Finished seed {seed}")
    return data


def run_program_with_different_seeds(seed_count=3, discount_factor=DISCOUNT_FACTOR, gaussian_width=GAUSSIAN_WIDTH,
                exploration_rate=EXPLORATION_RATE, standard_episodes=STANDARD_RUNNING_LENGTH,
                kmeans_episodes=KMEANS_RUNNING_LENGTH, kmeans_type=KMEANS_TYPE, render_mode=RENDER_MODE,
                game_mode=GAME_MODE, save_plot=True):
    datas = []
    for seed in range(seed_count):
        data = run_program(seed=seed, discount_factor=discount_factor, gaussian_width=gaussian_width,
                exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                kmeans_episodes=kmeans_episodes, kmeans_type=kmeans_type, render_mode=render_mode,
                game_mode=game_mode, save_plot=False)
        datas.append(data)
    datas = np.asarray(datas)
    bucket_data = plotHelper.average_every_n(datas, list_of_list=True, n=5)
    avg_reward = plotHelper.average_of_diff_seeds(bucket_data)
    error_bounds = plotHelper.get_upper_lower_error_bounds(bucket_data, avg_reward)
    x = np.arange(0, len(data), 5)

    #avg_reward = np.ones_like(avg_reward)*50
    #error_bounds = [avg_reward - avg_reward*0.5, avg_reward*2 - avg_reward]

    print(f"{avg_reward}")
    print(f"{error_bounds}")

    plt.errorbar(x, avg_reward, yerr=error_bounds, fmt='-o')
    plt.title(f"{game_mode} {SEED_COUNT}-count avg plot")
    plt.show()



if __name__ == '__main__':
    fileHelper.createDirIfNotExist(path)
    #run_program_with_different_seeds(seed_count=SEED_COUNT)
    run_program(3)