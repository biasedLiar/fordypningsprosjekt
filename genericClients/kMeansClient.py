import gymnasium
import pygame
import numpy as np
import matplotlib.pyplot as plt

from classes.genericBasic import *
import helper.fileHelper as fileHelper
import helper.plotHelper as plotHelper

SEED = 1
SEED_COUNT = 10
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

K_MEANS_K = 20

STANDARD_RUNNING_LENGTH = 50
KMEANS_RUNNING_LENGTH = 100
KMEANS_TYPE = STANDARD


path = f"mplots\\generic\\{GAME_MODE}\\{KMEANS_TYPE}-kmeans\\{GAUSSIAN_WIDTH}g"



def run_program(seed=SEED, discount_factor=DISCOUNT_FACTOR, gaussian_width=GAUSSIAN_WIDTH,
                exploration_rate=EXPLORATION_RATE, standard_episodes=STANDARD_RUNNING_LENGTH,
                kmeans_episodes=KMEANS_RUNNING_LENGTH, kmeans_type=KMEANS_TYPE, render_mode=RENDER_MODE,
                game_mode=GAME_MODE, k=K_MEANS_K, save_plot=True):

    env = gymnasium.make(game_mode, render_mode=render_mode)
    env.action_space.seed(seed)
    np.random.seed(seed)

    model = GenericModel(env, gaussian_width, exploration_rate, K=k)

    rewards = 0.  # Accumulative episode rewards
    actions = []  # Episode actions
    states = []  # Episode states
    state, info = env.reset(seed=seed)
    states.append(state)
    episodes = 0
    data = []
    while True:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                    env.close()
                    exit()

        if episodes < standard_episodes:
            action = model.get_action_without_kmeans(state)
        else:
            #print("REached")
            action = model.get_action_kmeans(state)

        actions.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        states.append(state)

        rewards += float(reward)

        if terminated or truncated:
            print(f"rewards: {rewards}")
            if episodes >= standard_episodes:
                data.append(rewards)
            for i, state in enumerate(states):
                model.states = np.vstack((model.states, state))
                model.rewards = np.hstack((model.rewards, np.power(discount_factor, len(states) - 1 - i) * rewards))
                if i > 0:
                    model.state_action_transitions[actions[i - 1]].append((len(model.states) - 2, len(model.states) - 1))
                    model.state_action_transitions_from[actions[i - 1]].append(len(model.states) - 2)
                    model.state_action_transitions_to[actions[i - 1]].append(len(model.states) - 1)

            rewards = 0.
            actions.clear()
            states.clear()
            state, info = env.reset()
            states.append(state)
            episodes += 1
            if episodes == standard_episodes:
                print("Calculating kmeans centers...")
                model.calc_standard_kmeans()
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