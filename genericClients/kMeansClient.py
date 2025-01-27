import gymnasium
import pygame
import numpy as np

from classes.genericBasic import *

render_mode = None  # Set to None to run without graphics
game_mode = "CartPole-v1"
#game_mode = "LunarLander-v2"
env = gymnasium.make(game_mode, render_mode=render_mode)
seed = 0
env.action_space.seed(seed)
np.random.seed(seed)
discount_factor = 0.99999  # Low discount penalize longer episodes.
# For example, shorter paths to the goal will receive higher reward than longer paths,
# even though the rewards from the environment are the same.
# Not necessarily applicable to every environment, such as cart pole, where the goal is to stay alive as long as possible,
# unless the reward is also exponential and thus cancels out the discount factor.

# Hyperparameters
gaussian_width = 0.3  # Sets the width of the Gaussian function that controls how much far away states should influence the action choice
exploration_rate = 0.1  # Controls when actions with little data should be chosen, 0: never, 1: always


STANDARD_RUNNING_LENGTH = 15
KMEANS_RUNNING_LENGTH = 50


class Model:
    def __init__(self):
        self.states: list[np.ndarray] = []  # States are stored here
        self.rewards: list[float] = []  # Value for each state index

        # A list for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(env.action_space.n)]  # type: ignore


model = GenericModel(env, gaussian_width, exploration_rate)

rewards = 0.  # Accumulative episode rewards
actions = []  # Episode actions
states = []  # Episode states
state, info = env.reset(seed=seed)
states.append(state)
episodes = 0

while True:
    if render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                env.close()
                exit()

    if episodes <= STANDARD_RUNNING_LENGTH:
        action = model.get_action_without_kmeans(state)
    else:
        action = model.get_action_kmeans(state)

    actions.append(action)
    state, reward, terminated, truncated, info = env.step(action)
    states.append(state)

    rewards += float(reward)

    if terminated or truncated:
        print(f"rewards: {rewards}")
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
        if episodes == STANDARD_RUNNING_LENGTH:
            print("Calculating kmeans centers...")
            model.calc_standard_kmeans()
            print("Finished calculating kmeans centers")
        if episodes > KMEANS_RUNNING_LENGTH:
            break
