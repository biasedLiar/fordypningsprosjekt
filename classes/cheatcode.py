import gymnasium
import pygame
import numpy as np

render_mode = "human"  # Set to None to run without graphics

env = gymnasium.make("CartPole-v1", render_mode=render_mode)
seed = 0
env.action_space.seed(seed)
np.random.seed(seed)

# Hyperparameters
gaussian_width = 0.3  # Sets the width of the Gaussian function that controls how much far away states should influence the action choice
exploration_rate = 0.1  # Controls when actions with little data should be chosen, 0: never, 1: always


class Model:
    def __init__(self):
        self.states: list[np.ndarray] = []  # States are stored here
        self.rewards: list[float] = []  # Value for each state index

        # A list for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(env.action_space.n)]  # type: ignore


model = Model()

rewards = 0.  # Accumulative episode rewards
actions = []  # Episode actions
states = []  # Episode states
state, info = env.reset(seed=seed)
states.append(state)

while True:
    if render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                env.close()
                exit()

    states_mean = np.array([0.])  # Used to normalize the state space
    states_std = np.array([1.])  # Used to normalize the state space
    if len(model.states) > 0:
        states_mean = np.mean(model.states, axis=0)
        states_std = np.std(model.states, axis=0)
    for i, _ in enumerate(states_std):
        if states_std[i] == 0.:
            states_std[i] = 1.

    action_rewards = [0. for _ in model.state_action_transitions]
    weight_sums = [0. for _ in model.state_action_transitions]
    for i, _ in enumerate(model.state_action_transitions):
        for state_from, state_to in model.state_action_transitions[i]:
            dist = (state - states_mean) / states_std - (model.states[state_from] - states_mean) / states_std
            weight = np.exp(-np.sum(np.square(dist)) / gaussian_width)
            weight_sums[i] += weight
            action_rewards[i] += weight * model.rewards[state_to]
        if weight_sums[i] > 0.:
            action_rewards[i] /= weight_sums[i]

    def get_action():
        for i, _ in enumerate(model.state_action_transitions):
            if weight_sums[i] == 0:
                return i  # Return action that has never been chosen before
            if weight_sums[i] / np.max(weight_sums) < exploration_rate:
                return i  # Return action that has little data for the current state
        return np.argmax(action_rewards)

    action = get_action()

    actions.append(action)
    state, reward, terminated, truncated, info = env.step(action)
    states.append(state)

    rewards += float(reward)

    if terminated or truncated:
        print(f"rewards: {rewards}")
        for i, state in enumerate(states):
            model.states.append(state)
            model.rewards.append(rewards)
            if i > 0:
                model.state_action_transitions[actions[i - 1]].append((len(model.states) - 2, len(model.states) - 1))

        rewards = 0.
        actions.clear()
        states.clear()
        state, info = env.reset()
        states.append(state)