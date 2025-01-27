import numpy as np
from helper.K_MeansTypes import *
from sklearn.cluster import KMeans
from sklearn import preprocessing



class GenericModel:
    def __init__(self, env, gaussian_width, exploration_rate, kmeans_type=STANDARD, split_kmeans=False, K=20, no_learning=False):
        self.gaussian_width = gaussian_width
        self.action_space_size = env.action_space.n
        self.exploration_rate = exploration_rate
        self.kmeans_type = kmeans_type
        self.split_kmeans = split_kmeans
        self.K = K
        self.no_learning = no_learning

        self.states: np.ndarray = np.empty((0, env.observation_space.shape[0]))  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index


        self.actions: list[int] = list(range(self.action_space_size))

        # A list for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(self.action_space_size)]  # type: ignore


        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]


        self.scaler = None
        self.standardized_states = []
        self.kmeans_centers: list[np.ndarray] = []
        self.kmeans_action_reward_list: list[list[float]] = []
        self.kmeans_action_weight_list: list[list[float]] = []

        self.split_kmeans_clusters: list[list[np.ndarray]] = [[] for _ in range(self.action_space_size)]
        self.split_kmeans_clusters: list[list[float]] = [[] for _ in range(self.action_space_size)]


    def get_action_without_kmeans(self, state):
        states_mean = np.array([0.])  # Used to normalize the state space
        states_std = np.array([1.])  # Used to normalize the state space
        if len(self.states) > 0:
            states_mean = np.mean(self.states, axis=0)
            states_std = np.std(self.states, axis=0)
        for i, _ in enumerate(states_std):
            if states_std[i] == 0.:
                states_std[i] = 1.

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]

        for action in self.actions:
            if len(self.state_action_transitions_from[action]) > 0:
                dist = (state - states_mean) / states_std - (self.states[self.state_action_transitions_from[action]] - states_mean) / states_std
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]

        for action, _ in enumerate(self.state_action_transitions):
            if weight_sums[action] == 0:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate:
                return action  # Return action that has little data for the current state
        return np.argmax(action_rewards)

    def calc_standard_kmeans(self):
        self.scaler = preprocessing.StandardScaler().fit(self.states)
        self.standardized_states = self.scaler.transform(self.states)
        self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states).cluster_centers_
        self.kmeans_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(self.kmeans_centers)


    def get_action_kmeans(self, state):
        standardized_state = self.scaler.transform([state])

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]


        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.kmeans_action_reward_list[action]) / weight_sums[action]
                #action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]

        if self.no_learning:
            return np.argmax(action_rewards)

        for action, _ in enumerate(self.state_action_transitions):
            if weight_sums[action] == 0:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate:
                return action  # Return action that has little data for the current state
        return np.argmax(action_rewards)

    def calc_kmeans_center_rewards(self, centers):
        action_rewards_list = []
        weight_sums_list = []
        for kmean in centers:
            action_rewards = [0. for _ in range(self.action_space_size)]
            weight_sums = [0. for _ in range(self.action_space_size)]
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = kmean - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                    weight_sums[action] = np.sum(weight)
                    action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]
            action_rewards_list.append(action_rewards)
            weight_sums_list.append(weight_sums)

        return np.asarray(action_rewards_list).transpose(), weight_sums_list