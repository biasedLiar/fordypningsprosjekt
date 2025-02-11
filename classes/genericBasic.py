import numpy as np
from helper.K_MeansTypes import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from matplotlib.ticker import NullFormatter



class GenericModel:
    def __init__(self, env, gaussian_width, exploration_rate, weighted_kmeans=True,
                 use_vectors=False, split_kmeans=False, K=20, no_learning=True, use_kmeans=True, vector_type=1):
        self.gaussian_width = gaussian_width
        self.action_space_size = env.action_space.n
        self.observation_space_size = env.observation_space.shape[0]
        self.exploration_rate = exploration_rate
        self.weighted_kmeans = weighted_kmeans
        self.split_kmeans = split_kmeans
        self.use_vectors = use_vectors
        self.vector_type = vector_type
        self.K = K
        self.no_learning = no_learning
        self.delta = 10**-8
        self.use_kmeans = use_kmeans

        self.states: np.ndarray = np.empty((0, env.observation_space.shape[0]))  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index


        self.actions: list[int] = list(range(self.action_space_size))

        # A list for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(self.action_space_size)]  # type: ignore


        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]

        self.num_states_when_ran_kmeans = -1
        self.scaler = None
        self.standardized_states = []
        self.kmeans_centers: list[np.ndarray] = []
        self.kmeans_action_reward_list: list[list[float]] = []
        self.kmeans_action_weight_list: list[list[float]] = []
        self.center_vectors = []
        self.center_max_rewards = []

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
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]

        for action, _ in enumerate(self.state_action_transitions):
            if weight_sums[action] == 0:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate:
                return action  # Return action that has little data for the current state
        return np.argmax(action_rewards)

    def scatterplot2d(self):
        X = np.concatenate((self.standardized_states[:, 0], self.kmeans_centers[:, 0]))
        Y = np.concatenate((self.standardized_states[:, 1], self.kmeans_centers[:, 1]))
        colors = np.concatenate(
            (np.ones_like(self.standardized_states[:, 0]), np.ones_like(self.kmeans_centers[:, 0]) * 2))
        plt.scatter(X, Y, c=colors, alpha=0.5)
        plt.show()
        plt.clf()
        X = np.concatenate((self.standardized_states[:, 2], self.kmeans_centers[:, 2]))
        Y = np.concatenate((self.standardized_states[:, 3], self.kmeans_centers[:, 3]))
        colors = np.concatenate(
            (np.ones_like(self.standardized_states[:, 0]), np.ones_like(self.kmeans_centers[:, 0]) * 2))
        plt.scatter(X, Y, c=colors, alpha=0.5)
        plt.show()
        plt.clf()
        input()

    def tsne(self, after_kmeans=False):
        print("Starting TSNE...")
        self.standardized_states = self.scaler.transform(self.states)
        state_length = len(self.standardized_states)
        perplexity = 20
        X = np.concatenate((self.standardized_states, self.kmeans_centers))

        Y = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
        ).fit_transform(X)
        print(f"({self.num_states_when_ran_kmeans}-{state_length-self.num_states_when_ran_kmeans}-{len(Y)})")
        plt.title(label=f"Perplexity={perplexity}, K={self.K}, Middle={after_kmeans}")
        plt.scatter(Y[self.num_states_when_ran_kmeans: state_length, 0], Y[self.num_states_when_ran_kmeans: state_length, 1], c="b")
        plt.scatter(Y[:self.num_states_when_ran_kmeans, 0], Y[:self.num_states_when_ran_kmeans, 1], c="g")
        plt.scatter(Y[state_length:, 0], Y[state_length:, 1], c="r")
        #ax.xaxis.set_major_formatter(NullFormatter())
        #ax.yaxis.set_major_formatter(NullFormatter())
        #ax.axis("tight")
        plt.show()
        print("Finished TSNE!")


    def calc_standard_kmeans(self, run_tsne=False):
        self.num_states_when_ran_kmeans= len(self.states)
        if self.weighted_kmeans:
            self.weights = self.get_kmeans_weights()
        else:
            self.weights = np.ones_like(self.rewards)
        self.scaler = preprocessing.StandardScaler().fit(self.states)
        self.standardized_states = self.scaler.transform(self.states)
        self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states, sample_weight=self.weights).cluster_centers_
        if run_tsne:
            self.tsne()
        if self.use_vectors:
            self.center_vectors = self.calc_kmeans_center_vector(self.kmeans_centers)
        self.kmeans_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(self.kmeans_centers)
        self.center_max_rewards = np.max(self.kmeans_action_reward_list, axis=0)

    def get_kmeans_weights(self):
        formatted_input = np.asarray(self.rewards) / (max(np.abs(self.rewards)))
        softmaxed_rewards = softmax(formatted_input)
        return softmaxed_rewards

    def get_action_kmeans(self, state):
        standardized_state = self.scaler.transform([state])

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]


        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
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

    def get_action_with_vector(self, state):
        if self.vector_type == 1:
            new_states = self.calc_new_states_for_actions(state)
        else:
            new_states = self.calc_new_states_for_actions_by_closest(state)
        expected_values = [self.calc_value_of_state(destination) for destination in new_states]
        return np.argmax(expected_values)

    def calc_new_states_for_actions(self, state):
        standardized_state = self.scaler.transform([state])

        new_standardized_states = []

        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sum = np.sum(weight)
                vector = self.center_vectors[:, action, :].transpose().dot(weight) / weight_sum
                new_standardized_states.append(standardized_state + vector)

        return new_standardized_states

    def calc_new_states_for_actions_by_closest(self, state):
        standardized_state = self.scaler.transform([state])

        new_standardized_states = []
        dist = standardized_state - self.kmeans_centers
        center_index = np.argmin(np.sum(np.square(dist), axis=1))
        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                vector = self.center_vectors[center_index, action, :]
                new_standardized_states.append(standardized_state + vector)
        return new_standardized_states

    def calc_value_of_state(self, standardized_state):
        dist = standardized_state - self.kmeans_centers
        weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
        weight_sum = np.sum(weight)
        action_rewards = weight.dot(self.center_max_rewards) / weight_sum
        return action_rewards


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

    def calc_kmeans_center_vector(self, centers):
        # In this step, we use weighted average to give each center 1 vector per point.
        # When picking actions, we use vectors to estimate location, and use location to get reward.
        action_vector_list = []
        weight_sums_list = []
        for kmean in centers:
            action_vectors = np.empty((self.action_space_size, self.observation_space_size))
            weight_sums = [0. for _ in range(self.action_space_size)]
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = kmean - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                    weight_sums[action] = np.sum(weight)
                    vectors = self.states[self.state_action_transitions_to[action]] - self.states[self.state_action_transitions_from[action]]
                    action_vectors[action] = vectors.transpose().dot(weight) / weight_sums[action]

            action_vector_list.append(action_vectors)
            weight_sums_list.append(weight_sums)
        print(action_vectors)
        print("\n\n\n")
        print(action_vector_list)
        np_vector_list = np.asarray(action_vector_list)
        print(f"{np_vector_list.shape=}")
        return np_vector_list