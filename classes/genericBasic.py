import time

import numpy as np
import sklearn.utils

from helper.kmeans_model import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from matplotlib.ticker import NullFormatter

from helper.special_plot import Plot
from sklearn.utils import gen_even_slices



class GenericModel:
    def __init__(self, action_space_n, observation_space_shape, gaussian_width, exploration_rate, weighted_kmeans=True,
                 K=20, do_standardize=True, use_search_tree=False,
                 search_tree_depth=-1):
        self.gaussian_width = gaussian_width
        self.gaussian_width_vector = gaussian_width
        self.action_space_size = action_space_n
        self.observation_space_size = observation_space_shape
        self.exploration_rate = exploration_rate
        self.weighted_kmeans = weighted_kmeans
        self.do_standardize = do_standardize
        self.K = K
        self.delta = 10**-8

        self.states: np.ndarray = np.empty((0, self.observation_space_size))  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index


        self.actions: list[int] = list(range(self.action_space_size))

        # A list for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(self.action_space_size)]  # type: ignore


        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]

        self.state_vectors: list[np.ndarray] = [np.empty((0, self.observation_space_size)) for i in range(self.action_space_size)]

        self.saved_state_action_transitions: list[list[tuple[int,
                                                       int]]] = [[] for _ in range(self.action_space_size)]  # type: ignore


        self.saved_state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.saved_state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]

        self.num_states_when_ran_kmeans = -1
        self.scaler = None
        self.standardized_states = []
        self.kmeans_centers: list[np.ndarray] = []
        self.center_action_reward_list: list[list[float]] = []
        self.kmeans_action_weight_list: list[list[float]] = []
        self.center_vectors = []

        self.use_search_tree = use_search_tree
        self.search_tree_depth = search_tree_depth

        self.midway = False


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
            if weight_sums[action] == 0 and not self.midway:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate and not self.midway:
                return action  # Return action that has little data for the current state
        return np.argmax(action_rewards)
        
    def standardize(self, nodes):
        if not self.do_standardize:
            return nodes

        if nodes.ndim == 1:
            return self.scaler.transform([nodes])

        return self.scaler.transform(nodes)

    def tsne(self, blue_states=[]):
        state_length = len(self.standardized_states)
        center_length = len(self.kmeans_centers)
        perplexity = 20
        if len(blue_states) != 0:
            X = np.concatenate((self.standardized_states, self.kmeans_centers, blue_states))
        else:
            X = np.concatenate((self.standardized_states, self.kmeans_centers))
        Y = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
        ).fit_transform(X)
        plt.title(label=f"Perplexity={perplexity}, K={self.K}, gaussian={self.gaussian_width}")
        plt.scatter(Y[:self.num_states_when_ran_kmeans, 0], Y[:self.num_states_when_ran_kmeans, 1], c="black", s=1)

        if len(blue_states) != 0:
            colors_map = np.arange(len(blue_states))
            plt.scatter(Y[state_length+center_length:, 0],
                        Y[state_length+center_length:, 1], c=colors_map, cmap="jet", s=16)


        plt.scatter(Y[state_length:state_length+center_length, 0], Y[state_length:state_length+center_length, 1], s=16,
                    facecolors="none", edgecolors="r")
        plt.show()
        plt.clf()

    def tsne_of_path(self, path):
        standardized_path = self.standardize(path)
        self.tsne(blue_states=standardized_path)



    def calc_search_tree_state_vectors(self, ignore_kmeans=True):
        if self.do_standardize:
            self.scaler = preprocessing.StandardScaler().fit(self.states)
            self.standardized_states = self.scaler.transform(self.states)
        else:
            self.standardized_states = self.states

        for i in range(self.action_space_size):
            self.state_vectors[i] = self.standardized_states[self.state_action_transitions_to[i]] - self.standardized_states[
                    self.state_action_transitions_from[i]]

    def calc_search_tree_kmeans(self, run_tsne=False, write_logs=True):

        self.num_states_when_ran_kmeans = len(self.states)
        if self.weighted_kmeans:
            self.weights = self.get_kmeans_weights()
        else:
            self.weights = np.ones_like(self.rewards)

        if len(self.states) / 2 < self.K:
            self.K = int(np.floor(len(self.states) / 2))
        self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states,
                                                                                               sample_weight=self.weights).cluster_centers_

        if run_tsne:
            self.tsne()
        self.center_vectors = self.calc_tree_kmeans_center_vectors(self.kmeans_centers)

        self.center_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(
            self.kmeans_centers)


    def calc_tree_kmeans_center_vectors(self, centers):
        action_vector_list = []
        for kmean in centers:
            action_vectors = np.empty((self.action_space_size, self.observation_space_size))
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = kmean - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width_vector) + self.delta
                    weight_sums = np.sum(weight)
                    vectors = self.state_vectors[action]
                    action_vectors[action] = vectors.transpose().dot(weight) / weight_sums

            action_vector_list.append(action_vectors)
        np_vector_list = np.asarray(action_vector_list)

        np_vector_list = np.transpose(np_vector_list, axes=[1, 0, 2])
        return np_vector_list

    def calc_standard_kmeans(self, run_tsne=False, write_logs=True):

        self.num_states_when_ran_kmeans= len(self.states)
        if self.weighted_kmeans:
            self.weights = self.get_kmeans_weights()
        else:
            self.weights = np.ones_like(self.rewards)

        if self.do_standardize:
            self.scaler = preprocessing.StandardScaler().fit(self.states)
            self.standardized_states = self.scaler.transform(self.states)
        else:
            self.standardized_states = self.states


        if len(self.states)/2 < self.K:
            self.K = int(np.floor(len(self.states)/2))
        self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states, sample_weight=self.weights).cluster_centers_

        if run_tsne:
            self.tsne()
        self.center_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(self.kmeans_centers)

    def remove_unecessary_centers(self):
        a, c = self.kmeans_centers.shape
        b, c = self.standardized_states.shape
        expanded_centers = self.kmeans_centers + np.zeros((b, 1, 1))
        expanded_states = self.standardized_states[:, np.newaxis, :] + np.zeros((1, a, 1))
        expanded_centers = np.sum(np.square(expanded_states - expanded_centers), axis=2)
        closest = np.unique(expanded_centers.argmin(axis=1))
        self.kmeans_centers = self.kmeans_centers[closest, :]


    def calc_kmeans_center_rewards(self, centers):
        action_rewards_list = []
        weight_sums_list = []
        for center in centers:
            action_rewards = [0. for _ in range(self.action_space_size)]
            weight_sums = [0. for _ in range(self.action_space_size)]
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = center - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                    weight_sums[action] = np.sum(weight)
                    action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]
            action_rewards_list.append(action_rewards)
            weight_sums_list.append(weight_sums)
        return np.asarray(action_rewards_list).transpose(), weight_sums_list



    def get_kmeans_weights(self):
        formatted_input = (2*np.asarray(self.rewards) / (max(np.abs(self.rewards)))) - 1
        formatted_input = 1 / (1 + np.exp(-5*formatted_input))
        return formatted_input

        formatted_input = np.asarray(self.rewards) / max(np.abs(self.rewards))
        return formatted_input


    def get_action_kmeans(self, state, debug=False):
        standardized_state = self.standardize(state)

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]
        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.center_action_reward_list[action]) / weight_sums[action]

        return np.argmax(action_rewards)

    def estimate_resulting_states(self, origin_state, ignore_kmeans=True):
        new_standardized_states = []
        if ignore_kmeans:
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = origin_state - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                    weight_sum = np.sum(weight)
                    vectors = self.state_vectors[action]
                    vector = vectors.transpose().dot(weight) / weight_sum
                    new_standardized_states.append(origin_state + vector)
                else:
                    new_standardized_states.append(origin_state)
        else:
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = origin_state - self.kmeans_centers
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                    weight_sum = np.sum(weight)
                    vectors = self.center_vectors[action]
                    vector = vectors.transpose().dot(weight) / weight_sum
                    new_standardized_states.append(origin_state + vector)
                else:
                    new_standardized_states.append(origin_state)

        return new_standardized_states


    def get_action_values(self, state, ignore_kmeans=True):
        action_values = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]
        if ignore_kmeans:
            for action in self.actions:
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = state - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                    weight_sums[action] = np.sum(weight)
                    action_values[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / \
                                             weight_sums[action]
        else:
            for action in self.actions:
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = state - self.kmeans_centers
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                    weight_sums[action] = np.sum(weight)
                    action_values[action] = np.sum(weight * self.center_action_reward_list[action]) / \
                                             weight_sums[action]
        return action_values

    def search_tree_action_values(self, current_state, search_depth, ignore_kmeans=True):
        action_values = []
        if search_depth <= 0:
            action_values = self.get_action_values(current_state, ignore_kmeans=ignore_kmeans)
        else:
            new_states = self.estimate_resulting_states(current_state, ignore_kmeans=ignore_kmeans)
            for state in new_states:
                action_values.append(np.max(self.search_tree_action_values(state, search_depth - 1, ignore_kmeans=ignore_kmeans)))

        return action_values


    def get_action_search_tree(self, state, ignore_kmeans = True):
        state = self.standardize(state)
        action_values = self.search_tree_action_values(state, search_depth=self.search_tree_depth, ignore_kmeans=ignore_kmeans)
        action = np.argmax(action_values)
        return action
