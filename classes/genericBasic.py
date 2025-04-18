import numpy as np
import sklearn.utils

from helper.K_MeansTypes import *
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
                 use_vectors=False, split_kmeans=False, K=20, no_learning=True, use_kmeans=True, vector_type=1,
                 do_standardize=True, use_special_kmeans=False):
        self.gaussian_width = gaussian_width
        self.gaussian_width_vector = gaussian_width
        self.action_space_size = action_space_n
        self.observation_space_size = observation_space_shape
        self.exploration_rate = exploration_rate
        self.weighted_kmeans = weighted_kmeans
        self.split_kmeans = split_kmeans
        self.use_vectors = use_vectors
        self.vector_type = vector_type
        self.do_standardize = do_standardize
        self.K = K
        self.no_learning = no_learning
        self.delta = 10**-8
        self.use_kmeans = use_kmeans

        self.states: np.ndarray = np.empty((0, self.observation_space_size))  # States are stored here
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



        self.num_batches_analyzed = 600
        self.num_batches_analyzed = 300
        self.use_special_kmeans = use_special_kmeans
        self.show_special = False
        self.special_kmeans_gaussian = 0.1


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
        
    def standardize(self, nodes):
        if not self.do_standardize:
            return nodes

        if nodes.ndim == 1:
            return self.scaler.transform([nodes])

        return self.scaler.transform(nodes)

    def tsne(self, blue_states=[]):
        print("Starting TSNE...")
        state_length = len(self.standardized_states)
        center_length = len(self.kmeans_centers)
        print(f"{self.num_states_when_ran_kmeans=}")
        print(f"{state_length=}")
        print(f"{center_length=}")
        print(f"{len(blue_states)=}")
        perplexity = 20
        if len(blue_states) != 0:
            X = np.concatenate((self.standardized_states, self.kmeans_centers, blue_states))
        else:
            X = np.concatenate((self.standardized_states, self.kmeans_centers))
        print(f"{len(X)=}")
        Y = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
        ).fit_transform(X)
        print(f"({self.num_states_when_ran_kmeans}-{state_length - self.num_states_when_ran_kmeans}-{len(Y)})")
        plt.title(label=f"Perplexity={perplexity}, K={self.K}, gaussian={self.gaussian_width}")
        plt.scatter(Y[:self.num_states_when_ran_kmeans, 0], Y[:self.num_states_when_ran_kmeans, 1], c="black", s=1)

        if len(blue_states) != 0:
            print(f"{len(X)} - {state_length+center_length} = {len(X)-state_length-center_length}")
            colors_map = np.arange(len(blue_states))
            # Starts at blue, ends at red
            plt.scatter(Y[state_length+center_length:, 0],
                        Y[state_length+center_length:, 1], c=colors_map, cmap="jet", s=16)


        plt.scatter(Y[state_length:state_length+center_length, 0], Y[state_length:state_length+center_length, 1], s=16,
                    facecolors="none", edgecolors="r")
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        # ax.axis("tight")
        plt.show()
        plt.clf()
        print("Finished TSNE!")


    def tsne_based_on_reward(self, blue_states=[]):
        print("Starting TSNE...")
        state_length = len(self.standardized_states)
        center_length = len(self.kmeans_centers)
        print(f"{self.num_states_when_ran_kmeans=}")
        print(f"{state_length=}")
        print(f"{center_length=}")
        print(f"{len(blue_states)=}")
        perplexity = 20
        if len(blue_states) != 0:
            X = np.concatenate((self.standardized_states, self.kmeans_centers, blue_states))
        else:
            X = np.concatenate((self.standardized_states, self.kmeans_centers))
        print(np.sqrt(self.rewards))
        print(f"{len(X)=}")
        Y = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
        ).fit_transform(X)
        print(f"({self.num_states_when_ran_kmeans}-{state_length - self.num_states_when_ran_kmeans}-{len(Y)})")
        plt.title(label=f"Perplexity={perplexity}, K={self.K}, gaussian={self.gaussian_width}")
        plt.scatter(Y[:self.num_states_when_ran_kmeans, 0], Y[:self.num_states_when_ran_kmeans, 1],
                    c=self.rewards, cmap="jet", s=np.sqrt(self.rewards))


        plt.scatter(Y[state_length:state_length+center_length, 0], Y[state_length:state_length+center_length, 1], s=16,
                    facecolors="none", edgecolors="black")

        if len(blue_states) != 0:
            print(f"{len(X)} - {state_length+center_length} = {len(X)-state_length-center_length}")
            colors_map = np.arange(len(blue_states))
            # Starts at blue, ends at red
            plt.scatter(Y[state_length+center_length:, 0],
                        Y[state_length+center_length:, 1], c=colors_map, cmap="jet", s=16)
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        # ax.axis("tight")
        plt.show()
        plt.clf()
        print("Finished TSNE!")

    def tsne_of_path(self, path):
        standardized_path = self.standardize(path)
        self.tsne(blue_states=standardized_path)
        print("Done")

    def get_batches(self, data, batch_size):
        np.random.shuffle(data)
        num_batches = int(np.floor(data.shape[0]/batch_size))
        batch_slices = list(gen_even_slices(data.shape[0], num_batches))
        batches = [data[batch_slice] for batch_slice in batch_slices]
        return batches
        #Continue creating batches.


    def get_centers_special_kmeans(self, data, write_logs=True):
        if write_logs:
            print(f"{data.shape=}")
        data_length = data.shape[0]
        model = Model(K=self.K, D=4, sigma=self.gaussian_width, lambda_=0.5, learning_rate=0.02)
        model.mu -= 0.5
        if self.show_special:
            plot = Plot(model, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
        i = 0
        hashable_standardized_states = [tuple(state) for state in self.standardized_states.tolist()]
        weight_dict = dict(zip(hashable_standardized_states, (self.weights*self.weights.shape[0])))
        while i < self.num_batches_analyzed:
            batches = self.get_batches(data, 100)
            for batch in batches:
                for x in batch:
                    model.update(x, weighted=self.weighted_kmeans, weight=weight_dict[tuple(x)])
                # TODO update wiehgts so they are normalized
                # Check if updates are finished
                if self.show_special:
                    plot.draw_frame_no_label(batch=batch)
                i += 1
            if write_logs:
                print(i)
        if self.show_special:
            plot.clf()


        return model.mu


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


        if self.use_special_kmeans:
            self.kmeans_centers = self.get_centers_special_kmeans(self.standardized_states, write_logs=write_logs)
            print(f"{self.kmeans_centers.shape} centers before pruning.")
            self.remove_unecessary_centers(write_logs=write_logs)
            print(f"{self.kmeans_centers.shape} centers after pruning.")
        else:
            if len(self.states)/2 < self.K:
                print(f"K too big, reduced.")
                self.K = int(np.floor(len(self.states)/2))
                pass
            self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states, sample_weight=self.weights).cluster_centers_

        if write_logs:
            print(f"{self.kmeans_centers.shape} centers for {self.K=}.")
        if run_tsne:
            self.tsne()
        if self.use_vectors:
            self.center_vectors = self.calc_kmeans_center_vector(self.kmeans_centers)
        self.kmeans_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(self.kmeans_centers)
        self.center_max_rewards = np.max(self.kmeans_action_reward_list, axis=0)


    def remove_unecessary_centers(self, write_logs=True):
        a, c = self.kmeans_centers.shape
        b, c = self.standardized_states.shape
        expanded_centers = self.kmeans_centers + np.zeros((b, 1, 1))
        expanded_states = self.standardized_states[:, np.newaxis, :] + np.zeros((1, a, 1))
        expanded_centers = np.sum(np.square(expanded_states - expanded_centers), axis=2)
        closest = np.unique(expanded_centers.argmin(axis=1))
        self.kmeans_centers = self.kmeans_centers[closest, :]
        if write_logs:
            print(f"{self.kmeans_centers.shape=}")



    def get_kmeans_weights(self):
        formatted_input = np.asarray(self.rewards) / (max(np.abs(self.rewards)))
        softmaxed_rewards = softmax(formatted_input)
        return softmaxed_rewards

    def get_action_kmeans(self, state, debug=False):
        standardized_state = self.standardize(state)

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]


        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.kmeans_action_reward_list[action]) / weight_sums[action]
                #action_rewards[action] = np.sum(weight * self.rewards[self.state_action_transitions_to[action]]) / weight_sums[action]

        if debug:
            print(f"{standardized_state=}")
            print(f"\n{action_rewards=}")
            print(f"\n{weight_sums=}")

        if self.no_learning:
            return np.argmax(action_rewards)


        for action, _ in enumerate(self.state_action_transitions):
            if weight_sums[action] == 0:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate:
                return action  # Return action that has little data for the current state



        return np.argmax(action_rewards)

    def check_vector(self, old_state, new_real_state):
        old_state = self.standardize(old_state)
        new_real_state = self.standardize(new_real_state)
        if self.vector_type == 1:
            new_states = self.calc_new_states_for_actions(old_state)
        else:
            new_states = self.calc_new_states_for_actions_by_closest(old_state)
        expected_values = [self.calc_value_of_state(destination) for destination in new_states]
        if self.vector_type == 1:
            return np.argmax(expected_values)
        else:
            return np.argmax(expected_values)

    def get_action_with_vector(self, state):
        standardized_state = self.standardize(state)
        if self.vector_type == 1:
            new_states = self.calc_new_states_for_actions(standardized_state)
        else:
            new_states = self.calc_new_states_for_actions_by_closest(standardized_state)
        expected_values = [self.calc_value_of_state(destination) for destination in new_states]
        return np.argmax(expected_values)




    def calc_new_states_for_actions(self, standardized_state):
        new_standardized_states = []

        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sum = np.sum(weight)
                vector = self.center_vectors[:, action, :].transpose().dot(weight) / weight_sum
                new_standardized_states.append(standardized_state + vector)

        return new_standardized_states

    def calc_new_states_for_actions_by_closest(self, standardized_state):
        new_standardized_states = []
        dist = standardized_state - self.kmeans_centers
        center_index = np.argmin(np.sum(np.square(dist), axis=1))
        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                vector = self.center_vectors[center_index, action, :]
                new_standardized_states.append(standardized_state + vector)
        return new_standardized_states

    def calc_value_of_state(self, standardized_state):

        action_rewards = [0. for _ in self.state_action_transitions]
        weight_sums = [0. for _ in self.state_action_transitions]

        for action, _ in enumerate(self.state_action_transitions):
            if len(self.state_action_transitions_from[action]) > 0:
                dist = standardized_state - self.kmeans_centers
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
                weight_sums[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight * self.kmeans_action_reward_list[action]) / weight_sums[action]

        return max(action_rewards)


    def calc_kmeans_center_rewards(self, centers):
        action_rewards_list = []
        weight_sums_list = []
        for kmean in centers:
            action_rewards = [0. for _ in range(self.action_space_size)]
            weight_sums = [0. for _ in range(self.action_space_size)]
            for action, _ in enumerate(self.state_action_transitions):
                if len(self.state_action_transitions_from[action]) > 0:
                    dist = kmean - self.standardized_states[self.state_action_transitions_from[action]]
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width) + self.delta
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
                    weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width_vector) + self.delta
                    weight_sums[action] = np.sum(weight)
                    vectors = self.standardized_states[self.state_action_transitions_to[action]] - self.standardized_states[self.state_action_transitions_from[action]]
                    '''
                    print(vectors[0, :])

                    from_stand = self.scaler.transform([self.states[self.state_action_transitions_from[action]][0]])
                    to_stand = self.scaler.transform([self.states[self.state_action_transitions_to[action]][0]])
                    print(to_stand - from_stand)
                    input()
                    '''
                    action_vectors[action] = vectors.transpose().dot(weight) / weight_sums[action]

            action_vector_list.append(action_vectors)
            weight_sums_list.append(weight_sums)
        np_vector_list = np.asarray(action_vector_list)
        return np_vector_list