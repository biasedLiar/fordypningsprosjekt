import numpy as np

class KMeansController:
    def __init__(self, node_array, kmeans, nodes_action_rewards, scaler, observation, gaussian_distance=0.3):
        self.kmeans =  kmeans
        self.node_array = node_array
        self.nodes_action_rewards = nodes_action_rewards
        self.gaussian_distance = gaussian_distance
        self.evald_kmeans, self.weight_sums = self.eval_kmeans()
        self.scaler = scaler


        self.current_normalized_obs = self.scaler.transform([observation])
        #interact with observation here



    def pick_action(self):
        action_rewards = [0., 0.]
        weight_sums = [0., 0.]
        for i, node in enumerate(self.kmeans):
            dist = self.current_normalized_obs - node
            weight = np.exp(-np.sum(np.square(dist)) / self.gaussian_distance)
            for j in range(2):
                weight_sums[j] += weight
                action_rewards[j] += weight * self.evald_kmeans[i][j]
        for i in range(2):
            if weight_sums[i] > 0.:
                action_rewards[i] /= weight_sums[i]
        return np.argmax(action_rewards)



    def update_result(self, observation, terminated):
        self.current_normalized_obs = self.scaler.transform([observation])

    def finished_round(self, terminated):
        self.current_normalized_obs = -1

    def eval_kmeans(self):
        action_rewards_list = []
        weight_sums_list = []
        for kmean in self.kmeans:
            action_rewards = [0., 0.]
            weight_sums = [0., 0.]
            for i, node in enumerate(self.node_array):
                dist = kmean - node
                weight = np.exp(-np.sum(np.square(dist)) / self.gaussian_distance)
                j = self.nodes_action_rewards[i][0]
                weight_sums[j] += weight
                action_rewards[j] += weight * self.nodes_action_rewards[i][1]
            for i in range(2):
                if weight_sums[i] > 0.:
                    action_rewards[i] /= weight_sums[i]
            action_rewards_list.append(action_rewards)
            weight_sums_list.append(weight_sums)
        return action_rewards_list, weight_sums_list
