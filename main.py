# This is a sample Python script.
from multiprocessing import Pool, cpu_count
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import numpy as np
from helper.fileHelper import *
import markdown as md


def main():
    def calc_search_tree_kmeans(self):
        if len(self.states) / 2 < self.K:
            self.K = int(np.floor(len(self.states) / 2))
        self.kmeans_centers = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(self.standardized_states,
                                                                                               sample_weight=self.weights).cluster_centers_
        self.center_vectors = self.calc_tree_kmeans_center_vectors(self.kmeans_centers)
        self.center_action_reward_list, self.kmeans_action_weight_list = self.calc_kmeans_center_rewards(
            self.kmeans_centers)

def write_file():
    path =""

if __name__ == "__main__":
    main()