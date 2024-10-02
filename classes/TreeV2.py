from TreeNodeV2 import *
from helper.strategy_names import *


class TreeV2:
    def __init__(self, bucket_accuracy, observation, strategy):
        self.discrete_nodes = {}
        self.bucket_accuracy = bucket_accuracy
        self.current_node = self.create_new_node(self.calc_bucket(observation))
        self.visited_nodes = []
        self.strategy = strategy



    ################## Publicly called functions ####################################


    def start_round(self, observation):
        bucket = self.calc_bucket(observation)
        if bucket in self.discrete_nodes:
            self.current_node = self.discrete_nodes[bucket]
        else:
            self.current_node = self.create_new_node(observation)


    def pick_action(self):
        action = self.current_node.pick_action(self.strategy)
        return action

    def update_result(self):
        print("test")


    ############### Internal helper functions ####################################
    def create_new_node(self, bucket):
        new_node = TreeNodeV2(bucket)
        if bucket in self.discrete_nodes:
            raise Exception("Creating node for existing bucket")
        self.discrete_nodes[bucket] = new_node
        return new_node


    def calc_bucket(self, observation):
        bucket = [round(i/self.bucket_accuracy)*self.bucket_accuracy for i in observation]
        bucket[1] = round(observation[1]/1.2)
        return str(bucket)












