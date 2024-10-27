from classes.TreeNodeV3 import *
from helper.strategy_names import *
from classes.State import *


class TreeV3:
    def __init__(self, observation):
        self.state_actions = [{}, {}]
        self.nodes = {}
        self.current_node = self.create_new_node(State(observation))
        self.visited_nodes = []
        raise NotImplementedError("Finish function")




    ################## Publicly called functions ####################################


    def start_round(self, observation):
        raise NotImplementedError("Finish function")


    def pick_action(self):

        raise NotImplementedError("Finish function")

    def update_result(self, observation, terminated=False):

        raise NotImplementedError("Finish function")


    def finished_round(self, terminated=True):
        raise NotImplementedError("Finish function")


    def get_num_nodes(self):
        return len(self.discrete_nodes)

    ############### Internal helper functions ####################################

    def create_new_node(self, state):
        if state in self.nodes.keys():
            node = self.nodes[state]
        else:
            node = TreeNodeV3(state)
        return node







