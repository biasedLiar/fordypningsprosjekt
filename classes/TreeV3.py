import numpy as np
from heapq import nsmallest
from classes.TreeNodeV3 import *
from helper.strategy_names import *
from classes.State import *


class TreeV3:
    def __init__(self, observation, num_nodes_checked=3):
        self.state_actions = [{}, {}]
        self.state_actions_linalg = [np.zeros(0), np.zeros(0)]
        self.nodes = {}
        self.current_node = self.create_new_node(State(observation))
        self.visited_nodes = []
        self.num_nodes_checked = num_nodes_checked
        raise NotImplementedError("Finish function")




    ################## Publicly called functions ####################################


    def start_round(self, observation):
        raise NotImplementedError("Finish function")


    def pick_action(self):
        #Find vectors
        current_state = self.current_node.state

        #evaluate vectors
        #choose
        raise NotImplementedError("Finish function")

    def update_result(self, observation, terminated=False):

        raise NotImplementedError("Finish function")


    def finished_round(self, terminated=True):
        raise NotImplementedError("Finish function")


    def get_num_nodes(self):
        return len(self.discrete_nodes)

    ############### Internal helper functions ####################################


    def calc_new_states(self, state):
        new_states = []
        for i in range(2):
            current_state = np.array(state)
            if len(self.state_actions_linalg[i]) == 0:
                test = 2
                return -1
            if len(self.state_actions_linalg[i]) > self.num_nodes_checked:
                distances = np.linalg.norm(self.state_actions_linalg[i] - current_state, axis=1)
                enumerated_smallest = nsmallest(self.num_nodes_checked, enumerate(distances), key=lambda r: r[1])
                indexes = [self.state_actions_linalg[i[0]] for i in enumerated_smallest]
            else:
                tesrt=1



    def create_new_node(self, state):
        #TODO write expection check that node does not already exist
        if state in self.nodes.keys():
            node = self.nodes[state]
        else:
            node = TreeNodeV3(state)
        return node







