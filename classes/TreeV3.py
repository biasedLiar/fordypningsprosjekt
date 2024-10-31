import numpy as np
from heapq import nsmallest
from classes.TreeNodeV3 import *
from helper.strategy_names import *
from classes.State import *


class TreeV3:
    def __init__(self, observation, num_nodes_checked=3, stop_exploring_distance=0.4):
        self.state_actions = [[], []]
        self.state_actions_linalg = [np.zeros(0), np.zeros(0)]
        self.nodes = {}
        self.nodes_array = []
        self.nodes_array_linalg = np.zeroes(0)
        self.current_node = self.create_new_node(State(observation))
        self.visited_nodes = []
        self.num_nodes_checked = num_nodes_checked
        self.stop_exploring_distance = stop_exploring_distance
        raise NotImplementedError("Finish function")




    ################## Publicly called functions ####################################


    def start_round(self, observation):
        raise NotImplementedError("Finish function")


    def pick_action(self):
        #Find vectors
        current_state = self.current_node.state
        new_states, farthest_distance = self.calc_new_states(current_state)
        if self.should_explore(farthest_distance):
            action = 0 if farthest_distance[0] > farthest_distance[1] else 1
        else:
            evs = self.get_evs_of_states(new_states)
            action = np.argmin(evs)
        return action

    def update_result(self, observation, terminated=False):

        raise NotImplementedError("Finish function")


    def finished_round(self, terminated=True):
        raise NotImplementedError("Finish function")


    def get_num_nodes(self):
        return len(self.discrete_nodes)

    ############### Internal helper functions ####################################


    def calc_new_states(self, state):
        new_states = []
        farthest_distance = []
        for i in range(2):
            current_state = np.array(state)
            if len(self.state_actions_linalg[i]) == 0:
                new_states.append(-1)
                farthest_distance.append(-1)
            else:
                distances = np.linalg.norm(self.state_actions_linalg[i] - current_state.obj, axis=1)
                enumerated = enumerate(distances)
                if len(self.state_actions[i]) > self.num_nodes_checked:
                    enumerated = nsmallest(self.num_nodes_checked, enumerated, key=lambda r: r[1])
                max_distance = max(enumerated, key=lambda r: r[1])
                states = [self.state_actions[i][j[0]] for j in enumerated]
                nodes = [self.nodes[state] for state in states]
                destination = current_state.new_state_from_average(nodes, i)

                new_states.append(destination)
                farthest_distance.append(max_distance)

        return new_states, farthest_distance

    def get_evs_of_states(self, states):
        evs = []
        for state in states:
            evs.append(self.calc_value_of_state(state))
        return evs


    def calc_value_of_state(self, state):
        current_state = np.array(state)
        if len(self.nodes_linalg) == 0:
            return -1
        else:
            distances = np.linalg.norm(self.nodes_array_linalg - current_state.obj, axis=1)
            enumerated = enumerate(distances)
            if len(enumerated) > self.num_nodes_checked:
                enumerated = nsmallest(self.num_nodes_checked, enumerated, key=lambda r: r[1])
            nodes = [self.nodes_array[j[0]] for j in enumerated]
            ev = (node.ev for node in nodes)/len(nodes)
            return ev




    def should_explore(self, distances):
        if distances[0] == -1 or distances[1] == -1:
            return True
        max_d = max(distances)
        min_d = min(distances)
        return max_d - min_d > self.stop_exploring_distance



    def get_average_ev(self, nodes):
        ev = (node.ev for node in nodes)/len(nodes)

    def create_new_node(self, state):
        #TODO write expection check that node does not already exist
        if state in self.nodes.keys():
            node = self.nodes[state]
        else:
            node = TreeNodeV3(state)
        return node







