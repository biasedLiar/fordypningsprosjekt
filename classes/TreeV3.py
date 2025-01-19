import numpy as np
from heapq import nsmallest
from classes.TreeNodeV3 import *
from helper.strategy_names import *
from classes.State import *


class TreeV3:
    def __init__(self, observation, num_nodes_checked=3, stop_exploring_distance=0.2, layers_checked=1, gamma=0.8):
        self.state_actions = [[], []]
        self.state_actions_linalg = [np.zeros((0, 4)), np.zeros((0, 4))]
        self.nodes = {}
        self.nodes_array = []
        self.nodes_array_linalg = np.zeros((0, 4))
        self.gamma = gamma
        self.current_node = self.create_new_node(State(observation))
        self.visited_nodes = []
        self.steps_alive = 0
        self.generations = 0
        self.num_nodes_checked = num_nodes_checked
        self.stop_exploring_distance = stop_exploring_distance
        self.most_recent_action = -1
        self.layers_checked = layers_checked




    ################## Publicly called functions ####################################


    def start_round(self, observation):
        self.steps_alive = 0
        self.update_result(observation)

    def pick_action(self):
        evs = self.multi_lvl_evs(self.current_node.state, self.layers_checked)
        action = np.argmin(evs)
        self.most_recent_action = action
        return action




    def update_result(self, observation, terminated=False):
        if terminated:
            raise Exception("Update tried after termination")
            return
        new_node = self.create_new_node(State(observation))
        self.current_node.children[self.most_recent_action] = new_node
        self.current_node.most_recent_action = self.most_recent_action
        self.visited_nodes.append(self.current_node)
        self.current_node = new_node




    def finished_round(self, terminated=True):
        if terminated:
            self.current_node.times_visited += 1
            self.current_node.ev = 100
        else:
            self.current_node.times_visited += 1
            self.current_node.ev = 0
        prev_node = self.current_node
        for node in reversed(self.visited_nodes):
            node.times_visited += 1
            node.ev = prev_node.ev * self.gamma
            self.add_node_to_tree(node)
            prev_node = node

        self.perform_linalg_conversion()

        self.visited_nodes = []
        self.generations += 1





    def get_num_nodes(self):
        return len(self.nodes)

    ############### Internal helper functions ####################################

    def multi_lvl_evs(self, current_state, layers_left):
        new_states, farthest_distances = self.calc_new_states(current_state)

        # Check if node should explore
        if self.is_bottom_lvl(layers_left) and self.should_explore_given(farthest_distances):
            if farthest_distances[0] == -1:
                action = 0
            elif farthest_distances[1] == -1:
                action = 1
            else:
                action = 0 if farthest_distances[0] >= farthest_distances[1] else 1
            evs = [1, 1]
            evs[action] = 0
            return evs


        if layers_left == 1:
            evs = self.get_evs_of_states(new_states)
        else:
            evs = []
            for state in new_states:
                next_evs = self.multi_lvl_evs(state, layers_left - 1)
                evs.append(min(next_evs))

        return evs




    def add_node_to_tree(self, node):
        self.nodes[node.state] = node
        self.nodes_array.append(node)
        action = node.most_recent_action
        self.state_actions[action].append(node)

    def perform_linalg_conversion(self):
        for i in range(2):
            old_size = np.size(self.state_actions_linalg[i], axis=0)
            new_size = len(self.state_actions[i])
            if old_size == new_size:
                continue
            new_action_states = np.zeros((new_size - old_size, 4))
            for j in range(old_size, new_size):
                new_action_states[j - old_size] = self.state_actions[i][j].state.obs
            total_array = np.concatenate((self.state_actions_linalg[i], new_action_states), axis=0)
            self.state_actions_linalg[i] = total_array

        old_size = np.size(self.nodes_array_linalg, axis=0)
        new_size = len(self.nodes_array)
        if old_size != new_size:
            new_nodes_linalg = np.zeros((new_size - old_size, 4))
            for j in range(old_size, new_size):
                new_nodes_linalg[j - old_size] = self.nodes_array[j].state.obs
            total_array = np.concatenate((self.nodes_array_linalg, new_nodes_linalg), axis=0)
            self.nodes_array_linalg = total_array


    def calc_new_states(self, state):
        new_states = []
        farthest_distance = []
        for i in range(2):
            if len(self.state_actions_linalg[i]) == 0:
                new_states.append(-1)
                farthest_distance.append(-1)
            else:
                distances = np.linalg.norm(self.state_actions_linalg[i] - state.obs, axis=1)
                enumerated = enumerate(distances)
                #length1 = sum(1 for _, __ in enumerated)
                enumerated = nsmallest(self.num_nodes_checked, enumerated, key=lambda r: r[1])
                max_distance = max(enumerated, key=lambda r: r[1])[1]
                nodes = [self.state_actions[i][j[0]] for j in enumerated]
                destination = self.new_state_from_average(state, nodes, i)

                new_states.append(destination)
                new_var = max_distance
                farthest_distance.append(new_var)

        return new_states, farthest_distance

    def get_evs_of_states(self, states):
        evs = []
        for state in states:
            evs.append(self.calc_value_of_state(state))
        return evs


    def calc_value_of_state(self, state):
        if len(self.nodes_array_linalg) == 0:
            return -1
        else:
            distances = np.linalg.norm(self.nodes_array_linalg - state.obs, axis=1)
            enumerated = enumerate(distances)
            if len(distances) > self.num_nodes_checked:
                enumerated = nsmallest(self.num_nodes_checked, enumerated, key=lambda r: r[1])
            nodes = [self.nodes_array[j[0]] for j in enumerated]
            ev = sum([node.ev for node in nodes])/len(nodes)
            return ev


    def is_bottom_lvl(self, layers_left):
        return layers_left == self.layers_checked

    def should_explore_given(self, distances):
        if distances[0] == -1 or distances[1] == -1:
            return True
        max_d = max(distances)
        min_d = min(distances)
        return max_d - min_d > self.stop_exploring_distance



    def get_average_ev(self, nodes):
        ev = (node.ev for node in nodes)/len(nodes)

    def create_new_node(self, state):
        if state in self.nodes.keys():
            node = self.nodes[state]
            raise Exception("Repeat node, this should not happen")
        else:
            node = TreeNodeV3(state, ev_multiplier=self.gamma)
        return node

    def new_state_from_average(self, state, nodes, action):
        deltas = np.zeros(4)
        for node in nodes:
            start = node.state.obs
            end = node.children[action].state.obs
            delta = end - start
            deltas += delta
        if len(nodes) == 0:
            print("Test")
            input()
        deltas /= len(nodes)
        destination = State(state.obs + deltas)
        return destination

    def get_linalg_action_rewards(self):
        ordered_action_rewards = []
        for observation in self.nodes_array_linalg:
            state = State(observation)
            node = self.nodes[state]
            ordered_action_rewards.append((node.most_recent_action, node.ev))
        return ordered_action_rewards




