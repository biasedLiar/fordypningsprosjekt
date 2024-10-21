from classes.TreeNodeV2 import *
from helper.strategy_names import *


class TreeV2:
    def __init__(self, bucket_accuracy, observation, strategy=MAXIMIZE_POINTS, num_neighbors_copied=3, use_many_neighbors=False):
        self.discrete_nodes = {}
        self.bucket_accuracy = bucket_accuracy
        self.current_node = self.create_new_node(observation)
        self.visited_nodes = []
        self.strategy = strategy
        self.steps_alive = 0
        self.iterations = 0
        self.num_neighbours_copied = num_neighbors_copied
        self.use_many_neighbors = use_many_neighbors




    ################## Publicly called functions ####################################


    def start_round(self, observation):
        self.steps_alive = 0
        bucket = self.calc_str_bucket(observation)
        if bucket in self.discrete_nodes:
            self.current_node = self.discrete_nodes[bucket]
        else:
            self.current_node = self.create_new_node(observation)


    def pick_action(self):
        if self.current_node.should_copy_neigbor(self.strategy):
            action = self.current_node.action_picked_previously_by_other_node
            if action == -1 or self.should_update_action_from_other_node():
                action = self.get_action_of_closest_neighbor()
                self.current_node.action_chosen_by_other_node(action)
        else:
            action = self.current_node.pick_action(self.strategy)
        return action

    def update_result(self, observation, terminated=False):
        if terminated:
            return
        self.visited_nodes.append(self.current_node)

        bucket = self.calc_str_bucket(observation)
        if bucket in self.discrete_nodes:
            self.current_node = self.discrete_nodes[bucket]
        else:
            self.current_node = self.create_new_node(observation)
        self.steps_alive += 1


    def finished_round(self, terminated=True):
        if terminated:
            self.current_node.just_died()
        else:
            self.current_node.just_won()
        prev_node = self.current_node
        if self.iterations > 300:
            #print(f"iterations is now at {self.iterations}")
            test = 2
        for node in reversed(self.visited_nodes):
            node.update(prev_node)
            prev_node = node

        for node in self.discrete_nodes.values():
            if len(node.chosen_actions) != 0:
                raise Exception("Node not emptying chosen action list")
        self.visited_nodes = []
        self.iterations += 1


    def get_num_nodes(self):
        return len(self.discrete_nodes)

    ############### Internal helper functions ####################################

    def create_new_node(self, observation):
        bucket = self.calc_str_bucket(observation)
        new_node = TreeNodeV2(bucket, 5.5)
        if bucket in self.discrete_nodes:
            raise Exception("Creating node for existing bucket")
        self.discrete_nodes[bucket] = new_node
        return new_node

    def calc_str_bucket(self, observation):
        bucket = self.calc_bucket(observation)
        return str(bucket)

    def calc_bucket(self, observation):
        bucket = [round(i/self.bucket_accuracy)*self.bucket_accuracy for i in observation]
        bucket[2] = round(observation[2]/(self.bucket_accuracy*0.1))*self.bucket_accuracy*0.1
        return bucket

    def bucket_to_state(self, node):
        handled_string = node.str_bucket.replace("[", "").replace("]", "").replace(",", "")
        state = [float(element) for element in handled_string.split()]
        return state

    def get_action_of_closest_neighbor(self):
        if self.use_many_neighbors:
            neighbors = self.find_closest_n_neighbors(self.num_neighbours_copied)
            action = self.vote_on_action(neighbors)
        else:
            neighbor = self.find_closest_neighbor()
            action = neighbor.pick_action_for_other_node(self.strategy)
        return action

    def find_closest_neighbor(self):
        closest_neighbor = self.current_node
        closest_distance = -1
        for distant_node in self.discrete_nodes.values():
            distance = self.distance_from_current_node_to(distant_node)
            if distance > 0 and (closest_distance == -1 or distance < closest_distance):
                closest_neighbor = distant_node
                closest_distance = distance
        return closest_neighbor

    def find_closest_n_neighbors(self, n):
        if len(self.discrete_nodes) == 1:
            return [self.current_node]

        if len(self.discrete_nodes) - 1 < n*n:
            n= np.floor(np.sqrt(len(self.discrete_nodes)))

        distances = {}
        for distant_node in self.discrete_nodes.values():
            distance = self.distance_from_current_node_to(distant_node)
            if distance > 0:
                distances[distant_node] = distance
        sorted_neighbors = [k for k, v in sorted(distances.items(), key=lambda node: node[1])]
        nearest_n_neigbors = sorted_neighbors[:int(n)]
        return nearest_n_neigbors

    def vote_on_action(self, neighbors):
        votes = [0, 0]
        for node in neighbors:
            recommended_action = node.pick_action_for_other_node(self.strategy)
            votes[recommended_action] += 1
        return int(np.round(votes[1]/len(neighbors)))

    def distance_from_current_node_to(self, node):
        distant_state = self.bucket_to_state(node)
        current_state = self.bucket_to_state(self.current_node)
        distance = np.sum([np.square(distant_state[i] - current_state[i]) for i in range(4)])
        return distance

    def should_update_action_from_other_node(self):
        num = self.current_node.times_visited
        while num > 2:
            if num%3 != 0:
                return False
            num = num/3
        return num == 1











