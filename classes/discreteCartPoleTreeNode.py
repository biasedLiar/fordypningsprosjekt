import random
from helper.strategy_names import *


##### Constants ########
RISK_THRESHOLD = 4.5
EXPLORE_PERCENTAGE = 0.5
STRATEGY = BALANCED
ACCURACY_LIMIT = 0.1

PRINT_CHOICES = False

########################

class DCartPoleTreeNode:
    def __init__(self, state, time_step, strategy, id):
        self.name = "test"
        self.children = [{}, {}]
        self.state = state
        self.time_step = time_step
        self.is_final = False
        self.max_depth = [1, 1]
        self.max_unexplored_depth = [1, 1]
        self.risk_weighted_score = self.update_risk_weighted_score()
        self.state_bucket = self.calc_state_bucket(state)
        self.holes_beneath = [1, 1]
        self.most_recent_choice = -1
        self.strategy = strategy
        self.visited = 0
        self.id = id
        self.odds_of_unvisited_below = 1

    def mark_final(self):
        self.max_depth = [0, 0]
        self.max_unexplored_depth = [0, 0]
        self.risk_weighted_score = [0, 0]
        self.holes_beneath = [0, 0]
        self.is_final = True
        self.visited += 1


    def update(self):
        if bool(self.children[0]) and bool(self.children[1]):
            self.max_depth = [self.average_max_depth(self.children[0]), self.average_max_depth(self.children[1])]
            self.max_unexplored_depth = [self.average_max_unexplored_depth(self.children[0]), self.average_max_unexplored_depth(self.children[1])]
            if max(self.max_unexplored_depth) >= 0.5:
                self.max_unexplored_depth = [depth + 1 for depth in self.max_unexplored_depth]
            self.holes_beneath = [self.average_holes_beneath(self.children[0]), self.average_holes_beneath(self.children[1])]
        else:
            i = self.most_recent_choice
            self.max_depth[i] = self.average_max_depth(self.children[i])
            self.max_unexplored_depth[i] = self.average_max_unexplored_depth(self.children[i])
            self.holes_beneath[i] = self.average_holes_beneath(self.children[i])
        self.visited += 1
        self.risk_weighted_score = self.update_risk_weighted_score()
        test = 1


    def get_state_bucket(self):
        bucket = [round(val/ACCURACY_LIMIT)*ACCURACY_LIMIT for val in self.state]
        return str(bucket)

    def calc_state_bucket(self, new_state):
        bucket = [round(val/ACCURACY_LIMIT)*ACCURACY_LIMIT for val in new_state]
        return str(bucket)

    def register_move(self, new_state, new_time_step, direction):
        bucket = self.calc_state_bucket(new_state)
        if bucket in self.children[direction]:
            self.children[direction][bucket].strategy = self.strategy
        else:
            id = self.id + str(self.most_recent_choice)
            self.children[direction][bucket] = DCartPoleTreeNode(new_state, new_time_step, self.strategy, id)
        return self.children[direction][bucket]

    def register_existing(self, new_node, new_time_step, direction):
        bucket = self.calc_state_bucket(new_node.get_state_bucket())
        if bucket in self.children[direction]:
            self.children[direction][bucket].strategy = self.strategy
        else:
            id = self.id + str(self.most_recent_choice)
            self.children[direction][bucket] = new_node
            self.children[direction][bucket].strategy = self.strategy
        return self.children[direction][bucket]

    def child_has_unexplored(self, child):
        for state, node in self.children[child].items():
            if max(node.max_unexplored_depth) > 0:
                return True
        return False

    def pick_action(self, new_strategy=None):
        if new_strategy != None:
            self.strategy = new_strategy

        if self.strategy == MAXIMIZE_POINTS:
            choice = self.maximize_points()
        elif self.strategy == BALANCED:
            if self.time_step == 0 and self.consider_explore():
                choice = self.explore()
                self.strategy = EXPLORE
            else:
                choice = self.maximize_points()
        elif self.strategy == EXPLORE:
            choice = self.explore()
        else:
            choice = 0
        self.most_recent_choice = choice
        if self.most_recent_choice == -1:
            test = 1
        return choice

    def states_are_equal(self, obs1, obs2):
        if len(obs1) != len(obs2):
            return False
        for i in range(len(obs1)):
            if obs1[i] != obs2[i]:
                return False
        return True

    def risk_weighted_score_of_child(self, child):
        if self.is_final:
            return 0
        if not child:
            return 1 + RISK_THRESHOLD
        total_sum = self.average_max_depth(child)
        unexplored_sum = self.average_max_unexplored_depth(child)
        unexplored_sum += (RISK_THRESHOLD if unexplored_sum > 0 else 0)

        return max(total_sum, unexplored_sum)

    def best_risk_weighted_score(self):
        return max(self.risk_weighted_score)

    def average_max_depth(self, child):
        if self.is_final:
            return 0
        if not child:
            return 1
        sum = 0
        visited = 0
        for key, node in child.items():
            visited += node.visited
            sum += node.visited * (max(node.max_depth[0], node.max_depth[1]))
        sum /= visited
        return sum + 1

    def average_max_unexplored_depth(self, child):
        if self.is_final:
            return 0
        if not child:
            return 1
        sum = 0
        more_than_1 = 0
        visited = 0
        for key, node in child.items():
            visited += node.visited
            if not node.is_final:
                sum += node.visited * max(node.max_unexplored_depth)
                more_than_1 += node.visited * (1 if max(node.max_unexplored_depth) >= 1 else 0)
        sum /= visited
        if more_than_1 < visited*0.5:
            sum = -1
        return sum + 1

    def average_holes_beneath(self, child):
        if self.is_final:
            return 0
        if not child:
            return 1
        sum = 0
        visited = 0
        for key, node in child.items():
            visited += node.visited
            sum += node.visited * (node.holes_beneath[0] + node.holes_beneath[1])
        sum /= visited
        return sum


    def update_risk_weighted_score(self):
        return [self.risk_weighted_score_of_child(self.children[0]),
                                        self.risk_weighted_score_of_child(self.children[1])]

    def explore(self):
        left_explorable = not self.children[0] or self.child_has_unexplored(0)
        right_explorable = not self.children[1] or self.child_has_unexplored(1)
        if left_explorable and not right_explorable:
            return 0
        if right_explorable and not left_explorable:
            return 1
        return self.maximize_points()

    def maximize_points(self):
        left_score = self.risk_weighted_score[0]
        right_score = self.risk_weighted_score[1]
        if left_score == right_score:
            return 0
        if PRINT_CHOICES:
            print(f"{self.id=}, Choosing {0 if left_score > right_score else 1}, {left_score=}, {right_score=}")
            input()
        return 0 if left_score > right_score else 1

    def consider_explore(self):
        should_explore = random.random() < EXPLORE_PERCENTAGE
        return should_explore

    def set_root_strategy(self, strategy):
        self.strategy = strategy

    def visualize_tree(self):
        if self.children[0] == None and self.children[1] == None:
            return [1]
        if self.children[0] != None and self.children[1] != None:
            vis_list = [1]
            visual_0 = self.children[0].visualize_tree()
            visual_1 = self.children[1].visualize_tree()
            for i in range(max(len(visual_1), len(visual_0))):
                sum = 0
                if len(visual_0) > i:
                    sum += visual_0[i]
                if len(visual_1) > i:
                    sum += visual_1[i]
                vis_list.append(sum)
            return [1] + vis_list
        i = 0 if self.children[0] != None else 1
        return [1] + self.children[i].visualize_tree()

    def show_selected_path(self):
        if self.most_recent_choice == -1:
            return ""
        return str(self.most_recent_choice) + self.children[self.most_recent_choice].show_selected_path()












