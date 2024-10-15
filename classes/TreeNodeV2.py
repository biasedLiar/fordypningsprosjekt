from helper.strategy_names import *
import numpy as np

class TreeNodeV2:
    def __init__(self, str_bucket, unknown_reward=0, odds_of_exploration_threshold=1/3):
        self.str_bucket = str_bucket
        self.children = [{}, {}]
        self.unknown_reward = unknown_reward
        self.times_visited = 0
        self.times_action_picked = [0, 0]
        self.times_action_died = [0, 0]
        self.times_action_won = [0, 0]
        self.ev = [0, 0]
        self.update_ev()
        self.chosen_actions = []
        self.odds_of_exploration = [1, 1]
        self.odds_of_exploration_threshold = odds_of_exploration_threshold
        self.action_picked_previously_by_other_node = -1

    def update_ev(self):
        if self.times_visited != np.sum(self.times_action_picked):
            test = 2
        ev = [0, 0]
        if self.times_visited > 0:
            test = 1
        for i in range(2):
            if self.times_action_picked[i] == 0:
                ev[i] = self.unknown_reward
            else:
                sum = 0
                for node in self.children[i].values():
                    sum += (max(node.ev) + max(node.times_action_won) * 20 + 1) * node.times_visited
                sum /= self.times_action_picked[i]
                ev[i] = sum
        self.ev = ev

    def update_odds_of_exploration(self):
        for i in range(2):
            if self.children[i]:
                prob = 0
                times_visited = 0
                for node in self.children[i].values():
                    prob += node.odds_of_exploration[i]
                    times_visited += 1
                prob /= times_visited
                if prob > 1:
                    test = 2

                prob *= (1 - self.times_action_died[i]/self.times_action_picked[i])
                self.odds_of_exploration[i] = prob
            elif self.times_action_died[i] == 0:
                self.odds_of_exploration[i] = 1
            else:
                self.odds_of_exploration[i] = 0
        if np.sum(self.odds_of_exploration) == 0:
            test = 2



    def just_died(self):
        chosen_action = self.chosen_actions.pop()
        self.times_visited += 1
        self.times_action_picked[chosen_action] += 1
        self.times_action_died[chosen_action] += 1
        self.update_ev()
        self.update_odds_of_exploration()

    def just_won(self):
        chosen_action = self.chosen_actions.pop()
        self.times_visited += 1
        self.times_action_picked[chosen_action] += 1
        self.times_action_won[chosen_action] += 1
        self.update_ev()
        self.update_odds_of_exploration()
        print("Winner.....")

    def update(self, prev_node):

        chosen_action = self.chosen_actions.pop()
        if not prev_node.str_bucket in self.children[chosen_action]:
            self.children[chosen_action][prev_node.str_bucket] = prev_node


        self.times_visited += 1
        self.times_action_picked[chosen_action] += 1

        self.update_ev()
        self.update_odds_of_exploration()


    def pick_action(self, strategy=MAXIMIZE_POINTS):
        choice = self.pick_action_for_other_node(strategy)
        self.chosen_actions.append(choice)
        return choice

    def pick_action_for_other_node(self, strategy=MAXIMIZE_POINTS):
        if strategy == EXPLORE:
            choice = self.explore()
        elif strategy == MAXIMIZE_POINTS:
            choice = self.maximize_points()
        else:
            raise Exception("No known strategy selected")
        return choice

    def action_chosen_by_other_node(self, chosen_action):
        self.chosen_actions.append(chosen_action)

    def maximize_points(self):
        choice = 0 if self.ev[0] >= self.ev[1] else 1
        return choice

    def explore(self):
        if abs(self.odds_of_exploration[0] - self.odds_of_exploration[1]) < self.odds_of_exploration_threshold:
            #print(f"Odds are about {self.odds_of_exploration[0]}")
            choice = self.maximize_points()
        else:
            choice = 0 if self.odds_of_exploration[0] >= self.odds_of_exploration[1] else 1
        return choice


    def should_copy_neigbor(self, strategy=MAXIMIZE_POINTS):
        if (self.times_action_picked[0] != 0) and (self.times_action_picked[1] != 0):
            return False

        if strategy == MAXIMIZE_POINTS:
            return True
        if strategy == EXPLORE:
            return abs(self.odds_of_exploration[0] - self.odds_of_exploration[1]) < self.odds_of_exploration_threshold
        else:
            raise Exception("Unknown strategy chosen")
        return False

