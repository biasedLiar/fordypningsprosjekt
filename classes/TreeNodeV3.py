from helper.strategy_names import *
import numpy as np
from classes.State import *

class TreeNodeV3:
    def __init__(self, state, ev_multiplier=0.8):
        self.children = [-1, -1]
        self.state = state
        self.ev = 0
        self.ev_multiplier = ev_multiplier
        self.most_recent_action = -1
        self.times_visited = 0

    def update_ev(self):
        raise NotImplementedError("Finish function")

    def update_odds_of_exploration(self):
        raise NotImplementedError("Finish function")


    def just_died(self):
        raise NotImplementedError("Finish function")
        self.times_visited += 1
        self.ev = 100

    def just_won(self):
        raise NotImplementedError("Finish function")
        self.times_visited += 1
        self.ev = 0

    def update(self, prev_node):
        raise NotImplementedError("Finish function")
        self.times_visited += 1
        self.ev = prev_node.ev * self.ev_multiplier

    def pick_action(self, strategy=MAXIMIZE_POINTS):
        raise NotImplementedError("Finish function")

    def pick_action_for_other_node(self, strategy=MAXIMIZE_POINTS):
        raise NotImplementedError("Finish function")

    def maximize_points(self):
        raise NotImplementedError("Finish function")

    def explore(self):
        raise NotImplementedError("Finish function")






    def __str__(self):
        return "State: " + str(self.state) + ", ev: " + str(self.ev)

