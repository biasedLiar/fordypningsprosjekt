from helper.strategy_names import *
import numpy as np

class TreeNodeV3:
    def __init__(self, state):
        self.children = [[], []]
        self.state = state
        raise NotImplementedError("Finish function")

    def update_ev(self):
        raise NotImplementedError("Finish function")

    def update_odds_of_exploration(self):
        raise NotImplementedError("Finish function")


    def just_died(self):
        raise NotImplementedError("Finish function")

    def just_won(self):
        raise NotImplementedError("Finish function")

    def update(self, prev_node):
        raise NotImplementedError("Finish function")

    def pick_action(self, strategy=MAXIMIZE_POINTS):
        raise NotImplementedError("Finish function")

    def pick_action_for_other_node(self, strategy=MAXIMIZE_POINTS):
        raise NotImplementedError("Finish function")

    def maximize_points(self):
        raise NotImplementedError("Finish function")

    def explore(self):
        raise NotImplementedError("Finish function")

