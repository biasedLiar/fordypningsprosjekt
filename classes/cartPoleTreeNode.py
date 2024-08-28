import random

EXPLORE = 0
MAXIMIZE_POINTS = 1

RISK_THRESHOLD = 4.5

class CartPoleTreeNode:
    def __init__(self, state, time_step, is_final):
        self.name = "test"
        self.children = [None, None]
        self.state = state
        self.time_step = time_step
        self.max_depth = 0 if is_final else 1
        self.max_unexplored_depth = 0 if is_final else 1
        self.risk_weighted_score = self.update_risk_weighted_score()
        self.holes_beneath = 0 if is_final else 2
        self.strategy = EXPLORE

    def update(self):
        if self.children[0] != None and self.children[0] != None:
            self.max_depth = max(self.children[0].max_depth, self.children[0].max_depth) + 1
            self.max_unexplored_depth = max(self.children[0].max_depth, self.children[0].max_depth)
            if self.max_unexplored_depth != 0:
                self.max_unexplored_depth += 1
            self.holes_beneath = self.children[0].holes_beneath + self.children[0].holes_beneath
        else:
            i = 0 if self.children[0] != None else 1
            self.max_depth = self.children[i].max_depth + 1
            self.max_unexplored_depth = self.children[i].max_unexplored_depth + 1 if self.children[i].max_unexplored_depth != 0 else 1
            self.holes_beneath = self.children[i].holes_beneath + 1
        self.risk_weighted_score = self.update_risk_weighted_score()


    def make_move(self, new_state, new_time_step, is_final, direction):
        if self.children[direction] != None:
            if new_state != self.children[direction].state or is_final != self.children[direction].is_final:
                print("States are not matching")
        else:
            self.children[direction] = CartPoleTreeNode(new_state, new_time_step, is_final)
        return self.children[direction]

    def choose_dir(self, new_strategy=None):
        if new_strategy != None:
            self.strategy = new_strategy
        if self.strategy == MAXIMIZE_POINTS:
            return self.maximize_points()
        return 0

    def update_risk_weighted_score(self):
        return max(self.max_depth, self.max_unexplored_depth + RISK_THRESHOLD)

    def maximize_points(self):
        if self.children[0].risk_weighted_score == self.children[1].risk_weighted_score:
            return random.getrandbits(1)
        return 0 if self.children[0].risk_weighted_score > self.children[1].risk_weighted_score else 1








