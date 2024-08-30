import random

##### Strategies ########
EXPLORE = 0
BALANCED = 1
MAXIMIZE_POINTS = 2
#########################

##### Constants ########
RISK_THRESHOLD = 4.5
EXPLORE_PERCENTAGE = 0.5
STRATEGY = BALANCED

########################

class CartPoleTreeNode:
    def __init__(self, state, time_step, strategy, id):
        self.name = "test"
        self.children = [None, None]
        self.state = state
        self.time_step = time_step
        self.is_final = False
        self.max_depth = 1
        self.max_unexplored_depth = 1
        self.risk_weighted_score = self.update_risk_weighted_score()
        self.holes_beneath = 2
        self.most_recent_choice = -1
        self.strategy = strategy
        self.visited = 0
        self.id = id

    def mark_final(self):
        self.max_depth = 0
        self.max_unexplored_depth = 0
        self.holes_beneath = 0
        self.is_final = True
        self.risk_weighted_score = 0


    def update(self):
        if self.children[0] != None and self.children[1] != None:
            self.max_depth = max(self.children[0].max_depth, self.children[1].max_depth) + 1
            self.max_unexplored_depth = max(self.children[0].max_unexplored_depth, self.children[1].max_unexplored_depth)
            if self.max_unexplored_depth != 0:
                self.max_unexplored_depth += 1
            self.holes_beneath = self.children[0].holes_beneath + self.children[1].holes_beneath
        else:
            i = 0 if self.children[0] != None else 1
            self.max_depth = self.children[i].max_depth + 1
            self.max_unexplored_depth = self.children[i].max_unexplored_depth + 1 if self.children[i].max_unexplored_depth != 0 else 1
            self.holes_beneath = self.children[i].holes_beneath + 1
        self.visited += 1
        self.risk_weighted_score = self.update_risk_weighted_score()


    def register_move(self, new_state, new_time_step, direction):
        if self.children[direction] != None:
            self.children[direction].strategy = self.strategy
        else:
            id = self.id + str(self.most_recent_choice)
            self.children[direction] = CartPoleTreeNode(new_state, new_time_step, self.strategy, id)
        return self.children[direction]

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
                choice = self.balanced()
        elif self.strategy == EXPLORE:
            return self.explore()
        else:
            choice = 0
        self.most_recent_choice = choice
        return choice

    def states_are_equal(self, obs1, obs2):
        if len(obs1) != len(obs2):
            return False
        for i in range(len(obs1)):
            if obs1[i] != obs2[i]:
                return False
        return True

    def update_risk_weighted_score(self):
        return 0 if self.is_final else max(self.max_depth, (self.max_unexplored_depth + RISK_THRESHOLD) if self.max_unexplored_depth != 0 else 0)

    def explore(self):
        left_explorable = self.children[0] == None or self.children[0].max_unexplored_depth != 0
        right_explorable = self.children[1] == None or self.children[1].max_unexplored_depth != 0
        if left_explorable and not right_explorable:
            return 0
        if right_explorable and not left_explorable:
            return 1
        return self.balanced()

    def balanced(self):
        left_score = 1 + RISK_THRESHOLD if self.children[0] == None else self.children[0].risk_weighted_score
        right_score = 1 + RISK_THRESHOLD if self.children[1] == None else self.children[1].risk_weighted_score
        if left_score == right_score:
            return 0
        return 0 if left_score > right_score else 1

    def maximize_points(self):
        left_score = 1+RISK_THRESHOLD if self.children[0] == None else self.children[0].risk_weighted_score
        right_score = 1+RISK_THRESHOLD if self.children[1] == None else self.children[1].risk_weighted_score
        if left_score == right_score:
            return 0
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












