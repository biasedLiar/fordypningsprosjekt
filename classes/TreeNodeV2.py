from helper.strategy_names import *

class TreeNodeV2:
    def __init__(self, bucket):
        self.bucket = bucket

    def pick_action(self, strategy):
        if strategy == EXPLORE:
            choice = self.explore()
        else:
            raise Exception("No known strategy selected")
        return choice

    def explore(self):
        return 0

