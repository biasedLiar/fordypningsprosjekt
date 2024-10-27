class State:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        if isinstance(other, State):
            return self.state == other.state
        return NotImplemented
