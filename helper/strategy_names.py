EXPLORE = 0
BALANCED = 1
MAXIMIZE_POINTS = 2

def name_of_strategy(strat):
    match strat:
        case 0:
            return "Explore"
        case 1:
            return "Balanced"
        case 2:
            return "Maximize Points"
        case _:
            return "Unkown strategy"
