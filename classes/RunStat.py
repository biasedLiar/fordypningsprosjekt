class RunStat:
    def __init__(self, mode, avg, std, plot_string, gw=-1, k=-1, seeds=100, segments=1):
        self.mode = mode
        self.avg = avg
        self.std = std
        self.plot_string = plot_string
        self.gw = gw
        self.k = k
        self.seeds = seeds
        self.segments = segments

    def get_stat_string(self):
        myString = f'{self.avg} reward, {self.std} std, '
        if self.k != -1:
            myString += f"for k:{self.k}, "
        if self.gw != -1:
            myString += f"gw:{self.gw}, "
        if self.segments != 1:
            myString += f"with {self.segments} segments "
        myString += f"over {self.seeds} seeds.\n\n"
        return myString

