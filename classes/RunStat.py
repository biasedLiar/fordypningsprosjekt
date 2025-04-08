class RunStat:
    def __init__(self, mode, data, plot_string, gw, k, seeds, segments=1):
        self.mode = mode
        self.data= data
        self.plot_string = plot_string
        self.gw = gw
        self.k = k
        self.seeds = seeds
        self.segments = segments

    def get_stat_string(self):
        myString = f'{self.data} reward for k:{self.k}, gw:{self.gw}'
        if self.segments != 1:
            myString += f", with {self.segments} segments"
        myString += f" over {self.seeds} seeds.\n\n"
        return myString

