import numpy as np

class RunStat:
    def __init__(self, mode, avg, std, plot_string, gw=-1, k=-1, seeds=100, segments=1, kmeans_time=-1,
                 post_kmeans_time=-1, total_sleeping_steps=-1):
        self.mode = mode
        self.avg = avg
        self.std = std
        self.plot_string = plot_string
        self.gw = gw
        self.k = k
        self.seeds = seeds
        self.segments = segments
        self.kmeans_time = kmeans_time
        self.post_kmeans_time = post_kmeans_time
        self.total_sleeping_steps = total_sleeping_steps
        self.steps_per_second = -1 if post_kmeans_time == -1 else np.round(total_sleeping_steps/post_kmeans_time, 2)

    def get_stat_string(self):
        myString = f'{self.avg} reward, {self.std} std, '
        if self.k != -1:
            myString += f"for k:{self.k}, "
        if self.gw != -1:
            myString += f"gw:{self.gw}, "
        if self.segments != 1:
            myString += f"with {self.segments} segments "
        myString += f"over {self.seeds} seeds. "
        if self.steps_per_second != -1:
            myString += f" {self.steps_per_second} steps/second after learning"
        if self.kmeans_time != -1:
            myString += f" and {self.kmeans_time} seconds for midway."
        myString += f"\n\n"
        return myString

