from classes.RunStat import *
from datetime import datetime

class MarkdownStorer:
    def __init__(self):
        self.datas = {}
        self.run_count = 0
        self.max_seeds = 0
        self.best_results = None
        self.best_run_stats = None

    def add_data_point(self, mode, data, plot_string, gw, k, seeds):
        if not mode in self.datas.keys():
            self.datas[mode] = []
        runStat = RunStat(mode, data, plot_string, gw, k, seeds)
        self.datas[mode].append(runStat)
        self.run_count += 1
        self.max_seeds = max(self.max_seeds, seeds)
        if self.best_results == None or self.best_results < data:
            self.best_results = data
            self.best_run_stats = runStat
        print("Data stored...")


    def create_markdown(self, LINUX):
        print("Starting writing to file...")
        date = datetime.today().strftime('%Y-%m-%d-%H-%M')
        title = date + f"__{self.run_count}x{self.max_seeds}seeds"
        file_name = "markdown" + ("/" if LINUX else "\\") + title

        with open(file_name, 'ab+') as f:
            f.write(f'# Tests run at {date}.\n'.encode())
            f.write(f'## {self.run_count} tests run at with {len(self.datas.keys())} types.\n'.encode())
            f.write(f'## Best avg reward {self.best_results} achieved with {self.best_run_stats.mode} gw: {self.best_run_stats.gw}, k: {self.best_run_stats.k}.\n\n'.encode())


            for mode in self.datas:
                f.write(f'\n## {mode} tests\n'.encode())

                for runStat in self.datas[mode]:
                    f.write(f'{runStat.data} reward for k:{runStat.k}, gw:{runStat.gw} over {runStat.seeds} seeds.\n'.encode())

        print("Finished writing to file.")