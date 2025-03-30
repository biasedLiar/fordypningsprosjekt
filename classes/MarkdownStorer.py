import time

from classes.RunStat import *
from datetime import datetime
import helper.fileHelper as fileHelper
import numpy as np

class MarkdownStorer:
    def __init__(self, Ks = None, GWs = None, learn_length=None):
        self.datas = {}
        self.run_count = 0
        self.max_seeds = 0
        self.best_results = None
        self.best_run_stats = None
        self.Ks = Ks
        self.GWs = GWs
        self.start_time = time.time()
        self.comment = comment
        self.learn_length = learn_length

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



    def create_markdown(self, LINUX, PREFIX):
        end_time = time.time()
        minutes = round((end_time-self.start_time)/60)
        hours = np.floor(minutes / 60)
        minutes = minutes % 60
        print("Starting writing to file...")
        date = datetime.today().strftime('%Y-%m-%d__%H-%M')
        title = date + f"__{self.run_count}x{self.max_seeds}seeds.md"
        file_name = f"{PREFIX}markdown\\" + title


        file_name = fileHelper.osFormat(file_name, LINUX)

        with open(file_name, 'ab+') as f:
            date2 = datetime.today().strftime('%Y.%m.%d %H:%M')
            f.write(f'# Tests finished at {date2} after {hours} hours and {minutes} minutes.\n'.encode())
            if self.comment != None:
                f.write(f'# {self.comment}\n'.encode())
            f.write(f'## {self.run_count} tests run at with {len(self.datas.keys())} types.\n'.encode())

            if self.learn_length != None:
                f.write(f'# Learn length: {self.learn_length}.\n'.encode())


            if self.Ks != None:
                f.write(f'# K values of {str(self.Ks)} tested.\n'.encode())

            if self.GWs != None:
                f.write(f'# GW values of {str(self.GWs)} tested.\n'.encode())

            f.write(
                f'\nBest avg reward {self.best_results} achieved with {self.best_run_stats.mode} gw: {self.best_run_stats.gw}, k: {self.best_run_stats.k} over {self.best_run_stats.seeds} seeds.\n\n'.encode())


            for mode in self.datas:
                f.write(f'\n## {mode} tests\n'.encode())

                best_run_stat = None
                for runStat in self.datas[mode]:
                    if best_run_stat == None or runStat.data > best_run_stat.data:
                        best_run_stat = runStat

                f.write(
                    f'### Best avg reward: {best_run_stat.data} for {best_run_stat.mode} gw: {best_run_stat.gw}, k: {best_run_stat.k} over {best_run_stat.seeds} seeds.\n\n'.encode())

                for runStat in self.datas[mode]:
                    f.write(
                        f'{runStat.data} reward for k:{runStat.k}, gw:{runStat.gw} over {runStat.seeds} seeds.\n\n'.encode())

            f.write(f'\n# Data unformatted:\n\n\n'.encode())

            for mode in self.datas:
                f.write(f'\n## {mode} tests\n'.encode())

                for runStat in self.datas[mode]:
                    f.write(
                        f'{runStat.data}\n'.encode())

        print("Finished writing to file.")