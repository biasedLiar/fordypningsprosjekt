import time

from classes.RunStat import *
from datetime import datetime
import helper.fileHelper as fileHelper
import numpy as np
import os

class MarkdownStorer:
    def __init__(self, Ks = None, GWs = None, segments=None, learn_length=None, comment=None):
        self.datas = {}
        self.run_count = 0
        self.max_seeds = 0
        self.best_results = None
        self.best_run_stats = None
        self.Ks = Ks
        self.GWs = GWs
        self.segments = segments
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.comment = comment
        self.learn_length = learn_length
        self.date = datetime.today().strftime('%Y-%m-%d__%H-%M')

    def add_data_point(self, mode, avg, std, plot_string, gw, k, seeds, segments=1):
        if not mode in self.datas.keys():
            self.datas[mode] = []
        runStat = RunStat(mode, avg, std, plot_string, gw=gw, k=k, seeds=seeds, segments=segments)
        self.datas[mode].append(runStat)
        self.run_count += 1
        self.max_seeds = max(self.max_seeds, seeds)
        if self.best_results == None or self.best_results < avg:
            self.best_results = avg
            self.best_run_stats = runStat
        print("Data stored...")


    def update_markdown(self, LINUX, PREFIX):
        if self.run_count > 1 and time.time() - self.last_update_time < 60:
            return
        self.create_markdown(LINUX, PREFIX, is_update=True)
        self.last_update_time = time.time()


    def create_markdown(self, LINUX, PREFIX, is_update=False):
        end_time = time.time()
        minutes = round((end_time-self.start_time)/60)
        hours = np.floor(minutes / 60)
        minutes = minutes % 60
        print("Starting writing to file...")
        title = self.date
        if is_update:
            file_name = f"{PREFIX}markdown\\" + title + f".md"
        else:
            file_name = f"{PREFIX}results\\" + title + f"__{self.run_count}x{self.max_seeds}"+ f".md"

        file_name = fileHelper.osFormat(file_name, LINUX)


        with open(file_name, 'wb+') as f:
            date2 = datetime.today().strftime('%Y.%m.%d %H:%M')
            if is_update:
                f.write(f'# Tests updated at {date2} after {int(hours)} hours and {minutes} minutes.\n'.encode())
            else:
                f.write(f'# Tests finished at {date2} after {int(hours)} hours and {minutes} minutes.\n'.encode())
            if self.comment != None:
                f.write(f'# {self.comment}\n'.encode())
            f.write(f'## {self.run_count} tests run at with {len(self.datas.keys())} types.\n'.encode())

            if self.learn_length != None:
                f.write(f'# Learn length: {self.learn_length}.\n'.encode())
            if self.segments != None:
                f.write(f'# Segments: {self.segments}.\n'.encode())

            if self.Ks != -1:
                f.write(f'# K values of {str(self.Ks)} tested.\n'.encode())

            if self.GWs != None:
                f.write(f'# GW values of {str(self.GWs)} tested.\n'.encode())

            f.write(
                f'\n{self.best_run_stats.get_stat_string()}'.encode())


            for mode in self.datas:
                f.write(f'\n## {mode} tests\n'.encode())

                best_run_stat = None
                for runStat in self.datas[mode]:
                    if best_run_stat == None or runStat.avg > best_run_stat.avg:
                        best_run_stat = runStat

                f.write(
                    f'### {best_run_stat.get_stat_string()}'.encode())

                for runStat in self.datas[mode]:
                    f.write(runStat.get_stat_string().encode())

            f.write(f'\n# Data unformatted:\n\n\n'.encode())

            for mode in self.datas:
                f.write(f'\n## {mode} tests\navg:\n'.encode())

                for runStat in self.datas[mode]:
                    f.write(
                        f'{runStat.avg}\n'.encode())

                f.write(f'\nstd:\n'.encode())
                for runStat in self.datas[mode]:
                    f.write(
                        f'{runStat.std}\n'.encode())

        print("Finished writing to file.")

