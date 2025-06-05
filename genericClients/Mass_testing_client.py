import matplotlib.pyplot as plt
import os
import sys

import numpy as np

sys_path = '/home/eliaseb/PycharmProjects/master-thesis'

sys.path.append(sys_path)
num_arguments = len(sys.argv)
RUN_FROM_SCRIPT = (num_arguments > 1)

from multiprocessing import Pool, cpu_count
from classes.genericBasic import *
from classes.configHolder import *
import helper.fileHelper as fileHelper
import helper.plotHelper as plotHelper
from classes.MarkdownStorer import *
from classes.RunStat import *
import time
import genericClients.kMeansClient as kMeansClient

LINUX = True

SEED_COUNT = 100

GAUSSIANS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
GAUSSIANS = [0.2]

K_VALUES = [50, 100, 250, 500, 1000]
K_VALUES = [50]


EXPLORATION_RATES = [0.1]

MULTITHREADING = True
WRITE_LOGS = False


RUN_BASIC_NO_LEARN = True

#-----------------------------------

RUN_KMEANS_UNWEIGHTED = True

RUN_KMEANS_WEIGHTED = False

#-----------------------------------

RUN_SEARCH_TREE = False

RUN_SEARCH_TREE_KMEANS = True

#-----------------------------------

RUN_BASIC = False

WRITE_MARKDOWN = True
MAKE_GRAPHS = False

SEARCH_TREE_DEPTH = 4

date_string = datetime.today().strftime('%Y-%m-%d__%H-%M')
COMMENT = f"{('search-tree-depth: ' + str(SEARCH_TREE_DEPTH) if RUN_SEARCH_TREE or RUN_SEARCH_TREE_KMEANS else '')}\n" \
          f"Testing kmeans --- weighting"

PATH_PREFIX = ("master-thesis\\" if RUN_FROM_SCRIPT else "") + f"Finals\\{date_string}\\"
MD_PATH_PREFIX = ("master-thesis\\" if RUN_FROM_SCRIPT else "")

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                                     gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                                     exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.LEARNING_LENGTH,
                                     kmeans_episodes=kMeansClient.SLEEPING_LENGTH, weighted_kmeans=True,
                                     render_mode=kMeansClient.RENDER_MODE, game_mode=kMeansClient.GAME_MODE, k=-1,
                                     save_plot=True, ignore_kmeans=False, use_vectors=False, vector_type=1, learn=True,
                                     use_special_kmeans=False, markdownStorer=None, mode="insert_mode", write_logs=WRITE_LOGS,
                                     use_search_tree=False, search_tree_depth=-1, save_midway=False):

    if MULTITHREADING:
        config_holder = configHolder(gaussian_width=gaussian_width,
                                     exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                     kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans,
                                     render_mode=render_mode,
                                     game_mode=game_mode, k=k, ignore_kmeans=ignore_kmeans,
                                     learn=learn, do_standardize=True,
                                     write_logs=write_logs,
                                     search_tree_depth=search_tree_depth, use_search_tree=use_search_tree,
                                     save_midway=save_midway)
        datas = []
        kmeans_time = []
        post_kmeans_time = []
        total_sleeping_steps = []
        learn_datas = []
        num_nodes = []
        with Pool((cpu_count() - 1)) as p:
            old_datas = p.map(config_holder.run_with_seed, range(seed_count))
        for data in old_datas:
            datas.append(data[0])
            kmeans_time.append(data[1])
            post_kmeans_time.append(data[2])
            total_sleeping_steps.append(data[3])
            learn_datas.append(data[4])
            num_nodes.append(data[5])
    else:
        datas = []
        kmeans_time = []
        post_kmeans_time = []
        total_sleeping_steps = []
        for seed in range(seed_count):
            data = kMeansClient.run_program(seed=seed, gaussian_width=gaussian_width,
                                            exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                            eval_length=kmeans_episodes, weighted_kmeans=weighted_kmeans, render_mode=render_mode,
                                            game_mode=game_mode, k=k, save_plot=False, ignore_kmeans=ignore_kmeans, use_vectors=use_vectors,
                                            vector_type=vector_type, learn=learn, do_standardize=True, use_special_kmeans=use_special_kmeans,
                                            write_logs=write_logs, use_search_tree=use_search_tree, search_tree_depth=search_tree_depth,
                                            save_midway=save_midway)
            datas.append(data[0])
            kmeans_time.append(data[1])
            post_kmeans_time.append(data[2])
            total_sleeping_steps.append(data[3])

    datas = np.asarray(datas)
    kmeans_time = np.round(np.mean(np.asarray(kmeans_time)), 2).item()
    post_kmeans_time = np.round(np.mean(np.asarray(post_kmeans_time)), 2).item()
    total_sleeping_steps = np.round(np.mean(np.asarray(total_sleeping_steps)), 2).item()
    num_node_mean = np.round(np.mean(np.asarray(num_nodes)), 2).item()

    if MAKE_GRAPHS:
        plot_title = ("" if MULTITHREADING else "") + plot_title
        plotHelper.plot_with_max_min_mean_std(datas, plot_name, plot_title)
    avg = np.round(np.mean(datas), 2).item()
    std = np.round(np.std(datas), 2).item()


    if markdownStorer != None:
        markdownStorer.comment += f"\n{num_node_mean=}\n"
        markdownStorer.add_data_point(mode, avg, std, plot_name, gaussian_width, k, seed_count, kmeans_time=kmeans_time,
                                      post_kmeans_time=post_kmeans_time, total_steps=total_sleeping_steps)
        if LINUX:
            markdownStorer.update_markdown(LINUX, MD_PATH_PREFIX)

    if False:
        learn_data = np.array(learn_datas)
        learn_data = learn_data.transpose()
        means = np.mean(learn_data, axis=1)
        stds = np.std(learn_data, axis=1)
        timesteps = np.arange(learn_data.shape[0])
        plt.plot(timesteps, means, label='Mean', color='blue')

        plt.fill_between(timesteps, means - stds, means + stds, alpha=0.3, color='red', label='±1 Std Dev')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Mean and Standard Deviation Episode Length for σ=0.2')
        plt.legend()
        plt.grid(True)
        path = f"{PATH_PREFIX}special"
        fileHelper.createDirIfNotExist(path, linux=LINUX)
        name = path + f"\\learning_plots.png"
        name = fileHelper.osFormat(name, LINUX)
        plt.savefig(name)
        print(f"saving to {name=}")
        plt.clf()
    return datas



def run_gaussian_k():
    if WRITE_MARKDOWN:
        markdownStorer = MarkdownStorer(Ks=K_VALUES, GWs=GAUSSIANS, learn_length=kMeansClient.LEARNING_LENGTH, comment=COMMENT)
    else:
        markdownStorer = None
    for k in K_VALUES:
        print(f"Starting k: {k}.")
        start = time.time()
        for gaussian_width in GAUSSIANS:
            print(f"Starting gw: {gaussian_width}.")
            start_2 = time.time()
            labels = []
            datas_list = []
            if RUN_KMEANS_UNWEIGHTED:
                print("Starting Kmeans Unweighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                if MAKE_GRAPHS:
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__unweighted_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} unweighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width,
                                                         k=k, weighted_kmeans=False, use_vectors=False,
                                                         markdownStorer=markdownStorer, mode="Kmeans Unweighted")
                datas_list.append(datas)
                labels.append("unweighted")


            if RUN_KMEANS_WEIGHTED:
                print("Starting Kmeans Weighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                if MAKE_GRAPHS:
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__weighted_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} weighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width, k=k, weighted_kmeans=True,
                                                         use_vectors=False, markdownStorer=markdownStorer, mode="Kmeans Weighted")
                datas_list.append(datas)
                labels.append("weighted")

            if RUN_SEARCH_TREE_KMEANS:
                print("Starting Search Tree KMeans...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\search_tree{gaussian_width}g\\{k}k"
                if MAKE_GRAPHS:
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__kmeans_search_tree_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} search-tree kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, gaussian_width=gaussian_width,
                                                         k=k, weighted_kmeans=False, use_vectors=False,
                                                         markdownStorer=markdownStorer, mode="Kmeans search tree",
                                                         search_tree_depth=SEARCH_TREE_DEPTH, save_midway=True,
                                                         learn=False, ignore_kmeans=False, use_search_tree=True)
                datas_list.append(datas)
                labels.append("kmeans search tree")

            if RUN_SEARCH_TREE:
                if k == K_VALUES[0]:
                    print("Starting Search Tree...")
                    path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\search_tree"
                    if MAKE_GRAPHS:
                        fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__tree_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} tree plot"
                    basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                   gaussian_width=gaussian_width,
                                                                   weighted_kmeans=False, ignore_kmeans=True,
                                                                   use_vectors=False, markdownStorer=markdownStorer,
                                                                   mode="Search Tree", use_search_tree=True,
                                                                   search_tree_depth=SEARCH_TREE_DEPTH, save_midway=True,
                                                                   learn=False)
                    datas_list.append(basic_datas)
                    labels.append("tree")
                else:
                    datas_list.append(basic_datas)
                    labels.append("tree")

            if RUN_BASIC_NO_LEARN:
                if k == K_VALUES[0]:
                    print("Starting Sleeping No Kmeans...")
                    path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\basic"
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__basic_no_learn_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} basic-no_learn plot"
                    basic_NL_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                    gaussian_width=gaussian_width, k=-1,
                                                                    weighted_kmeans=False, ignore_kmeans=True,
                                                                    use_vectors=False, learn=False,
                                                                    markdownStorer=markdownStorer, mode="Sleeping No Kmeans")
                    datas_list.append(basic_NL_datas)
                    labels.append("basic no-learn")
                else:
                    datas_list.append(basic_NL_datas)
                    labels.append("basic no-learn")

            end_2 = time.time()
            print(f"\n\n{gaussian_width=}, {k=}: time:{end_2 - start_2}")
        end = time.time()
        print(f"\n\n\n\n{gaussian_width=}: time:{end - start}")
    if markdownStorer != None:
        markdownStorer.create_markdown(LINUX, MD_PATH_PREFIX)

if __name__ == '__main__':
    total_start = time.time()
    run_gaussian_k()
    total_end = time.time()
    print(f"\n\n\n\nComplete program time: {total_end-total_start}")