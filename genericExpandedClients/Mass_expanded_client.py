import matplotlib.pyplot as plt
import os
import sys


#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.getcwd()

print(dir_path)
print("\n\n\n")

sys.path.append('/home/eliaseb/PycharmProjects/fordypningsprosjekt')
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
import genericExpandedClients.kMeansExpandedClient as kMeansClient

LINUX = True

SEED_COUNT = 100


GAUSSIANS = [0.4]
GAUSSIANS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GAUSSIANS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

K_VALUES = [20, 50, 100, 250]
K_VALUES = [500, 1000]
K_VALUES = [50, 100, 250, 500, 1000]
K_VALUES = [50]

EXPLORATION_RATES = [0.1]

MULTITHREADING=True

SEGMENTS = [2, 3, 4]
SEGMENTS = [5]

EXPANDER_GAUSSIAN = 1.0

SEARCH_TREE_DEPTH = 3

RUN_BASIC_NO_LEARN = True
RUN_KMEANS_UNWEIGHTED = False
RUN_KMEANS_WEIGHTED = False
RUN_SEARCH_TREE = False
RUN_SEARCH_TREE_KMEANS = False


WRITE_MARKDOWN = True
MAKE_GRAPHS = False

WRITE_LOGS = False
COSINE_SIMILARITY = False

PATH_PREFIX = ("fordypningsprosjekt\\expanded_" if RUN_FROM_SCRIPT else "expanded_")

MD_PATH_PREFIX = ("fordypningsprosjekt\\" if RUN_FROM_SCRIPT else "") + "expanded_"

COMMENT = f"Generations of training: {kMeansClient.LEARNING_LENGTH}\n" \
          f"{EXPANDER_GAUSSIAN=}\n" \
          f"{COSINE_SIMILARITY=}\n" \
          f"{SEARCH_TREE_DEPTH=}" \
          f"\n\nNew reward schema\n" \
          f"linear weighting"

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                                     discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=0.3,
                                     exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.LEARNING_LENGTH,
                                     kmeans_episodes=kMeansClient.SLEEPING_LENGTH, weighted_kmeans=True, render_mode=kMeansClient.RENDER_MODE,
                                     game_mode=kMeansClient.GAME_MODE, k=None, ignore_kmeans=False,
                                     learn=True, markdownStorer=None,
                                     mode="insert_mode", write_logs=WRITE_LOGS, segments=1,
                                     use_search_tree=False, search_tree_depth=-1, save_midway=False):

    if MULTITHREADING:
        config_holder = configHolder(gaussian_width=gaussian_width,
                                     exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                     kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans,
                                     render_mode=render_mode,
                                     game_mode=game_mode, k=k, ignore_kmeans=ignore_kmeans,
                                     learn=learn,
                                     write_logs=write_logs, segments=segments,
                                     use_expanded=True, expander_gaussian=EXPANDER_GAUSSIAN,
                                     search_tree_depth=search_tree_depth, use_search_tree=use_search_tree,
                                     save_midway=save_midway)
        datas = []
        kmeans_time = []
        post_kmeans_time = []
        total_sleeping_steps = []
        with Pool((cpu_count() - 1)) as p:
            old_datas = p.map(config_holder.run_with_seed, range(seed_count))
        for data in old_datas:
            datas.append(data[0])
            kmeans_time.append(data[1])
            post_kmeans_time.append(data[2])
            total_sleeping_steps.append(data[3])
    else:
        datas = []
        kmeans_time = []
        post_kmeans_time = []
        total_sleeping_steps = []
        for seed in range(seed_count):
            data = kMeansClient.run_program(seed=seed, discount_factor=discount_factor, gaussian_width=gaussian_width,
                                            exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                            eval_length=kmeans_episodes, weighted_kmeans=weighted_kmeans, render_mode=render_mode,
                                            game_mode=game_mode, k=k,  ignore_kmeans=ignore_kmeans,
                                            learn=learn,
                                            write_logs=write_logs, segments=segments, expander_gaussian=EXPANDER_GAUSSIAN,
                                            use_search_tree=use_search_tree,
                                            search_tree_depth=search_tree_depth, save_midway=save_midway)
            datas.append(data[0])
            kmeans_time.append(data[1])
            post_kmeans_time.append(data[2])
            total_sleeping_steps.append(data[3])

    datas = np.asarray(datas)
    kmeans_time = np.round(np.mean(np.asarray(kmeans_time)), 2).item()
    post_kmeans_time = np.round(np.mean(np.asarray(post_kmeans_time)), 2).item()
    total_sleeping_steps = np.round(np.mean(np.asarray(total_sleeping_steps)), 2).item()


    plot_title = ("" if MULTITHREADING else "") + plot_title
    if MAKE_GRAPHS:
        plotHelper.plot_with_max_min_mean_std(datas, plot_name, plot_title)
    avg = np.round(np.mean(datas), 2).item()
    std = np.round(np.std(datas), 2).item()
    if markdownStorer != None:
        markdownStorer.add_data_point(mode, avg, std, plot_name, gaussian_width, k, seed_count, segments=segments, kmeans_time=kmeans_time,
                                      post_kmeans_time=post_kmeans_time, total_steps=total_sleeping_steps)
        markdownStorer.update_markdown(LINUX, PATH_PREFIX)
    return datas



def run_gaussian_k():
    if WRITE_MARKDOWN:
        markdownStorer = MarkdownStorer(Ks=K_VALUES, learn_length=kMeansClient.LEARNING_LENGTH, comment=COMMENT, segments=SEGMENTS)
    else:
        markdownStorer = None
    for k in K_VALUES:
        print(f"Starting k: {k}.")
        start = time.time()
        for segment in SEGMENTS:
            print(f"Starting segments: {segment}.")
            start_2 = time.time()
            labels = []
            datas_list = []
            if RUN_KMEANS_UNWEIGHTED:

                for gaussian_width in GAUSSIANS:
                    print("Starting Kmeans Unweighted...")
                    path = f"{PATH_PREFIX}mplots\\expanded\\{kMeansClient.GAME_MODE}\\{segment}s\\{k}k"
                    if MAKE_GRAPHS:
                        fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__unweighted_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"{segment}-segments, k={k} avg{SEED_COUNT} unweighted-kmeans plot"
                    datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                             gaussian_width=gaussian_width,
                                                             k=k, weighted_kmeans=False,
                                                             markdownStorer=markdownStorer, mode="Kmeans Unweighted",
                                                             segments=segment)
                    datas_list.append(datas)
                    labels.append("unweighted")

            if RUN_KMEANS_WEIGHTED:

                for gaussian_width in GAUSSIANS:
                    print("Starting Kmeans Weighted...")
                    path = f"{PATH_PREFIX}mplots\\expanded\\{kMeansClient.GAME_MODE}\\{segment}s\\{k}k"
                    if MAKE_GRAPHS:
                        fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__weighted_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"{segment}-segments, k={k} avg{SEED_COUNT} weighted-kmeans plot"
                    datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, k=k, weighted_kmeans=True,
                                                             gaussian_width=gaussian_width, learn=False,
                                                             markdownStorer=markdownStorer, mode="Kmeans Weighted",
                                                             segments=segment)
                    datas_list.append(datas)
                    labels.append("weighted")

            if RUN_SEARCH_TREE:
                if k == K_VALUES[0]:
                    for gaussian_width in GAUSSIANS:
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
                                                                       markdownStorer=markdownStorer,
                                                                       mode="Search Tree", use_search_tree=True,
                                                                       search_tree_depth=SEARCH_TREE_DEPTH,
                                                                       save_midway=True,
                                                                       learn=False, segments=segment)
                        datas_list.append(basic_datas)
                        labels.append("tree")

            if RUN_SEARCH_TREE_KMEANS:
                for gaussian_width in GAUSSIANS:
                    print("Starting Search Tree KMeans...")
                    path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\search_tree_kmeans"
                    if MAKE_GRAPHS:
                        fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__tree_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} tree plot"
                    basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                   gaussian_width=gaussian_width,
                                                                   weighted_kmeans=False, ignore_kmeans=False,
                                                                   k=k,
                                                                   markdownStorer=markdownStorer,
                                                                   mode="Search Tree KMeans", use_search_tree=True,
                                                                   search_tree_depth=SEARCH_TREE_DEPTH, save_midway=True,
                                                                   learn=False, segments=segment)
                    datas_list.append(basic_datas)
                    labels.append("tree")

            if RUN_BASIC_NO_LEARN:
                if k == K_VALUES[0]:
                    for gaussian_width in GAUSSIANS:
                        print("Starting Sleeping No Kmeans...")
                        path = f"{PATH_PREFIX}mplots\\expanded\\{kMeansClient.GAME_MODE}\\{segment}s\\basic"
                        fileHelper.createDirIfNotExist(path, linux=LINUX)
                        name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__basic_no_learn_plot.png"
                        name = fileHelper.osFormat(name, LINUX)

                        title = f"{segment}-segments, avg{SEED_COUNT} basic-no_learn plot"
                        basic_NL_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                        gaussian_width=gaussian_width,
                                                                        weighted_kmeans=False, ignore_kmeans=True,
                                                                        learn=False,
                                                                        markdownStorer=markdownStorer, mode="Sleeping No Kmeans",
                                                                        segments=segment)
                    datas_list.append(basic_NL_datas)
                    labels.append("basic no-learn")
                else:
                    datas_list.append(basic_NL_datas)
                    labels.append("basic no-learn")

            end_2 = time.time()
            print(f"\n\n{segment=}, {k=}: time:{end_2 - start_2}")
            if len(labels) > 1:
                path = f"plots\\expanded\\{kMeansClient.GAME_MODE}\\aggregate\\{segment}s\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)

                types =  f"{'_weighted' if RUN_KMEANS_WEIGHTED else ''}{'_unweighted' if RUN_KMEANS_UNWEIGHTED else ''}{'_vector' if RUN_KMEANS_VECTOR else ''}{'_basic' if RUN_BASIC else ''}"
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}" \
                              f"{types}.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"{segment}-segments, k={k} avg{SEED_COUNT}{types} plot"

                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=False)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}" \
                              f"{types}_std.png"
                name = fileHelper.osFormat(name, LINUX)


                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=True)
        end = time.time()
        print(f"\n\n\n\n{k=}: time:{end - start}")
    if markdownStorer != None:
        markdownStorer.create_markdown(LINUX, MD_PATH_PREFIX)

if __name__ == '__main__':
    total_start = time.time()
    run_gaussian_k()
    total_end = time.time()
    print(f"\n\n\n\nComplete program time: {total_end-total_start}")