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
import genericClients.kMeansClient as kMeansClient

LINUX = True

SEED_COUNT = 100

GAUSSIANS = [0.515, 0.535, 0.55, 0.565, 0.58]
GAUSSIANS = [0.3, 0.55, 0.6, 0.65, 0.7]
GAUSSIANS = [0.1]
GAUSSIANS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.2]

K_VALUES = [200, 250, 300, 350, 400]
K_VALUES = [100, 150, 200, 250, 300, 350, 400, 600, 800]
K_VALUES = [1000, 1250, 1500, 1750, 2000]
K_VALUES = [100, 250, 500]
K_VALUES = [20, 50, 100, 250]
K_VALUES = [1]
K_VALUES = [50, 100, 250, 500, 1000]


EXPLORATION_RATES = [0.1]

MULTITHREADING = True
WRITE_LOGS = False


RUN_BASIC_NO_LEARN = True
RUN_BASIC_NO_LEARN = False

#-----------------------------------

RUN_KMEANS_UNWEIGHTED = True
RUN_KMEANS_UNWEIGHTED = False

RUN_KMEANS_WEIGHTED = True
RUN_KMEANS_WEIGHTED = False

#-----------------------------------

RUN_SEARCH_TREE = False
RUN_SEARCH_TREE = True

RUN_SEARCH_TREE_KMEANS = True
RUN_SEARCH_TREE_KMEANS = False

#-----------------------------------

RUN_SPECIAL_KMEANS = True
RUN_SPECIAL_KMEANS = False

RUN_WEIGHTED_SPECIAL_KMEANS = True
RUN_WEIGHTED_SPECIAL_KMEANS = False

#-----------------------------------

RUN_BASIC = True
RUN_BASIC = False

#gw0.55-250
#gw0.4-250

WRITE_MARKDOWN = True
MAKE_GRAPHS = False

SEARCH_TREE_DEPTH = 2

date_string = datetime.today().strftime('%Y-%m-%d__%H-%M')
COMMENT = f"{('search-tree-depth: ' + str(SEARCH_TREE_DEPTH) if RUN_SEARCH_TREE or RUN_SEARCH_TREE_KMEANS else '')}\n" \
          f"Testing kmeans sigmoid weighting"

PATH_PREFIX = ("fordypningsprosjekt\\" if RUN_FROM_SCRIPT else "") + f"Finals\\{date_string}\\"
MD_PATH_PREFIX = ("fordypningsprosjekt\\" if RUN_FROM_SCRIPT else "")

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                                     discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                                     exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.LEARNING_LENGTH,
                                     kmeans_episodes=kMeansClient.SLEEPING_LENGTH, weighted_kmeans=True,
                                     render_mode=kMeansClient.RENDER_MODE, game_mode=kMeansClient.GAME_MODE, k=-1,
                                     save_plot=True, ignore_kmeans=False, use_vectors=False, vector_type=1, learn=True,
                                     use_special_kmeans=False, markdownStorer=None, mode="insert_mode", write_logs=WRITE_LOGS,
                                     use_search_tree=False, search_tree_depth=-1, save_midway=False):

    if MULTITHREADING:
        config_holder = configHolder(discount_factor=discount_factor, gaussian_width=gaussian_width,
                                     exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                     kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans,
                                     render_mode=render_mode,
                                     game_mode=game_mode, k=k, save_plot=False, ignore_kmeans=ignore_kmeans,
                                     use_vectors=use_vectors,
                                     vector_type=vector_type, learn=learn, do_standardize=True,
                                     use_special_kmeans=use_special_kmeans, write_logs=write_logs,
                                     search_tree_depth=search_tree_depth, use_search_tree=use_search_tree,
                                     save_midway=save_midway)

        old_datas = []
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
                                            kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans, render_mode=render_mode,
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

    if MAKE_GRAPHS:
        plot_title = ("" if MULTITHREADING else "") + plot_title
        plotHelper.plot_with_max_min_mean_std(datas, plot_name, plot_title)
    avg = np.round(np.mean(datas), 2).item()
    std = np.round(np.std(datas), 2).item()
    if markdownStorer != None:
        markdownStorer.add_data_point(mode, avg, std, plot_name, gaussian_width, k, seed_count, kmeans_time=kmeans_time,
                                      post_kmeans_time=post_kmeans_time, total_steps=total_sleeping_steps)
        if LINUX:
            markdownStorer.update_markdown(LINUX, MD_PATH_PREFIX)
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

            if RUN_BASIC:
                if k == K_VALUES[0]:
                    print("Starting No Kmeans...")
                    path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\basic"
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__basic_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} basic plot"
                    basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                   gaussian_width=gaussian_width, k=k,
                                                                   weighted_kmeans=False, ignore_kmeans=True,
                                                                   use_vectors=False, markdownStorer=markdownStorer, mode="basic")
                    datas_list.append(basic_datas)
                    labels.append("basic")
                else:
                    datas_list.append(basic_datas)
                    labels.append("basic")

            if RUN_SPECIAL_KMEANS:
                print("Starting Special Kmeans Unweighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__special_kmeans_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k}, avg{SEED_COUNT} special_kmeans plot"
                basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                               gaussian_width=gaussian_width, k=k,
                                                               weighted_kmeans=False, ignore_kmeans=False,
                                                               use_vectors=False, use_special_kmeans=True,
                                                               markdownStorer=markdownStorer, mode="Special Kmeans")
                datas_list.append(basic_datas)
                labels.append("special_kmeans")

            if RUN_WEIGHTED_SPECIAL_KMEANS:
                print("Starting Special Kmeans Weighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.LEARNING_LENGTH}_then_{kMeansClient.SLEEPING_LENGTH}__weighted_special_kmeans_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k}, avg{SEED_COUNT} weighted_special_kmeans plot"
                basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                               gaussian_width=gaussian_width, k=k,
                                                               weighted_kmeans=True, ignore_kmeans=False,
                                                               use_vectors=False, use_special_kmeans=True,
                                                               markdownStorer=markdownStorer, mode="Special Kmeans Unweighted")
                datas_list.append(basic_datas)
                labels.append("weighted_special_kmeans")

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
            '''
            if len(labels) > 1:
                path = f"plots\\generic\\{kMeansClient.GAME_MODE}\\aggregate\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)

                types =  f"{'_weighted' if RUN_KMEANS_WEIGHTED else ''}{'_unweighted' if RUN_KMEANS_UNWEIGHTED else ''}{'_basic' if RUN_BASIC else ''}"
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}" \
                              f"{types}.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT}{types} plot"

                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=False)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}" \
                              f"{types}_std.png"
                name = fileHelper.osFormat(name, LINUX)


                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=True)
            time.sleep(2)
            '''
        end = time.time()
        print(f"\n\n\n\n{gaussian_width=}: time:{end - start}")
        time.sleep(5)
    if markdownStorer != None:
        markdownStorer.create_markdown(LINUX, MD_PATH_PREFIX)

if __name__ == '__main__':
    total_start = time.time()
    run_gaussian_k()
    total_end = time.time()
    print(f"\n\n\n\nComplete program time: {total_end-total_start}")