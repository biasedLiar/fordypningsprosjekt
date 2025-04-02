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

SEED_COUNT = 1

COMMENT = None

GAUSSIANS = [0.515, 0.535, 0.55, 0.565, 0.58]
GAUSSIANS = [0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
GAUSSIANS = [0.3, 0.55, 0.6, 0.65, 0.7]
GAUSSIANS = [0.55]

K_VALUES = [200, 250, 300, 350, 400]
K_VALUES = [20, 50, 100, 250]
K_VALUES = [100, 150, 200, 250, 300, 350, 400, 600, 800]
K_VALUES = [1000, 1250, 1500, 1750, 2000]
K_VALUES = [100]

EXPLORATION_RATES = [0.1]

MULTITHREADING=False


RUN_KMEANS_UNWEIGHTED = True
RUN_KMEANS_UNWEIGHTED = False

RUN_KMEANS_WEIGHTED = True
RUN_KMEANS_WEIGHTED = False

RUN_KMEANS_VECTOR = True
RUN_KMEANS_VECTOR = False

RUN_KMEANS_VECTOR2 = True
RUN_KMEANS_VECTOR2 = False

RUN_BASIC = True
RUN_BASIC = False

RUN_BASIC_NO_LEARN = True
RUN_BASIC_NO_LEARN = False

RUN_SPECIAL_KMEANS = False
RUN_SPECIAL_KMEANS = True

RUN_WEIGHTED_SPECIAL_KMEANS = True
RUN_WEIGHTED_SPECIAL_KMEANS = False


#gw0.55-250
#gw0.4-250

WRITE_MARKDOWN = True
WRITE_LOGS = True

PATH_PREFIX = ("fordypningsprosjekt\\" if RUN_FROM_SCRIPT else None)

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.STANDARD_RUNNING_LENGTH,
                kmeans_episodes=kMeansClient.KMEANS_RUNNING_LENGTH, weighted_kmeans=True, render_mode=kMeansClient.RENDER_MODE,
                game_mode=kMeansClient.GAME_MODE, k=kMeansClient.K_MEANS_K, save_plot=True, ignore_kmeans=False,
                use_vectors=RUN_KMEANS_VECTOR, vector_type=1, learn=True, use_special_kmeans=False, markdownStorer=None,
                mode="insert_mode", write_logs=WRITE_LOGS):

    if MULTITHREADING:
        config_holder = configHolder(discount_factor=discount_factor, gaussian_width=gaussian_width,
                                     exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                     kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans,
                                     render_mode=render_mode,
                                     game_mode=game_mode, k=k, save_plot=False, ignore_kmeans=ignore_kmeans,
                                     use_vectors=use_vectors,
                                     vector_type=vector_type, learn=learn, do_standardize=True,
                                     use_special_kmeans=use_special_kmeans, write_logs=write_logs)
        datas = []
        pool = Pool(processes=(cpu_count() - 1))
        with Pool((cpu_count() - 1)) as p:
            datas = p.map(config_holder.run_with_seed, range(seed_count))

        datas = np.asarray(datas)
    else:
        datas = []
        for seed in range(seed_count):
            data = kMeansClient.run_program(seed=seed, discount_factor=discount_factor, gaussian_width=gaussian_width,
                                            exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                                            kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans, render_mode=render_mode,
                                            game_mode=game_mode, k=k, save_plot=False, ignore_kmeans=ignore_kmeans, use_vectors=use_vectors,
                                            vector_type=vector_type, learn=learn, do_standardize=True, use_special_kmeans=use_special_kmeans,
                                            write_logs=write_logs)
            datas.append(data)
        datas = np.asarray(datas)

    plot_title = ("" if MULTITHREADING else "") + plot_title
    plotHelper.plot_with_max_min_mean_std(datas, plot_name, plot_title)
    avg = np.round(np.mean(datas), 2).item()
    if markdownStorer != None:
        markdownStorer.add_data_point(mode, avg, plot_name, gaussian_width, k, seed_count)
    return datas



def run_gaussian_k():
    if WRITE_MARKDOWN:
        markdownStorer = MarkdownStorer(Ks=K_VALUES, GWs=GAUSSIANS, learn_length=kMeansClient.STANDARD_RUNNING_LENGTH, comment=COMMENT)
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
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__unweighted_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} unweighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, gaussian_width=gaussian_width,
                                                         k=k, weighted_kmeans=False, use_vectors=False,
                                                         markdownStorer=markdownStorer, mode="Kmeans Unweighted")
                datas_list.append(datas)
                labels.append("unweighted")

            if RUN_KMEANS_WEIGHTED:
                print("Starting Kmeans Weighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__weighted_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} weighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width, k=k, weighted_kmeans=True,
                                                         use_vectors=False, markdownStorer=markdownStorer, mode="Kmeans Weighted")
                datas_list.append(datas)
                labels.append("weighted")

            if RUN_KMEANS_VECTOR:
                print("Starting Vector...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__vector_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} vector-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width, k=k, weighted_kmeans=True,
                                                         use_vectors=True, markdownStorer=markdownStorer, mode="Kmeans Vector")
                datas_list.append(datas)
                labels.append("vector")

            if RUN_KMEANS_VECTOR2:
                print("Starting Vector-2...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__vector2_plot.png"
                name = fileHelper.osFormat(name, LINUX)

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} vector2-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width, k=k, weighted_kmeans=True,
                                                         use_vectors=True, vector_type=2, markdownStorer=markdownStorer, mode="Kmeans Vector-2")
                datas_list.append(datas)
                labels.append("vector2")

            if RUN_BASIC:
                if k == K_VALUES[0]:
                    print("Starting No Kmeans...")
                    path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\basic"
                    fileHelper.createDirIfNotExist(path, linux=LINUX)
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__basic_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} basic plot"
                    basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                   gaussian_width=gaussian_width, k=k,
                                                                   weighted_kmeans=False, ignore_kmeans=True,
                                                                   use_vectors=False, markdownStorer=markdownStorer, mode="No Kmeans")
                    datas_list.append(basic_datas)
                    labels.append("basic")
                else:
                    datas_list.append(basic_datas)
                    labels.append("basic")

            if RUN_SPECIAL_KMEANS:
                print("Starting Special Kmeans Unweighted...")
                path = f"{PATH_PREFIX}mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__special_kmeans_plot.png"
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
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__weighted_special_kmeans_plot.png"
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
                    name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__basic_no_learn_plot.png"
                    name = fileHelper.osFormat(name, LINUX)

                    title = f"gw={gaussian_width}, avg{SEED_COUNT} basic-no_learn plot"
                    basic_NL_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                                    gaussian_width=gaussian_width, k=k,
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
            if len(labels) > 1:
                path = f"plots\\generic\\{kMeansClient.GAME_MODE}\\aggregate\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path, linux=LINUX)

                types =  f"{'_weighted' if RUN_KMEANS_WEIGHTED else ''}{'_unweighted' if RUN_KMEANS_UNWEIGHTED else ''}{'_vector' if RUN_KMEANS_VECTOR else ''}{'_basic' if RUN_BASIC else ''}"
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
        end = time.time()
        print(f"\n\n\n\n{gaussian_width=}: time:{end - start}")
        time.sleep(5)
    if markdownStorer != None:
        markdownStorer.create_markdown(LINUX, PATH_PREFIX)

if __name__ == '__main__':
    total_start = time.time()
    run_gaussian_k()
    total_end = time.time()
    print(f"\n\n\n\nComplete program time: {total_end-total_start}")