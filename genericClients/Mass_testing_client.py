import matplotlib.pyplot as plt

from classes.genericBasic import *
import helper.fileHelper as fileHelper
import helper.plotHelper as plotHelper
import time
import genericClients.kMeansClient as kMeansClient

SEED_COUNT = 30


GAUSSIANS = [0.515, 0.535, 0.55, 0.565, 0.58]
GAUSSIANS = [0.55]

K_VALUES = [200, 250, 300, 350, 400]
K_VALUES = [250]

EXPLORATION_RATES = [0.1]



RUN_KMEANS_UNWEIGHTED = True
RUN_BASIC = True
RUN_KMEANS_WEIGHTED = True
RUN_KMEANS_VECTOR = False
#gw0.55-250

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.STANDARD_RUNNING_LENGTH,
                kmeans_episodes=kMeansClient.KMEANS_RUNNING_LENGTH, weighted_kmeans=True, render_mode=kMeansClient.RENDER_MODE,
                game_mode=kMeansClient.GAME_MODE, k=kMeansClient.K_MEANS_K, save_plot=True, ignore_kmeans=False):
    datas = []
    for seed in range(seed_count):
        data = kMeansClient.run_program(seed=seed, discount_factor=discount_factor, gaussian_width=gaussian_width,
                exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                kmeans_episodes=kmeans_episodes, weighted_kmeans=weighted_kmeans, render_mode=render_mode,
                game_mode=game_mode, k=k, save_plot=False, ignore_kmeans=ignore_kmeans)
        datas.append(data)
    datas = np.asarray(datas)
    plotHelper.plot_with_max_min_mean_std(datas, plot_name, plot_title)
    return datas


def run_gaussian_k():

    for gaussian_width in GAUSSIANS:
        start = time.time()
        for k in K_VALUES:
            start_2 = time.time()
            labels = []
            datas_list = []
            if RUN_KMEANS_UNWEIGHTED:
                path = f"mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__unweighted_plot.png"

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} unweighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, gaussian_width=gaussian_width, k=k, weighted_kmeans=False)
                datas_list.append(datas)
                labels.append("unweighted")

            if RUN_KMEANS_WEIGHTED:
                path = f"mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__weighted_plot.png"

                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} weighted-kmeans plot"
                datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT,
                                                         gaussian_width=gaussian_width, k=k, weighted_kmeans=True)
                datas_list.append(datas)
                labels.append("weighted")

            if RUN_BASIC and k == K_VALUES[0]:
                path = f"mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\basic"
                fileHelper.createDirIfNotExist(path)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}__weighted_plot.png"

                title = f"gw={gaussian_width}, avg{SEED_COUNT} nearest neighbor plot"
                basic_datas = run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, gaussian_width=gaussian_width, k=k, weighted_kmeans=False, ignore_kmeans=True)
                datas_list.append(basic_datas)
                labels.append("basic")
            else:
                datas_list.append(basic_datas)
                labels.append("basic")

            end_2 = time.time()
            print(f"\n\n{gaussian_width=}, {k=}: time:{end_2 - start_2}")
            if len(labels) > 1:
                path = f"mplots\\generic\\{kMeansClient.GAME_MODE}\\aggregate\\{gaussian_width}g\\{k}k"
                fileHelper.createDirIfNotExist(path)

                types =  f"{'_weighted' if RUN_KMEANS_WEIGHTED else ''}{'_unweighted' if RUN_KMEANS_UNWEIGHTED else ''}{'_vector' if RUN_KMEANS_VECTOR else ''}{'_basic' if RUN_BASIC else ''}"
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}" \
                              f"{types}.png"
                title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT}{types} plot"

                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=False)
                name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}" \
                              f"{types}_std.png"
                plotHelper.plot_multiple_graph_types(datas_list, labels, name, title, show_std=True)
            time.sleep(2)
        end = time.time()
        print(f"\n\n\n\n{gaussian_width=}: time:{end - start}")
        time.sleep(5)

if __name__ == '__main__':
    total_start = time.time()
    run_gaussian_k()
    total_end = time.time()
    print(f"\n\n\n\nComplete program time: {total_end-total_start}")