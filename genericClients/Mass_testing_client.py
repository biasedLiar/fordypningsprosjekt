import matplotlib.pyplot as plt

from classes.genericBasic import *
import helper.fileHelper as fileHelper
import helper.plotHelper as plotHelper
import time
import genericClients.kMeansClient as kMeansClient

SEED_COUNT = 10


GAUSSIANS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]

K_VALUES = [20, 35, 50, 70, 100, 150, 200]

EXPLORATION_RATES = [0.1]
#gw0.5-200

def run_program_with_different_seeds(plot_name, plot_title, seed_count=3,
                discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.STANDARD_RUNNING_LENGTH,
                kmeans_episodes=kMeansClient.KMEANS_RUNNING_LENGTH, kmeans_type=kMeansClient.KMEANS_TYPE, render_mode=kMeansClient.RENDER_MODE,
                game_mode=kMeansClient.GAME_MODE, k=kMeansClient.K_MEANS_K, save_plot=True):
    datas = []
    for seed in range(seed_count):
        data = kMeansClient.run_program(seed=seed, discount_factor=discount_factor, gaussian_width=gaussian_width,
                exploration_rate=exploration_rate, standard_episodes=standard_episodes,
                kmeans_episodes=kmeans_episodes, kmeans_type=kmeans_type, render_mode=render_mode,
                game_mode=game_mode, k=k, save_plot=False)
        datas.append(data)
    datas = np.asarray(datas)
    '''
    bucket_data = plotHelper.average_every_n(datas, list_of_list=True, n=5)
    avg_reward = plotHelper.average_of_diff_seeds(bucket_data)
    error_bounds = plotHelper.get_upper_lower_error_bounds(bucket_data, avg_reward)
    
    
    x = np.arange(0, len(data), 5)

    #avg_reward = np.ones_like(avg_reward)*50
    #error_bounds = [avg_reward - avg_reward*0.5, avg_reward*2 - avg_reward]

    plt.errorbar(x, avg_reward, yerr=error_bounds, fmt='-o')
    '''

    avg_data, max_data, min_data = plotHelper.average_max_min_diagrams(datas)

    plt.plot(max_data, label='max_steps')
    plt.plot(avg_data, label='avg_steps')
    plt.plot(min_data, label='min_steps')
    plt.xlabel("Iterations")
    plt.ylabel("Steps")
    plt.legend(loc="lower right")
    plt.title(plot_title)
    plt.savefig(plot_name)
    plt.show()
    plt.clf()



def run_gaussian_k():

    for gaussian_width in GAUSSIANS:
        start = time.time()
        for k in K_VALUES:
            start_2 = time.time()

            path = f"mplots\\generic\\{kMeansClient.GAME_MODE}\\{gaussian_width}g\\{k}k"
            fileHelper.createDirIfNotExist(path)
            name = path + f"\\{SEED_COUNT}seed__{kMeansClient.STANDARD_RUNNING_LENGTH}_then_{kMeansClient.KMEANS_RUNNING_LENGTH}_plot.png"

            title = f"gw={gaussian_width}, k={k} avg{SEED_COUNT} plot"
            run_program_with_different_seeds(name, title, seed_count=SEED_COUNT, gaussian_width=gaussian_width, k=k)

            end_2 =time.time()
            print(f"\n\n{gaussian_width=}, {k=}: time:{end_2-start_2}")
            time.sleep(2)
        end = time.time()
        print(f"\n\n\n\n{gaussian_width=}: time:{end - start}")
        time.sleep(5)

if __name__ == '__main__':
    run_gaussian_k()