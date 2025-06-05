import time

from genericClients import kMeansClient
from genericExpandedClients import kMeansExpandedClient

class configHolder:
    def __init__(self, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                 exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.LEARNING_LENGTH,
                 kmeans_episodes=kMeansClient.SLEEPING_LENGTH, weighted_kmeans=True, render_mode=kMeansClient.RENDER_MODE,
                 game_mode=kMeansClient.GAME_MODE, k=kMeansClient.K_MEANS_K, ignore_kmeans=False,
                 learn=False, do_standardize=True, write_logs=True, weighted_sigmoid=False,
                 use_expanded=False, segments=1, expander_gaussian=1,
                 use_search_tree=False, search_tree_depth=-1, save_midway=False):
        self.gaussian_width=gaussian_width
        self.exploration_rate = exploration_rate
        self.standard_episodes = standard_episodes
        self.kmeans_episodes = kmeans_episodes
        self.weighted_kmeans = weighted_kmeans
        self.weighted_sigmoid = weighted_sigmoid
        self.render_mode = render_mode
        self.game_mode = game_mode
        self.k = k
        self.ignore_kmeans=ignore_kmeans
        self.learn= learn
        self.do_standardize = do_standardize
        self.write_logs=write_logs
        self.use_expanded = use_expanded
        self.segments = segments
        self.expander_gaussian = expander_gaussian
        self.use_search_tree = use_search_tree
        self.search_tree_depth = search_tree_depth
        self.save_midway = save_midway


    def run_with_seed(self, seed):
        if self.use_expanded:
            data = kMeansExpandedClient.run_program(seed=seed,
                                            gaussian_width=self.gaussian_width,
                                            exploration_rate=self.exploration_rate,
                                            standard_episodes=self.standard_episodes,
                                            kmeans_episodes=self.kmeans_episodes, weighted_kmeans=self.weighted_kmeans,
                                            render_mode=self.render_mode,
                                            game_mode=self.game_mode, k=self.k,
                                            ignore_kmeans=self.ignore_kmeans, learn=self.learn,
                                            do_standardize=self.do_standardize, write_logs=self.write_logs,
                                            segments=self.segments, expander_gaussian=self.expander_gaussian,
                                            use_search_tree=self.use_search_tree, search_tree_depth=self.search_tree_depth,
                                            save_midway=self.save_midway, weighted_sigmoid=self.weighted_sigmoid)
        else:
            data = kMeansClient.run_program(seed=seed,
                                            gaussian_width=self.gaussian_width,
                                            exploration_rate=self.exploration_rate,
                                            standard_episodes=self.standard_episodes,
                                            eval_length=self.kmeans_episodes, weighted_kmeans=self.weighted_kmeans,
                                            render_mode=self.render_mode,
                                            game_mode=self.game_mode, k=self.k,
                                            ignore_kmeans=self.ignore_kmeans,
                                            write_logs=self.write_logs,
                                            use_search_tree=self.use_search_tree, search_tree_depth=self.search_tree_depth,
                                            save_midway=self.save_midway, weighted_sigmoid=self.weighted_sigmoid)
        return data