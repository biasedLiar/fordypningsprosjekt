import time

from genericClients import kMeansClient
from genericExpandedClients import kMeansExpandedClient
from genericExpandedClients import kMeansTrigExpandedClient

class configHolder:
    def __init__(self, discount_factor=kMeansClient.DISCOUNT_FACTOR, gaussian_width=kMeansClient.GAUSSIAN_WIDTH,
                 exploration_rate=kMeansClient.EXPLORATION_RATE, standard_episodes=kMeansClient.LEARNING_LENGTH,
                 kmeans_episodes=kMeansClient.SLEEPING_LENGTH, weighted_kmeans=True, render_mode=kMeansClient.RENDER_MODE,
                 game_mode=kMeansClient.GAME_MODE, k=kMeansClient.K_MEANS_K, save_plot=True, ignore_kmeans=False,
                 use_vectors=False, vector_type=1, learn=True, use_special_kmeans=False, do_standardize=True, write_logs=True,
                 use_expanded=False, segments=1, expander_gaussian=1, trig_expanded=False, use_cosine_similarity=False,
                 use_search_tree=False, search_tree_depth=-1, save_midway=False):
        self.discount_factor=discount_factor
        self.gaussian_width=gaussian_width
        self.exploration_rate = exploration_rate
        self.standard_episodes = standard_episodes
        self.kmeans_episodes = kmeans_episodes
        self.weighted_kmeans = weighted_kmeans
        self.render_mode = render_mode
        self.game_mode = game_mode
        self.k = k
        self.save_plot = save_plot
        self.ignore_kmeans=ignore_kmeans
        self.use_vectors = use_vectors
        self.vector_type=vector_type
        self.learn= learn
        self.use_special_kmeans = use_special_kmeans
        self.do_standardize = do_standardize
        self.write_logs=write_logs
        self.use_expanded = use_expanded
        self.segments = segments
        self.expander_gaussian = expander_gaussian
        self.trig_expanded = trig_expanded
        self.use_cosine_similarity = use_cosine_similarity
        self.use_search_tree = use_search_tree
        self.search_tree_depth = search_tree_depth
        self.save_midway = save_midway


    def run_with_seed(self, seed):
        if self.use_expanded:
            data = kMeansExpandedClient.run_program(seed=seed, discount_factor=self.discount_factor,
                                            gaussian_width=self.gaussian_width,
                                            exploration_rate=self.exploration_rate,
                                            standard_episodes=self.standard_episodes,
                                            kmeans_episodes=self.kmeans_episodes, weighted_kmeans=self.weighted_kmeans,
                                            render_mode=self.render_mode,
                                            game_mode=self.game_mode, k=self.k, save_plot=self.save_plot,
                                            ignore_kmeans=self.ignore_kmeans,
                                            use_vectors=self.use_vectors,
                                            vector_type=self.vector_type, learn=self.learn,
                                            use_special_kmeans=self.use_special_kmeans,
                                            do_standardize=self.do_standardize, write_logs=self.write_logs,
                                            segments=self.segments, expander_gaussian=self.expander_gaussian,
                                            use_cosine_similarity=self.use_cosine_similarity,
                                            use_search_tree=self.use_search_tree, search_tree_depth=self.search_tree_depth,
                                            save_midway=self.save_midway)
        elif self.trig_expanded:
            data = kMeansTrigExpandedClient.run_program(seed=seed, discount_factor=self.discount_factor,
                                            gaussian_width=self.gaussian_width,
                                            exploration_rate=self.exploration_rate,
                                            standard_episodes=self.standard_episodes,
                                            kmeans_episodes=self.kmeans_episodes, weighted_kmeans=self.weighted_kmeans,
                                            render_mode=self.render_mode,
                                            game_mode=self.game_mode, k=self.k, save_plot=self.save_plot,
                                            ignore_kmeans=self.ignore_kmeans,
                                            use_vectors=self.use_vectors,
                                            vector_type=self.vector_type, learn=self.learn,
                                            use_special_kmeans=self.use_special_kmeans,
                                            do_standardize=self.do_standardize, write_logs=self.write_logs,
                                            segments=self.segments, expander_gaussian=self.expander_gaussian,
                                            use_cosine_similarity=self.use_cosine_similarity)
        else:
            data = kMeansClient.run_program(seed=seed, discount_factor=self.discount_factor,
                                            gaussian_width=self.gaussian_width,
                                            exploration_rate=self.exploration_rate,
                                            standard_episodes=self.standard_episodes,
                                            kmeans_episodes=self.kmeans_episodes, weighted_kmeans=self.weighted_kmeans,
                                            render_mode=self.render_mode,
                                            game_mode=self.game_mode, k=self.k, save_plot=self.save_plot,
                                            ignore_kmeans=self.ignore_kmeans,
                                            use_vectors=self.use_vectors,
                                            vector_type=self.vector_type, learn=self.learn,
                                            use_special_kmeans=self.use_special_kmeans,
                                            do_standardize=self.do_standardize, write_logs=self.write_logs,
                                            use_search_tree=self.use_search_tree, search_tree_depth=self.search_tree_depth,
                                            save_midway=self.save_midway)

        return data