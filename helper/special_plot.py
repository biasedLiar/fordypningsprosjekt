import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from helper.kmeans_model import Model


class Plot:

    def __init__(self, model: Model, xlim: tuple[float, float], ylim: tuple[float, float], alpha: float = 0.3):
        self.model = model
        self.fig, self.ax = plt.subplots()

        plt.xticks(())
        plt.yticks(())
        plt.xlim(xlim)
        plt.ylim(ylim)

        self.scatter_batch = self.ax.scatter([], [], linewidths=0, color='lightgray')
        self.scatter_mus = self.ax.scatter([], [], linewidths=0, color='#888888', marker=MarkerStyle("D"))
        self.scatters_batch_labeled = [
            self.ax.scatter([], [], linewidths=0, color=cycle['color'], alpha=alpha)
            for _, cycle in zip(range(self.model.mu.shape[0]), matplotlib.rcParams["axes.prop_cycle"]())
        ]
        self.scatters_mus_labeled = [
            self.ax.scatter([], [], linewidths=0, color=cycle['color'], marker=MarkerStyle("D"))
            for _, cycle in zip(range(self.model.mu.shape[0]), matplotlib.rcParams["axes.prop_cycle"]())
        ]

    def __draw(self, batch: np.ndarray, labels: np.ndarray):
        self.scatter_batch.set_offsets(batch)
        self.scatter_mus.set_offsets(self.model.mu)
        for scatter_batch_labeled, scatter_mus_labeled in zip(self.scatters_batch_labeled, self.scatters_mus_labeled):
            scatter_batch_labeled.set_visible(False)
            scatter_mus_labeled.set_visible(False)

        for label in range(1, labels.max() + 1):
            mu_offsets = self.model.mu[labels == label]
            if len(mu_offsets) > 0:
                self.scatters_mus_labeled[label - 1].set_offsets(mu_offsets)
                self.scatters_mus_labeled[label - 1].set_visible(True)

        x_offsets: dict[int, list] = {}
        for x in batch:
            closest_mu = np.argmin(np.sum(np.square(self.model.mu - x), axis=1))
            x_offsets.setdefault(labels[closest_mu], []).append(x)

        for label in range(1, labels.max() + 1):
            if label in x_offsets:
                self.scatters_batch_labeled[label - 1].set_offsets(x_offsets[label])
                self.scatters_batch_labeled[label - 1].set_visible(True)

    def save(self, title: str, batch: np.ndarray, labels: np.ndarray):
        self.__draw(batch, labels)
        plt.tight_layout(pad=0.0)
        plt.savefig(title, bbox_inches='tight', pad_inches=.0)

    def show(self, batch: np.ndarray, labels: np.ndarray):
        self.__draw(batch, labels)
        plt.show()

    def draw_frame(self, batch: np.ndarray, labels: np.ndarray):
        self.__draw(batch, labels)
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)

    def draw_frame_no_label(self, batch: np.ndarray):
        labels = np.zeros(batch.shape[0], dtype=int)
        self.__draw(batch, labels)
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)

    def clf(self):
        plt.clf()
