'''functions for quick graphing'''
from datetime import datetime
# from cuml import UMAP as umap
import numpy as np
from matplotlib import pyplot as plt

class GraphUtils():
    def __init__(self, logger):
        self.logger = logger

    def categorical_plot_group(self, x, y, legend_labels, title, filename):
        '''creates and saves a categorical plot'''
        self.logger.log(f"Plotting bar chart: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(y)):
            ax.scatter(x[i], y[i], label=legend_labels[i])
    
        # plt.xticks(range(x[0]), (str(i) for i in x[0]))
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)
        self.save_plt_fig(fig, filename, (lgd, ttl, ))

    def create_boxplot(self, data, title, filename):
        '''creates and saves a single boxplot'''
        self.logger.log(f"Plotting boxplot: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        ax.suptitle(title)
        self.save_plt_fig(fig, filename)

    def create_boxplot_group(self, data, labels, title, filename):
        '''creates and saves a group of boxplot'''
        self.logger.log(f"Plotting boxplot group: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        fig.suptitle(title)
        ax.set_xticklabels(labels)
        self.save_plt_fig(self, fig, filename)

    def create_scatter_plot(self, data, labels, title, filename, legend_names=None):
        '''creates and saves a scatter plot'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
        if legend_names is None:
            legend = ax.legend(*scatter.legend_elements(), \
                        loc="upper left", title="Cluster no.", bbox_to_anchor=(1, 0.5))
        else:
            handles, _ = scatter.legend_elements(num=None)
            legend = ax.legend(handles, legend_names, loc="upper left", title="Legend", bbox_to_anchor=(1, 0.5))

        ttl = fig.suptitle(title)

        self.save_plt_fig(fig, filename, [ttl, legend])

    def plot_tsne(self, new_values, labels, title):
        self.logger.log(f"Plotting TSNE figure")
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
    
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        for i in range(len(x)):
            ax.scatter(x[i], y[i])

        for i in range(len(x)):
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')

        fig.suptitle(title)
    
        name = "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.logger.output_path / name
        self.logger.log(f"Saving TSNE figure to {path}")
        fig.savefig(path)

    def save_plt_fig(self, fig, filename, bbox_extra_artists=None):
        '''Save a plot figure to file with timestamp'''
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = f"{filename}_{current}"
        if self.logger is not None:
            output_path = self.logger.output_path / output_path

        if bbox_extra_artists is None:
            fig.savefig(output_path)
        else:
            fig.savefig(output_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

        plt.close(fig)

    def plot_umap(self, embedding, title):
        self.logger.log("Plotting UMAP")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
        # ax.gca().set_aspect('equal', 'datalim')
        fig.suptitle(title)

        name = "UMAP_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.logger.output_path / name
        fig.savefig(path)
