'''functions for quick graphing'''
from datetime import datetime
import math
# from cuml import UMAP as umap
import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn import cluster, metrics
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture as BGMM

class GraphUtils():
    def __init__(self, logger):
        self.logger = logger

    def calculate_BGMM(self, data, n_components, title, filename):
        self.logger.log(f"Calculating GMM with {n_components} components")
        bgmm = BGMM(n_components=n_components).fit(data)
        labels = bgmm.predict(data)
        self.create_scatter_plot(data, labels, f"BGMM {title}", f'BGMM_{title}')
        probs = bgmm.predict_proba(data)
        probs_output = self.logger.output_path / f'BGMM_probs_{title}.txt'
        np.savetxt(probs_output, probs)

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

    def create_scatter_plot(self, data, labels, title, filename):
        '''creates and saves a scatter plot'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
        legend = ax.legend(*scatter.legend_elements(), \
                    loc="upper left", title="Cluster no.", bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)

        self.save_plt_fig(fig, filename, [ttl, legend])

    def get_best_cluster_size(self, X, clusters):
        '''measure silhouette scores for the given cluster sizes and return the best k and its score'''
        self.logger.log("Getting best k-means cluster size with average silhouette score")
        avg_sil = []
        for n in clusters:
            kmeans = cluster.KMeans(n_clusters=n)
            kmeans.fit(X)
            labels = kmeans.labels_
            silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
            avg_sil.append(silhouette_score)
            self.logger.log(f"n = {n}, silhouette score = {silhouette_score}")

        k = clusters[avg_sil.index(max(avg_sil))]
        max_n = math.ceil(max(avg_sil) * 100)
        self.logger.log(f"Max silhouette score with {k} clusters")

        return (k, max_n)

    def k_means_cluster(self, data, title, filename):
        '''Creates and saves k-means clusters using the best cluster size based on silhouette score'''
        self.logger.log("k-means clustering")
        max_binary_test = math.floor(math.log2(len(data)))
        (k, s) = self.get_best_cluster_size(data, list(2**i for i in range(1,max_binary_test)))
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.labels_
        self.create_scatter_plot(data, labels, f"{title} with {s}% silhoutte score", filename)


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

    def tsne_plot(self, model, perplex, title):
        '''create and save a t-SNE plot'''
        self.logger.log("Creating t-SNE plot")
        self.logger.log("Getting labels and tokens for t-SNE")
        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)
    
        self.logger.log(f"Creating TSNE model with perplexity {perplex}")
        tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

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

    def umap_plot(self, model, title):
        self.logger.log("Creating UMAP")
        '''create and save a umap plot'''
        labels = []
        tokens = []

        self.logger.log("Extracting labels and token")
        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)

        self.logger.log("Creating UMAP")
        reducer = umap.UMAP(verbose=True)
        embedding = reducer.fit_transform(tokens)

        self.logger.log("Plotting UMAP")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
        # ax.gca().set_aspect('equal', 'datalim')
        fig.suptitle(title)

        name = "UMAP_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.logger.output_path / name
        fig.savefig(path)
