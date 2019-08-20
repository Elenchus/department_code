import math
# from cuml import UMAP as umap
import umap
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def categorical_plot_group(logger, x, y, legend_labels, title, filename):
    logger.log(f"Plotting bar chart: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y)):
        ax.scatter(x[i], y[i], label=legend_labels[i])
    
    # plt.xticks(range(x[0]), (str(i) for i in x[0]))
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ttl = fig.suptitle(title)
    save_plt_fig(logger, fig, filename, (lgd,ttl, ))

def create_boxplot(logger, data, title, filename):
    logger.log(f"Plotting boxplot: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    ax.suptitle(title)
    save_plt_fig(logger, fig, filename)

def create_boxplot_group(logger, data, labels, title, filename):
    logger.log(f"Plotting boxplot group: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    fig.suptitle(title)
    ax.set_xticklabels(labels)
    save_plt_fig(logger, fig, filename)

def create_scatter_plot(logger, data, labels, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
    legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster no.", bbox_to_anchor=(1, 0.5))
    ttl = fig.suptitle(title)

    save_plt_fig(logger, fig, filename, [ttl, legend])

def get_best_cluster_size(logger, X, clusters):
    logger.log("Getting best k-means cluster size with average silhouette score")
    avg_sil = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        avg_sil.append(silhouette_score)
        logger.log(f"n = {n}, silhouette score = {silhouette_score}")

    k = clusters[avg_sil.index(max(avg_sil))]
    max_n = math.ceil(max(avg_sil) * 100)
    logger.log(f"Max silhouette score with {k} clusters")

    return (k, max_n)

def save_plt_fig(logger, fig, filename, bbox_extra_artists=None):
    current = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = f"{filename}_{current}"
    if logger != None:
        output_path = logger.output_path / output_path

    if bbox_extra_artists == None:
        fig.savefig(output_path)
    else:
        fig.savefig(output_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

    plt.close(fig)

def tsne_plot(logger, model, perplex, title):
    logger.log("Getting labels and tokens for t-SNE")
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    logger.log(f"Creating TSNE model with perplexity {perplex}")
    tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    logger.log(f"Plotting TSNE figure")
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    for i in range(len(x)):
        ax.scatter(x[i],y[i])
    for i in range(len(x)):
        ax.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    fig.suptitle(title)
    
    name = "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    path = logger.output_path / name
    logger.log(f"Saving TSNE figure to {path}")
    fig.savefig(path)

def umap_plot(logger, model, title):
    labels = []
    tokens = []

    logger.log("Extracting labels and token")
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    logger.log("Creating UMAP")
    reducer = umap.UMAP(verbose=True)
    embedding = reducer.fit_transform(tokens)

    logger.log("Plotting UMAP")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
    # ax.gca().set_aspect('equal', 'datalim')
    fig.suptitle(title)

    name = "UMAP_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    path = logger.output_path / name
    fig.savefig(path)
