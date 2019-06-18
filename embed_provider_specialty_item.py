import os
import FileUtils
import gc
import itertools
import pandas as pd
from datetime import datetime
from functools import partial
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE

def sum_patient_vectors(model, vocab, patient):
    similarities = []
    for claim in patient:
        if claim[1] not in vocab:
            continue

        similarities.append(model.similarity(claim[1], claim[2]))
        
    return (sum(similarities) / len(similarities), min(similarities))
    

def tsne_plot(model, perplex):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

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
    
    return fig

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "embed_specialty_item")
    logger.log("Starting")
    path = 'C:/Data/MBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['ITEM', 'SPR_RSP']).astype(str)
    string_list = data.values.tolist()
    no_unique_items = len(data['ITEM'].unique())
    no_unique_rsp = len(data['SPR_RSP'].unique())

    del data
    gc.collect()

    logger.log("Embedding vectors")
    size = ceil(sqrt(sqrt(no_unique_items + no_unique_rsp)))

    model = Word2Vec(
            string_list,
            size=size,
            window= 2,
            min_count=20, # if this is not 1 then sum_patient_vectors needs to change, or will throw error -> after changing from 20 the silhouette score changed...
            workers=3,
            iter=5)

    del string_list
    gc.collect()

    X = model[model.wv.vocab]

    logger.log("Clustering")
    # cluster_no = [32, 64, 96, 128, len(no_unique_rsp), 160, 192, 224, 256]
    # avg_sil = []
    # for n in cluster_no:
    #     kmeans = cluster.KMeans(n_clusters=n)
    #     kmeans.fit(X)
    #     labels = kmeans.labels_
    #     silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    #     avg_sil.append(silhouette_score)
    #     logger.log(f"n = {n}, silhouette score = {silhouette_score}")

    # k = cluster_no[avg_sil.index(max(avg_sil))]
    # logger.log(f"Max silhouette score with {k} clusters")
    k = 143

    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    logger.log("Re-loading parquet file")
    data = pd.read_parquet(filename, columns=['PIN', 'ITEM', 'SPR_RSP']).astype(str)

    all_claims = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

    del data
    gc.collect()

    logger.log("Summing patient vectors")
    func = partial(sum_patient_vectors, [model, model.vocab])
    p = Pool(processes=6)
    data_map = p.imap(partial, all_claims)
    p.close()
    p.join()

    del all_claims
    gc.collect()

    logger.log("Creating lists")
    avg_sims, min_sims = zip(*data_map)

    logger.log("Plotting")
    tsne_fig = tsne_plot(model, 143)
    tsne_fig.savefig(logger.output_path + "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster no.")

    fig.savefig(logger.output_path + "items_and_specialties_k-means_" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    b_plot_avg = plt.figure()
    b_ax_avg = b_plot_avg.add_subplot(111)
    b_ax_avg.boxplot(avg_sims)
    b_ax_avg.title("Average patient item/specialty similarity score")
    b_plot_avg.savefig(logger.output_path + "patient_average_similarity_boxplot" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    b_plot_min = plt.figure()
    b_ax_min = b_plot_min.add_subplot(111)
    b_ax_min.boxplot(avg_sims)
    b_ax_min.title("Unique referrers per patient per year")
    b_plot_min.savefig(logger.output_path + "patient_minimum_similarity_boxplot" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    logger.log("Finished", '!')