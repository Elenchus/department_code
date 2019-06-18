import os
import FileUtils
import gc
import pandas as pd
from datetime import datetime
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import metrics

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "embed.log")
    logger.log("Starting")
    path = 'C:/Data/MBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['ITEM', 'SPR_RSP']).astype(str)
    string_list = data.values.tolist()
    unique_items = data['ITEM'].unique()
    unique_rsp = data['SPR_RSP'].unique()

    del data
    gc.collect()

    logger.log("Embedding vectors")
    size = ceil(sqrt(sqrt(len(unique_items) + len(unique_rsp))))

    model = Word2Vec(
            string_list,
            size=size,
            window= 2,
            min_count=20,
            workers=3,
            iter=5)

    X = model[model.wv.vocab]

    logger.log("Clustering")
    cluster_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, len(unique_rsp), 256]
    avg_sil = []
    for n in cluster_no:
        kmeans = cluster.KMeans(n_clusters=n)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        avg_sil.append(silhouette_score)

    k = cluster_no[avg_sil.index(max(avg_sil))]
    print(f"Max silhouette score with {k} clusters")

    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    logger.log("Plotting")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster no.")

    fig.savefig(logger.output_path + "items_and_specialties_k-means_" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    logger.log("Finished", '!')