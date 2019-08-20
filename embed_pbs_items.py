import os
import gc
import file_utils
import pandas as pd

from copy import deepcopy
from datetime import datetime
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE

def get_items_per_day(patient):
    same_day_claim = []
    days_of_service = patient[1].groupby('SPPLY_DT')
    for day in days_of_service:
        claims = list(map(str, day[1]['ITM_CD'].values))
        same_day_claim.append(claims)

    return same_day_claim

if __name__ == "__main__":
    logger = file_utils.logger(__name__, "embed.log")
    logger.log("Starting")
    path = 'C:/Data/PBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['PTNT_ID', 'SPPLY_DT', 'ITM_CD'])
    patients = data.groupby('PTNT_ID')
    patient_ids = data['PTNT_ID'].unique()
    unique_items = data['ITM_CD'].unique()

    del data
    gc.collect()

    logger.log("Combining patient information")
    p = Pool(processes=6)
    data_map = p.imap(get_items_per_day, patients)
    p.close()
    p.join()

    logger.log("Flattening output")
    same_day_claims = []    
    for i in data_map:
        for j in i:
            same_day_claims.append(j) 

    del data_map
    gc.collect()

    logger.log("Embedding vectors")
    size = ceil(sqrt(sqrt(len(unique_items))))

    model = Word2Vec(
            same_day_claims,
            size=size,
            window=max(len(l) for l in same_day_claims),
            min_count=1,
            workers=3,
            iter=1)

    X = model[model.wv.vocab]

    logger.log("Clustering")
    cluster_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256]
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

    fig.savefig(logger.output_path + "pbs_k-means_" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    logger.log("Finished", '!')