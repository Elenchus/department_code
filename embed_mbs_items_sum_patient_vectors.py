import os
from phd_utils import file_utils, graph_utils
import gc
import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import datetime
from functools import partial
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE

# map datetime to 0-based days?

# class IterableFormat(object):
#     def __init__(self, file):
#         self.file = file

#     def __iter__(self):
#         with open(self.file, newline='') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 yield row

def get_average_sil_score(tuple_of_n_index):
    (X, n, index) = tuple_of_n_index
    patient_kmeans = cluster.KMeans(n_clusters=n)
    patient_kmeans.fit(X)
    patient_labels = patient_kmeans.labels_
    
    return (metrics.silhouette_score(X, patient_labels, metric='euclidean'), index)

def get_items_per_day(patient):
    same_day_claim = []
    days_of_service = patient[1].groupby('DOS')
    for day in days_of_service:
        claims = list(map(str, day[1]['ITEM'].values))
        same_day_claim.append(claims)

    return same_day_claim

def sum_patient_vectors(model, patient):
    codes = list(map(str, patient[1]['ITEM'].values))
    patient_vector = []
    for code in codes:
        code_vector = model[code]
        if len(patient_vector) == 0:
            patient_vector = list(x for x in code_vector)
        else:
            if len(patient_vector) != len(code_vector):
                raise("This isn't right")

            for i in range(len(patient_vector)):
                patient_vector[i] = patient_vector[i] + code_vector[i]

    return patient_vector

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
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
    logger = file_utils.Logger(__name__, "embed.log")
    logger.log("Starting")
    path = 'C:/Data/MBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['PIN', 'DOS', 'ITEM'])
    patients = data.groupby('PIN')
    patient_ids = data['PIN'].unique()
    unique_items = data['ITEM'].unique()
        
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
            min_count=20,
            workers=3,
            iter=5)

    # X = model[model.wv.vocab]

    logger.log("Summing patient days")
    func = partial(sum_patient_vectors, model)
    p = Pool(processes=6)
    patient_vectors = p.imap(func, patients)
    p.close()
    p.join()

    patient_vectors = np.array(patient_vectors)

    del model
    gc.collect()

    logger.log("Clustering patient vectors")
    cluster_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256]
    list_to_send = []
    for i in range(len(cluster_no)):
        list_to_send.append((patient_vectors, cluster_no[i], i))

    p = Pool(processes=6)
    patient_sil_score_map = p.imap(get_average_sil_score, list_to_send)
    p.close()
    p.join()

    patient_sil_scores, patient_sil_indices = map(list, zip(*patient_sil_score_map))

    patient_k = cluster_no[patient_sil_indices[patient_sil_scores.index(max(patient_sil_scores))]]
    print(f"Max silhouette score with {patient_k} clusters")

    patient_kmeans = cluster.KMeans(n_clusters=patient_k)
    patient_kmeans.fit(patient_vectors)
    patient_labels = patient_kmeans.labels_

    logger.log("Plotting")
    patient_fig = plt.figure()
    patient_ax = patient_fig.add_subplot(111)
    patient_scatter = patient_ax.scatter(patient_vectors[:, 0], patient_vectors[:, 1], c=patient_labels)
    patient_legend = patient_ax.legend(*patient_scatter.legend_elements(), loc="upper left", title="Cluster no.")

    patient_fig.savefig("patient_k-means_" + datetime.now().strftime("%Y%m%dT%H%M%S"))

    logger.log("Finished", '!')

    # cur = get_unique_per_patient(files[0])