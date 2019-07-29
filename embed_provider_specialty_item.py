import os
import FileUtils
import gc
import itertools
import pandas as pd
from datetime import datetime
from FileUtils import tsne_plot
from functools import partial
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster

def sum_patient_vectors(model, patient):
    similarities = []
    pin = patient[0]
    for claims in patient:
        if claims == pin:
            continue
        for claim in claims:
            if claim[1] not in model.wv.vocab or claim[2] not in model.wv.vocab:
                continue

            similarities.append(model.similarity(claim[1], claim[2]))
    if len(similarities) != 0:
        return (sum(similarities) / len(similarities), min(similarities))

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "embed_specialty_item")

    filename = FileUtils.get_mbs_files()[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['ITEM', 'SPR_RSP']).astype(str)
    string_list = data.values.tolist()
    no_unique_items = len(data['ITEM'].unique())
    unique_rsp = data['SPR_RSP'].unique()
    no_unique_rsp = len(unique_rsp)

    del data
    gc.collect()

    logger.log("Embedding vectors")
    size = ceil(sqrt(sqrt(no_unique_items + no_unique_rsp)))

    model = Word2Vec(string_list, size=size, window= 2, min_count=20, workers=3,iter=5)

    del string_list
    gc.collect()

    X = model[model.wv.vocab]

    rsps_not_in_model = []
    for rsp in unique_rsp:
        if rsp not in model.wv.vocab:
            no_unique_rsp = no_unique_rsp - 1

    logger.log(f"Clustering with {no_unique_rsp} clusters")
    cluster_no = [32, 64, 96, 128, no_unique_rsp, 160, 192, 224, 256]
    k, score = FileUtils.get_best_cluster_size(logger, X, cluster_no)

    kmeans = cluster.KMeans(n_clusters=no_unique_rsp)
    kmeans.fit(X)
    labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    # logger.log("Plotting t-SNE and k-means")
    # FileUtils.tsne_plot(logger, model, no_unique_rsp, "t-")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    # legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster no.")

    # fig.savefig(logger.output_path + "items_and_specialties_k-means_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    FileUtils.create_scatter_plot(logger, X, labels, "Item/provider specialty clusters", "items_and_specialties_k_means")
    logger.log("Re-loading parquet file")
    data = pd.read_parquet(filename, columns=['PIN', 'ITEM', 'SPR_RSP']).astype(str)

    patients = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

    del data
    gc.collect()

    logger.log("Summing patient vectors")
    # func = partial(sum_patient_vectors, model)
    # p = Pool(processes=6)
    # data_map = p.imap(func, patients)
    # p.close()
    # p.join()

    # data_map = [i for i in data_map if i]
    
    
    data_map = []
    for patient in patients:
        x = sum_patient_vectors(model, patient)
        if x is not None:
            data_map.append(x)
    
    logger.log("Creating lists")
    avg_sims, min_sims = zip(*data_map)

    logger.log("Plotting patient boxplots")
    FileUtils.create_boxplot(logger, avg_sims, "Average patient item/specialty similarity score", "patient_average_similarity_boxplot")
    FileUtils.create_boxplot(logger, min_sims, "Minimum patient item/specialty similarity score", "patient_minimum_similarity_boxplot")

    logger.log("Finished", '!')