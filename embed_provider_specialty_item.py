import os
import FileUtils
import gc
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from FileUtils import tsne_plot
from functools import partial
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM

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
    logger = FileUtils.logger(__name__, "embed_specialty_item", "/mnt/c/data")

    filename = FileUtils.get_mbs_files()[0]
    
    logger.log("Loading parquet file")
    data = pd.read_parquet(filename, columns=['ITEM', 'SPR_RSP']).astype(str)
    data['SPR_RSP'] = data['SPR_RSP'].apply(lambda x: f"RSP_{x}")
    string_list = data.values.tolist()
    no_unique_items = len(data['ITEM'].unique())
    unique_rsp = data['SPR_RSP'].unique()
    no_unique_rsp = len(unique_rsp)
    provider_labels=data['SPR_RSP'].values.tolist()

    del data
    gc.collect()

    logger.log("Embedding vectors")
    size = ceil(sqrt(sqrt(no_unique_items + no_unique_rsp)))

    model = Word2Vec(string_list, size=size, window= 2, min_count=1, workers=5,iter=5)

    del string_list
    gc.collect()

    X = model[model.wv.vocab]

    rsps_not_in_model = []
    for rsp in unique_rsp:
        if rsp not in model.wv.vocab:
            no_unique_rsp = no_unique_rsp - 1

    logger.log("Performing PCA")
    X = model.wv.syn0
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log(f"Getting best cluster size")
    cluster_no = [2, 4, 8, 16, 32, 64, 96, 128, no_unique_rsp, 160, 192, 224, 256]
    k, score = FileUtils.get_best_cluster_size(logger, Y, cluster_no)

    logger.log(f"Clustering with {k} clusters")
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    # FileUtils.create_scatter_plot(logger, Y, labels, f"RSP clusters", f'RSP_clusters_kmeans')

    FileUtils.create_scatter_plot(logger, Y, labels, "Item/provider specialty clusters", "items_and_specialties_k_means")

    logger.log("Calculating GMM")
    n_components = 6
    bgmm = BGMM(n_components=n_components).fit(Y)
    labels = bgmm.predict(Y)
    FileUtils.create_scatter_plot(logger, Y, labels, f"BGMM for RSP clusters", f'BGMM_RSP')
    probs = bgmm.predict_proba(Y)
    probs_output = logger.output_path / f'BGMM_probs.txt'
    np.savetxt(probs_output, probs)

    # FileUtils.create_scatter_plot(logger, Y, provider_labels, "Provider specialty labels", "specialty_labels")

    # logger.log("Re-loading parquet file")
    # data = pd.read_parquet(filename, columns=['PIN', 'ITEM', 'SPR_RSP']).astype(str)

    # patients = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

    # del data
    # gc.collect()

    # logger.log("Summing patient vectors")
    # func = partial(sum_patient_vectors, model)
    # p = Pool(processes=6)
    # data_map = p.imap(func, patients)
    # p.close()
    # p.join()

    # data_map = [i for i in data_map if i]
    
    
    # data_map = []
    # for patient in patients:
    #     x = sum_patient_vectors(model, patient)
    #     if x is not None:
    #         data_map.append(x)
    
    # logger.log("Creating lists")
    # avg_sims, min_sims = zip(*data_map)

    # logger.log("Plotting patient boxplots")
    # FileUtils.create_boxplot(logger, avg_sims, "Average patient item/specialty similarity score", "patient_average_similarity_boxplot")
    # FileUtils.create_boxplot(logger, min_sims, "Minimum patient item/specialty similarity score", "patient_minimum_similarity_boxplot")
    cdv = FileUtils.code_converter()
    for x in list(cdv.valid_rsp_num_values): 
        try: 
            y = model.most_similar(f"RSP_{x}") 
            z = y[0][0] 
            if z[0:4] == 'RSP_': 
                z = z[4:] 
                logger.log(f"{cdv.convert_rsp_num(x)}: {cdv.convert_rsp_num(z)} at {y[0][1]}%") 
            else: 
                logger.log(f"{cdv.convert_rsp_num(x)}: item {z} at {y[0][1]}%") 
        except KeyError: 
            continue 

    logger.log("Finished", '!')