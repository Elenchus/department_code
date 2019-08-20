import os
from phd_utils import file_utils, graph_utils, model_utils
import gc
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from phd_utils.graph_utils import tsne_plot
from functools import partial
from gensim.models import Word2Vec
from math import ceil, sqrt
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn import cluster

def sum_similarity_pair(model, group):
    similarities = []
    id = group[0]
    for claims in group:
        if claims == id:
            continue
        for claim in claims:
            if len(claim) == 2:
                if (claim[0] in model.wv.vocab) and (claim[1] in model.wv.vocab):
                    similarities.append(model.similarity(claim[0], claim[1]))
            else:
                print(claim)

    if len(similarities) != 0:
        return (sum(similarities) / len(similarities), min(similarities), id)

if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "cluster_specialty_by_item_similarity")

    filename = file_utils.get_mbs_files()[0]
    
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

    logger.log("Re-loading parquet file")
    data = pd.read_parquet(filename, columns=['SPR_RSP', 'ITEM']).astype(str)

    specialties = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

    del data
    gc.collect()

    logger.log("Summing specialty similarities")    

    data_map = []
    for specialty in specialties:
        x = sum_similarity_pair(model, specialty)
        if x is not None:
            data_map.append(x)

    logger.log("Creating lists")
    avg_sims, min_sims, ids = zip(*data_map)

    logger.log("Plotting patient boxplots")
    graph_utils.create_boxplot(logger, avg_sims, "Average specialty similarity score", "specialty_average_similarity_boxplot")
    graph_utils.create_boxplot(logger, min_sims, "Minimum specialty similarity score", "specialty_minimum_similarity_boxplot")

    logger.log("Finding outlier specialties by average score")
    outlier_indices = model_utils.get_outlier_indices(avg_sims)
    list_of_outlier_number = [ids[i] for i in outlier_indices]

    logger.log(f"Outlier specialties by average score: {str(list_of_outlier_number)}")

    
    logger.log("Finding outlier specialties by minimum score")
    outlier_indices = model_utils.get_outlier_indices(min_sims)
    list_of_outlier_min = [ids[i] for i in outlier_indices]

    logger.log(f"Outlier specialties by minimum score: {str(list_of_outlier_min)}")

    frequency_of_specialties = []
    claims = pd.read_parquet(filename, columns=['SPR_RSP']).values.tolist()
    for spec in unique_rsp:
        frequency_of_specialties.append(claims.count(int(spec)))

    graph_utils.create_boxplot(logger, frequency_of_specialties, "Number of claims by specialty", "frequency_of_specialty")
    
    logger.log("Finding most frequent specialties by > q75 + 1.5 * iqr")
    spr_rsp = file_utils.CodeConverter()
    q75, q25 = np.percentile(frequency_of_specialties, [75, 25])
    iqr = q75 - q25
    list_of_top_specialties = []
    for i in range(len(frequency_of_specialties)):
        if frequency_of_specialties[i] > q75 + 1.5 * iqr:
            list_of_top_specialties.append((frequency_of_specialties[i], unique_rsp[i]))

    list_of_top_specialties.sort()
    for spec in list_of_top_specialties:
        logger.log(f"{spr_rsp.convert_rsp_num(spec[1])} - {spec[0]}")

    logger.log("Finished", '!')