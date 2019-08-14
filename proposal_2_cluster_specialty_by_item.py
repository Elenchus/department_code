import FileUtils
import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.neighbors import NearestNeighbors as kNN

logger = FileUtils.logger(__name__, f"proposal_2_rsp_item_cluster", "/mnt/c/data")
filenames = FileUtils.get_mbs_files()

full_cols = ["ITEM", "SPR_RSP", "NUMSERV"]
cols = ["ITEM", "SPR_RSP"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=full_cols)
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
    data = data.drop(['NUMSERV'], axis = 1)
    for i in range(len(cols)):
        assert data.columns[i] == cols[i]

    no_unique_rsps = len(data['ITEM'].unique())
    perplex = round(math.sqrt(math.sqrt(no_unique_rsps)))

    logger.log("Grouping items")
    data = sorted(data.values.tolist(), key = lambda x: x[0])
    groups = itertools.groupby(data, key = lambda x: x[0])
    sentences = []
    for rsp, group in groups:
        sentences.append([f"RSP_{x[1]}" for x in list(group)])

    model = w2v(sentences=sentences, min_count=20, size = perplex, iter = 5)

    X = model[model.wv.vocab]

    logger.log("Creating t-SNE plot")
    FileUtils.tsne_plot(logger, model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")

    logger.log("Creating UMAP")
    FileUtils.umap_plot(logger, model, "RSP cluster UMAP")

    # Y = X
    logger.log("Performing PCA")
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log("k-means clustering")
    (k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,8)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    FileUtils.create_scatter_plot(logger, Y, labels, f"RSP clusters", f'RSP_clusters_kmeans')

    logger.log("Calculating GMM")
    n_components = 6
    bgmm = BGMM(n_components=n_components).fit(Y)
    labels = bgmm.predict(Y)
    FileUtils.create_scatter_plot(logger, Y, labels, f"BGMM for RSP clusters", f'BGMM_RSP')
    probs = bgmm.predict_proba(Y)
    probs_output = logger.output_path / f'BGMM_probs.txt'
    np.savetxt(probs_output, probs)

    cdv = FileUtils.code_converter()
    for x in list(cdv.valid_rsp_num_values): 
        try: 
            y = model.most_similar(f"RSP_{x}") 
            z = y[0][0] 
            if z[0:4] == 'RSP_': 
                z = z[4:] 
                logger.log(f"{cdv.convert_rsp_num(x)}: {cdv.convert_rsp_num(z)} at {round(y[0][1], 2)}") 
            else: 
                logger.log(f"{cdv.convert_rsp_num(x)}: item {z} at {round(y[0][1], 2)}") 
        except KeyError: 
            continue 

    break