import file_utils
import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.neighbors import NearestNeighbors as kNN

logger = file_utils.logger(__name__, f"proposal_2_rsp_cluster", "/mnt/c/data")
filenames = file_utils.get_mbs_files()

# cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "INHOSPITAL", "BILLTYPECD"]
test_cols = ["SPR", "SPR_RSP"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=test_cols)
    assert len(data.columns) == len(test_cols)
    for i in range(len(test_cols)):
        assert data.columns[i] == test_cols[i]

    uniques = data['SPR_RSP'].unique().tolist()
    perplex = round(math.sqrt(math.sqrt(len(uniques))))

    logger.log("Grouping values")
    data = sorted(data.values.tolist())
    groups = itertools.groupby(data, key=lambda x: x[0])

    logger.log("Processing groups")
    words = []
    for uid, group in groups:
        sentence = [str(x[1]) for x in list(group)]
        words.append(sentence)

    model = Word2Vec(words, size = perplex)
    X = model[model.wv.vocab]

    logger.log("Creating t-SNE plot")
    file_utils.tsne_plot(logger, model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")

    logger.log("Creating UMAP")
    file_utils.umap_plot(logger, model, "RSP cluster UMAP")

    logger.log("Performing PCA")
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log("k-means clustering")
    (k, s) = file_utils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,8)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    file_utils.create_scatter_plot(logger, Y, labels, f"RSP clusters", f'RSP_clusters_kmeans')

    logger.log("Calculating GMM")
    n_components = 6
    bgmm = BGMM(n_components=n_components).fit(Y)
    labels = bgmm.predict(Y)
    file_utils.create_scatter_plot(logger, Y, labels, f"BGMM for RSP clusters", f'BGMM_RSP')
    probs = bgmm.predict_proba(Y)
    probs_output = logger.output_path / f'BGMM_probs.txt'
    np.savetxt(probs_output, probs)

    break