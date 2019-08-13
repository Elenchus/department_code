import FileUtils
import itertools
import pandas as pd
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.neighbors import NearestNeighbors as kNN

logger = FileUtils.logger(__name__, f"proposal_2_rsp_cluster", "/mnt/c/data")
filenames = FileUtils.get_mbs_files()

# cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "INHOSPITAL", "BILLTYPECD"]
test_cols = ["SPR", "SPR_RSP"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=test_cols)
    assert len(data.columns) == len(test_cols)
    for i in range(len(test_cols)):
        assert data.columns[i] == test_cols[i]

    logger.log("Grouping values")
    data = sorted(data.values.tolist())
    groups = itertools.groupby(data, key=lambda x: x[0])

    logger.log("Processing groups")
    words = []
    for uid, group in groups:
        sentence = [str(x[1]) for x in list(group)]
        words.append(sentence)

    model = Word2Vec(words)

    logger.log("Performing PCA")
    pca2d = PCA(n_components=2)
    pca2d.fit(model)
    Y = pca2d.transform(model)

    logger.log("k-means clustering")
    (k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,8)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    FileUtils.create_scatter_plot(logger, Y, labels, f"RSP clusters", f'RSP_clusters_kmeans')
    break