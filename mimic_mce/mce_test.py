from phd_utils import file_utils, graph_utils
import math

from gensim.models import KeyedVectors as w2v
from sklearn import cluster
from sklearn.decomposition import PCA

if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "mce_test", '/mnt/c/data')
    logger.log("Loading model")
    model = w2v.load_word2vec_format('mce_test_4.vec', binary = False)

    logger.log("Creating TSNE plot")
    graph_utils.tsne_plot(logger, model, math.sqrt(math.sqrt(len(model.wv.vocab))), "t-SNE plot of PBS Item/Drug Type/Provider Specialty")

    logger.log("Performing PCA")
    X = model.wv.syn0
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log("k-means clustering")
    k = graph_utils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,10)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    graph_utils.create_scatter_plot(logger, Y, labels, "MCE cluster test", 'mce_cluster_')
