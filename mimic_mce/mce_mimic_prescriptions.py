import FileUtils
import math

from gensim.models import KeyedVectors as w2v
from sklearn import cluster
from sklearn.decomposition import PCA

if __name__ == "__main__":
    test_type = "diagnoses"
    logger = FileUtils.logger(__name__, f"mce_mimic_{test_type}", '/mnt/c/data')
    logger.log("Loading model")
    model = w2v.load_word2vec_format(f'MIMIC_{test_type}.vec', binary = False)

    logger.log("Creating TSNE plot")
    perplex = math.ceil(math.sqrt(math.sqrt(len(model.wv.vocab))))
    FileUtils.tsne_plot(logger, model, perplex, f"t-SNE plot of MIMIC {test_type.capitalize()} Codes with perplexity {perplex}")

    logger.log("Performing PCA")
    X = model.wv.syn0
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log("k-means clustering")
    (k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,10)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    FileUtils.create_scatter_plot(logger, Y, labels, f"MCE MIMIC {test_type} test. Silhoutte score {s}%", f'mce_mimic_{test_type}')
