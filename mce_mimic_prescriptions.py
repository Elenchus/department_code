import FileUtils
import math

from gensim.models import KeyedVectors as w2v
from sklearn import cluster
from sklearn.decomposition import PCA

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "mce_mimic_prescriptions", '/mnt/c/data')
    logger.log("Loading model")
    model = w2v.load_word2vec_format('mimic_III_procedures.vec', binary = False)

    logger.log("Creating TSNE plot")
    FileUtils.tsne_plot(logger, model, math.sqrt(math.sqrt(len(model.wv.vocab))), "t-SNE plot of MIMIC Prescription Item")

    logger.log("Performing PCA")
    X = model.wv.syn0
    pca2d = PCA(n_components=2)
    pca2d.fit(X)
    Y = pca2d.transform(X)

    logger.log("k-means clustering")
    k = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,10)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(Y)
    labels = kmeans.labels_
    FileUtils.create_scatter_plot(logger, Y, labels, "MCE MIMIC PRESCRIPTIONS test", 'mce_mimic_prescriptions')
