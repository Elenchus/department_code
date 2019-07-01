import fasttext
import FileUtils
import math

from gensim.models import KeyedVectors as w2v
from sklearn import cluster

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "mce_test", '/mnt/c/data')
    model = w2v.load_word2vec_format('mce_test.vec', binary = False)
    FileUtils.tsne_plot(logger, model, math.sqrt(math.sqrt(len(model.wv.vocab))))

    X = model[model.wv.vocab]
    k = FileUtils.get_best_cluster_size(logger, X, list(2**i for i in range(1,11)))
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_

    FileUtils.create_scatter_plot(logger, X, labels, "MCE cluster test", 'mce_cluster_')
