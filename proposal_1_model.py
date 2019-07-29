import FileUtils
import math
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors as w2v
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as kNN

input_file = 'prop_1_hip_2003_epoch_1_dim_491_day.vec'
logger = FileUtils.logger(__name__, f"proposal_1_analysis_{input_file}", '/mnt/c/data')
logger.log(f'Opening {input_file}')
model = w2v.load_word2vec_format(input_file, binary = False)

logger.log("Creating TSNE plot")
perplex = math.ceil(math.sqrt(math.sqrt(len(model.wv.vocab))))
FileUtils.tsne_plot(logger, model, perplex, f"t-SNE plot of hip replacement patients with perplexity {perplex}")

logger.log("Creating UMAP plot")
FileUtils.umap_plot(logger, model, f"UMAP plot of hip replacement patients")

logger.log("Performing PCA")
X = model.wv.syn0
pca2d = PCA(n_components=2)
pca2d.fit(X)
Y = pca2d.transform(X)

logger.log("k-means clustering")
(k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,8)))
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(Y)
labels = kmeans.labels_
FileUtils.create_scatter_plot(logger, Y, labels, f"MCE hip replacement patients test. Silhoutte score {s}%", f'mce_mimic')

logger.log("Calculating distances from k-means clusters")
all_distances = kmeans.transform(Y) 
assert len(all_distances) == len(labels)
cluster_indices = {i: np.where(labels == i)[0] for i in range(kmeans.n_clusters)}
for i in cluster_indices.keys():
    cluster_distances = pd.Series([all_distances[x][i] for x in cluster_indices[i]])

    q1 = cluster_distances.quantile(0.25)
    q3 = cluster_distances.quantile(0.75)
    iqr = q3 - q1

    outlier_count = 0
    cluster_outlier_file = logger.output_path / "kmeans_outliers.txt"
    for idx, x in enumerate(cluster_distances):
        if x >= q3 + (1.5 * iqr):
            outlier_count = outlier_count + 1
            with open(cluster_outlier_file, 'a') as f:
                f.write(f'{model.index2word[cluster_indices[i][idx]]}: {x}\r\n') 

logger.log(f"{outlier_count} outliers detected")
# {i: Y[np.where(labels == i)] for i in range(kmeans.n_clusters)}

logger.log("Calculating unsupervised 1NN distance")
nn_calc = kNN(n_neighbors=1)
nn_calc.fit(Y) # 2d
distances, _ = nn_calc.kneighbors(n_neighbors=1)
distances = distances.tolist()
distances = [x[0] for x in distances]
distances = pd.Series(distances)
q1 = distances.quantile(0.25)
q3 =  distances.quantile(0.75)
iqr = q3 - q1
kNN_outlier_file = logger.output_path / "1NN_outliers.txt"
outlier_count = 0
out_labels = [0] * len(Y)
with open(kNN_outlier_file, 'w+') as f:
    for idx, distance in enumerate(distances):
        if distance >= q3 + (1.5 * iqr):
            f.write(f'{model.index2word[idx]}: {distance}\r\n')
            outlier_count = outlier_count + 1
            out_labels[idx] = 1
logger.log(f"{outlier_count} outliers detected")

logger.log("Plotting 1NN outliers")
FileUtils.create_scatter_plot(logger, Y, out_labels, "1NN cluster and outliers", "1NN")

logger.log("Calculating 1NN cosine-similarity distances from word vector similarity")
nearest = {}
for word in model.vocab.keys():
    nearest[word] = model.most_similar(word)[0][1]

values = list(nearest.values())
distances = pd.Series(values)
q1 = distances.quantile(0.25)
q3 = distances.quantile(0.75)
iqr = q3 - q1

outliers = []
keys  = nearest.keys()
values, keys = (list(t) for t in zip(*sorted(zip(values, keys))))
cosine_outlier_file = logger.output_path / "word2vec_cosine_similarity_outliers.txt"
outlier_count = 0
for i in range(len(keys)):
    if values[i] <= q1 - (0.5 * iqr):
        outlier_count = outlier_count + 1
        with open(cosine_outlier_file, 'a') as f:
            f.write(f'{keys[i]}: {values[i]}\r\n')

logger.log(f"{outlier_count} outliers detected")