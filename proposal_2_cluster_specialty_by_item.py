import FileUtils
import itertools
import keras
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
# full_cols = ["ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
cols = ["ITEM", "SPR_RSP"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=full_cols)
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
    # data["SPR_RSP"] = data["SPR_RSP"].map(str) + data["INHOSPITAL"].map(str)
    # data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)
    data = data.drop(['NUMSERV'], axis = 1)
    assert len(data.columns) == len(cols)
    for i in range(len(cols)):
        assert data.columns[i] == cols[i]

    no_unique_items = len(data['ITEM'].unique())
    perplex = round(math.sqrt(math.sqrt(no_unique_items)))

    logger.log("Grouping items")
    data = sorted(data.values.tolist(), key = lambda sentence: sentence[0])
    groups = itertools.groupby(data, key = lambda sentence: sentence[0])
    sentences = []
    max_sentence_length = 0
    for rsp, group in groups:
        sentence = list(set(str(x[1]) for x in list(group)))
        # sentence = list(str(x[1]) for x in list(group))
        if len(sentence) <= 1:
            continue

        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)

        sentences.append(sentence)
        # sentences.append([f"RSP_{x[1]}" for x in list(group)])

    model = w2v(sentences=sentences, min_count=20, size = perplex, iter = 5, window=max_sentence_length)

    X = model[model.wv.vocab]

    # logger.log("Creating t-SNE plot")
    # FileUtils.tsne_plot(logger, model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")

    # logger.log("Creating UMAP")
    # FileUtils.umap_plot(logger, model, "RSP cluster UMAP")

    # Y = X
    # logger.log("Performing PCA")
    # pca2d = PCA(n_components=2)
    # pca2d.fit(X)
    # Y = pca2d.transform(X)

    logger.log("Autoencoding")
    input_layer = keras.layers.Input(shape=(X.shape[1], ))
    enc = keras.layers.Dense(2, activation='sigmoid')(input_layer)
    dec = keras.layers.Dense(X.shape[1], activation='sigmoid')(enc)
    autoenc = keras.Model(inputs=input_layer, outputs=dec)
    autoenc.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    encr = keras.Model(input_layer, enc)
    Y = encr.predict(X)

    logger.log("k-means clustering")
    (k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,7)))
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

    logger.log("Calculating cosine similarities")
    cdv = FileUtils.code_converter()
    output_file = logger.output_path / "Most_similar.csv"
    with open(output_file, 'w+') as f:
        f.write("RSP,Most similar to,Cosine similarity\r\n")
        for rsp in list(cdv.valid_rsp_num_values): 
            try: 
                y = model.most_similar(str(rsp)) 
                z = y[0][0] 
                f.write(f"{cdv.convert_rsp_num(rsp),cdv.convert_rsp_num(z)},{round(y[0][1], 2)}\r\n") 
            except KeyError as err: 
                continue
            except Exception:
                raise

    break