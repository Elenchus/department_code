from phd_utils import file_utils, graph_utils
import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.neighbors import NearestNeighbors as kNN

logger = file_utils.Logger(__name__, f"proposal_2_rsp_item_cluster_take_2", "/mnt/c/data")
filenames = file_utils.get_mbs_files()

full_cols = ["SPR", "ITEM", "SPR_RSP", "NUMSERV"]
# full_cols = ["ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
cols = ["SPR", "ITEM"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=full_cols)
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
    data["SPR"] = data["SPR"].map(str) + "_" + data["SPR_RSP"].map(str)
    # data["SPR_RSP"] = data["SPR_RSP"].map(str) + data["INHOSPITAL"].map(str)
    # data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)
    data = data.drop(['NUMSERV', 'SPR_RSP'], axis = 1)
    assert len(data.columns) == len(cols)
    for i in range(len(cols)):
        assert data.columns[i] == cols[i]

    no_unique_rsps = len(data['SPR'].unique())
    perplex = round(math.sqrt(math.sqrt(no_unique_rsps)))

    logger.log("Grouping providers")
    data = sorted(data.values.tolist(), key = lambda sentence: sentence[0])
    groups = itertools.groupby(data, key = lambda sentence: sentence[0])
    sentences = []
    max_sentence_length = 0
    for _, group in groups:
        # sentence = list(set(str(x[1]) for x in list(group)))
        sentence = list(str(x[1]) for x in list(group))
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)

        sentences.append(sentence)
        # sentences.append([f"RSP_{x[1]}" for x in list(group)])

    model = w2v(sentences=sentences, min_count=20, size = perplex, iter = 5, window=max_sentence_length)

    X = model[model.wv.vocab]

    logger.log("Creating RSP vectors")
    rsp_dict = {}
    del data
    data = sorted(pd.read_parquet(filename, columns=['SPR_RSP', 'ITEM']).values.tolist())
    groups = itertools.groupby(data, key = lambda x: x[0])
    for rsp, group in groups:
        _, group = zip(*list(group))
        for item in group:
            item = str(item)
            if item not in model.wv.vocab:
                continue
                
            keys = rsp_dict.keys()
            if rsp not in keys:
                rsp_dict[rsp] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
            else:
                rsp_dict[rsp]['Sum'] += model[item]
                rsp_dict[rsp]['Average'] = ((rsp_dict[rsp]['Average'] * rsp_dict[rsp]['n']) + model[item]) / (rsp_dict[rsp]['n'] + 1)

    logger.log("Creating t-SNE plot")
    graph_utils.tsne_plot(logger, model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")

    logger.log("Creating UMAP")
    graph_utils.umap_plot(logger, model, "RSP cluster UMAP")

    sums = [rsp_dict[x]['Sum'] for x in rsp_dict.keys()]
    avgs = [rsp_dict[x]['Average'] for x in rsp_dict.keys()]
    # Y = X
    for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
        logger.log("Performing PCA")
        pca2d = PCA(n_components=2)
        pca2d.fit(matrix)
        Y = pca2d.transform(matrix)

        logger.log("k-means clustering")
        (k, s) = graph_utils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,7)))
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(Y)
        labels = kmeans.labels_
        graph_utils.create_scatter_plot(logger, Y, labels, f"RSP {name} clusters", f'RSP_clusters_kmeans_{name}')

        logger.log("Calculating GMM")
        n_components = 6
        bgmm = BGMM(n_components=n_components).fit(Y)
        labels = bgmm.predict(Y)
        graph_utils.create_scatter_plot(logger, Y, labels, f"BGMM for RSP clusters_{name}", f'BGMM_RSP_{name}')
        probs = bgmm.predict_proba(Y)
        probs_output = logger.output_path / f'BGMM_probs_{name}.txt'
        np.savetxt(probs_output, probs)

        # logger.log("Calculating cosine similarities")
        # cdv = FileUtils.code_converter()
        # output_file = logger.output_path / f"Most_similar_{name}.csv"
        # with open(output_file, 'w+') as f:
        #     f.write("RSP,Most similar to,Cosine similarity\r\n")
        #     for rsp in list(cdv.valid_rsp_num_values): 
        #         try: 
        #             y = model.most_similar(str(rsp)) 
        #             z = y[0][0] 
        #             f.write(f"{cdv.convert_rsp_num(rsp),cdv.convert_rsp_num(z)},{round(y[0][1], 2)}\r\n") 
        #         except KeyError as err: 
        #             continue
        #         except Exception:
        #             raise

    break