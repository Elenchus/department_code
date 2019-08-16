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

logger = FileUtils.logger(__name__, f"proposal_2_cluster_providers", "/mnt/c/data")
filenames = FileUtils.get_mbs_files()

full_cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "NUMSERV"]
# full_cols = ["ITEM", "SPR_RSP", "NUaMSERV", "INHOSPITAL"]
cols = ["SPR", "ITEM"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=full_cols)
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
    data["SPR"] = data["SPR"].map(str) + "_" + data["SPRPRAC"].map(str)
    # no_unique_rsps = len(data['SPR_RSP'].unique())
    data = data.drop(['NUMSERV', 'SPR_RSP', "SPRPRAC"], axis = 1)
    assert len(data.columns) == len(cols)
    for i in range(len(cols)):
        assert data.columns[i] == cols[i]

    no_unique_items = len(data['ITEM'].unique())
    perplex = round(math.sqrt(math.sqrt(no_unique_items)))

    logger.log("Grouping providers")
    data = sorted(data.values.tolist(), key = lambda x: x[0])
    groups = itertools.groupby(data, key = lambda x: x[0])
    sentences = []
    max_sentence_length = 0
    for provider, group in groups:
        sentence = sorted(list(str(x[1]) for x in list(group)))
        # sentence = list(set(str(x[1]) for x in list(group)))
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)

        sentences.append(sentence)
        # sentences.append([f"RSP_{x[1]}" for x in list(group)])

    max_sentence_length = 2
    model = w2v(sentences=sentences, min_count=20, size = perplex, iter = 5, window=max_sentence_length)

    logger.log("Creating vectors for patients")
    patient_dict = {}
    groups = itertools.groupby(data, key = lambda x: x[0])
    for pid, group in groups:
        _, group = zip(*list(group))
        for item in group:
            keys = patient_dict.keys()
            if pid not in keys:
                patient_dict[pid] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
            else:
                patient_dict[pid]['Sum'] += model[item]
                patient_dict[pid]['Average'] = ((patient_dict[pid]['Average'] * patient_dict[pid]['n']) + model[item]) / (patient_dict[pid]['n'] + 1)
                patient_dict[pid]['n'] += 1
    
    sums = [patient_dict[x]['Sum'] for x in patient_dict.keys()]
    avgs = [patient_dict[x]['Average'] for x in patient_dict.keys()]
    # X = model[model.wv.vocab]

    # logger.log("Creating t-SNE plot")
    # FileUtils.tsne_plot(logger, model, perplex, f"t-SNE plot of provider clusters with perplex {perplex}")

    # logger.log("Creating UMAP")
    # FileUtils.umap_plot(logger, model, "provider cluster UMAP")

    # Y = X

    for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
        logger.log("Performing PCA")
        pca2d = PCA(n_components=2)
        pca2d.fit(matrix)
        Y = pca2d.transform(matrix)

        logger.log("k-means clustering")
        (k, s) = FileUtils.get_best_cluster_size(logger, Y, list(2**i for i in range(1,7)))
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(Y)
        labels = kmeans.labels_
        FileUtils.create_scatter_plot(logger, Y, labels, f"provider {name} clusters", f'provider_{name}')

        logger.log("Calculating GMM")
        n_components = 6
        bgmm = BGMM(n_components=n_components).fit(Y)
        labels = bgmm.predict(Y)
        FileUtils.create_scatter_plot(logger, Y, labels, f"BGMM for provider {name} clusters", f'BGMM_provider_{name}')
        probs = bgmm.predict_proba(Y)
        probs_output = logger.output_path / f'BGMM_probs_{name}.txt'
        np.savetxt(probs_output, probs)
    
    # logger.log("Calculating cosine similarities")
    # cdv = FileUtils.code_converter()
    # output_file = logger.output_path / "Most_similar.csv"
    # with open(output_file, 'w+') as f:
    #     f.write("provider,Most similar to,Cosine similarity\r\n")
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