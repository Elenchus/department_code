import math
import itertools
from datetime import datetime
import keras
import pandas as pd
import numpy as np
import umap
from apyori import apriori as apyori
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from matplotlib import markers
from matplotlib import pyplot as plt
from multiprocessing import Pool
from phd_utils.mba_utils import MbaUtils
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture as BGMM
from tqdm import tqdm

class ModelUtils():
    def __init__(self, logger, graph_utils, code_converter):
        self.logger = logger
        self.graph_utils = graph_utils
        self.mba = MbaUtils(code_converter)

    def apriori_analysis(self, documents, output_file=None, item_list=None, min_support=0.01, min_confidence=0.8, min_lift = 1.1, directed=True):
        max_length=2
        rules = apyori(documents, min_support = min_support, min_confidence = min_confidence, min_lift = min_lift, max_length = max_length)
        d = {}
        if item_list is not None:
            for item in item_list:
                d[item] = {}

        for record in tqdm(rules):
            for stat in record[2]: # this will need to change if max_length is not 2
                if not stat[0]:
                    continue
                if not stat[1]:
                    continue
                
                assert len(stat[0]) == 1 #item numbers appear in frozensets -> can't be indexed
                assert len(stat[1]) == 1

                item_0 = next(iter(stat[0]))
                item_1 = next(iter(stat[1]))
                lift=stat[3]
                if item_list is None and item_0 not in d:
                    d[item_0] = {}

                green = '00'
                red_amount = min(int(255 * ((lift - min_lift) / 100)), 255)
                red = '{:02x}'.format(red_amount)
                blue_amount = 255 - red_amount
                blue = '{:02x}'.format(blue_amount)
                
                d[item_0][item_1] = {'color': f"#{red}{green}{blue}"}

        if output_file is not None:
            self.graph_utils.visual_graph(d, output_file, directed=directed)

        return d

    def calculate_BGMM(self, data, n_components, title, filename):
        self.logger.log(f"Calculating GMM with {n_components} components")
        bgmm = BGMM(n_components=n_components).fit(data)
        labels = bgmm.predict(data)
        self.graph_utils.create_scatter_plot(data, labels, f"BGMM {title}", f'BGMM_{title}')
        probs = bgmm.predict_proba(data)
        probs_output = self.logger.output_path / f'BGMM_probs_{title}.txt'
        np.savetxt(probs_output, probs)

    def cartesian_to_polar(self, data):
        polars = []
        for x, y in data:
            r = math.sqrt(x**2 + y **2)
            theta = math.atan(y / x)
            if x < 0:
                theta = theta + math.pi
            elif y < 0:
                theta = theta + (2 * math.pi)

            theta = math.degrees(theta)

            polars.append([r, theta])

        return polars

    def fp_growth_analysis(self, documents, output_file=None, min_support=0.01, min_conviction=1.1, directed=True):
        te = TransactionEncoder()
        te_ary = te.fit(documents).transform(documents)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = fpgrowth(df, min_support=min_support, use_colnames=True)

        rules = association_rules(freq_items, metric='conviction', min_threshold=min_conviction)
        rules['ante_len'] = rules['antecedents'].apply(len)
        rules['con_len'] = rules['consequents'].apply(len)
        rules= rules[(rules['ante_len'] == 1) & (rules['con_len'] == 1)]

        antecedents = [x for x, *_ in rules['antecedents'].values.tolist()]
        consequents = [x for x, *_ in rules['consequents'].values.tolist()]
        
        max_conviction = rules.loc[rules['conviction'] != np.inf, 'conviction'].max()
        conviction = rules['conviction'].replace(np.inf, (max_conviction if max_conviction > 1 else 9999)).values.tolist()
        min_conviction = rules['conviction'].min()

        d={}
        for i, a in enumerate(antecedents):
            if a not in d:
                d[a]={}

            b = consequents[i]
            if a == b:
                continue

            d[a][b] = None

            green = '00'
            red_amount = int(255 * ((conviction[i] - min_conviction) / (max_conviction - min_conviction)))
            red = '{:02x}'.format(red_amount)
            blue_amount = 255 - red_amount
            blue = '{:02x}'.format(blue_amount)
            d[a][b] = {'color': f"#{red}{green}{blue}"} #, 'label': str(conviction[i])

        if output_file is not None:
            self.graph_utils.visual_graph(d, output_file, directed=directed)

        return d

    def get_best_cluster_size(self, X, clusters):
        '''measure silhouette scores for the given cluster sizes and return the best k and its score'''
        self.logger.log("Getting best k-means cluster size with average silhouette score")
        avg_sil = []
        for n in clusters:
            kmeans = cluster.KMeans(n_clusters=n)
            kmeans.fit(X)
            labels = kmeans.labels_
            silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
            avg_sil.append(silhouette_score)
            self.logger.log(f"n = {n}, silhouette score = {silhouette_score}")

        k = clusters[avg_sil.index(max(avg_sil))]
        max_n = math.ceil(max(avg_sil) * 100)
        self.logger.log(f"Max silhouette score with {k} clusters")

        return (k, max_n)

    def k_means_cluster(self, data, max_clusters, title, filename, labels=None):
        '''Creates and saves k-means clusters using the best cluster size based on silhouette score'''
        self.logger.log("k-means clustering")
        max_binary_test = min((math.floor(math.log2(len(data) / 3)), math.floor(math.log2(max_clusters))))
        (k, s) = self.get_best_cluster_size(data, list(2**i for i in range(1,max_binary_test)))
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)
        if labels is None:
            labels = kmeans.labels_
            legend_names = None
        else:
            labels, legend_names = pd.factorize(labels)

        self.graph_utils.create_scatter_plot(data, labels, f"{title} with {s}% silhoutte score", filename, legend_names)

        return kmeans

    def generate_sentences_from_group(self, data, column, convert_to_string=True):
        documents = []
        for name, group in data:
            items = group[column].values.tolist()
            if convert_to_string:
                items = [str(item) for item in items]

            documents.append(items)

        return documents

    def get_outlier_indices(self, data):
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        list_of_outlier_indices = []
        for i in range(len(data)):
            if data[i] > q75 + 1.5 * iqr or data[i] < q25 - 1.5 * iqr:
                list_of_outlier_indices.append(i)

        return list_of_outlier_indices

    def multiprocess_generator(self, function, generator, threads=6):
        pool = Pool(processes=threads)
        result = []
        while True:
            r = pool.map(function, itertools.islice(generator, threads))
            if r:
                result.extend(r)
            else:
                break

        return result
        
    def one_layer_autoencoder_prediction(self, data, activation_function):
        self.logger.log("Autoencoding")
        act = "linear"
        input_layer = keras.layers.Input(shape=(data.shape[1], ))
        enc = keras.layers.Dense(2, activation=act)(input_layer)
        dec = keras.layers.Dense(data.shape[1], activation=act)(enc)
        autoenc = keras.Model(inputs=input_layer, outputs=dec)
        autoenc.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        autoenc.fit(data, data, epochs = 1000, batch_size=16, shuffle=True, validation_split=0.1, verbose=0)
        encr = keras.Model(input_layer, enc)
        Y = encr.predict(data)

        return Y

    def pairwise_neg_cor_low_sup(self, items, documents, max_support=0.01, hc=0.8):
        group_count = len(documents)
        max_occurrences = max_support * group_count
        counts = pd.DataFrame(0, index=items, columns=items)
        for doc in documents:
            for item in doc:
                for item_2 in doc:

                    counts.at[item, item_2] += 1
        
        pairs = {}
        for item in items:
            for item_2 in items:
                if item == item_2:
                    continue

                count = counts.at[item, item_2]
                count_1 = counts.at[item, item]
                count_2 = counts.at[item_2, item_2]
                if count > max_occurrences:
                    continue

                s_1 = count_1 / group_count
                s_2 = count_2 / group_count
                s = count / group_count
                cross_support=min(s_1, s_2) / max(s_1, s_2)
                if cross_support < hc:
                    continue

                if s >= s_1 * s_2:
                    continue

                if item not in pairs:
                    pairs[item] = {}

                pairs[item][item_2] = None
                
        return pairs

    def pca_2d(self, data):
        self.logger.log("Performing PCA")
        pca2d = PCA(n_components=2)
        pca2d.fit(data)
        output = pca2d.transform(data)

        return output
    
    def t_sne(self, model, perplex, title):
        '''create and save a t-SNE plot'''
        self.logger.log("Creating t-SNE plot")
        self.logger.log("Getting labels and tokens for t-SNE")
        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)
    
        self.logger.log(f"Creating TSNE model with perplexity {perplex}")
        tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        self.graph_utils.plot_tsne(new_values, labels, title)

    def sum_and_average_vectors(self, model, groups, ignore_missing = False):
        self.logger.log("Summing and averaging vectors")
        item_dict = {}
        for key, group in groups:
            if isinstance(group, itertools._grouper):
                _, group = zip(*list(group))

            for item in group:
                item = str(item)
                if item not in model.wv.vocab:
                    if ignore_missing:
                        continue
                    else:
                        raise KeyError(f"Item {item} is not in model vocab")
                    
                keys = item_dict.keys()
                if key not in keys:
                    item_dict[key] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
                else:
                    item_dict[key]['Sum'] += model[item]
                    item_dict[key]['Average'] = ((item_dict[key]['Average'] * item_dict[key]['n']) + model[item]) / (item_dict[key]['n'] + 1)
    
        sums = [item_dict[x]['Sum'] for x in item_dict.keys()]
        avgs = [item_dict[x]['Average'] for x in item_dict.keys()]

        return (sums, avgs)

    def u_map(self, model, title):
        self.logger.log("Creating UMAP")
        '''create and save a umap plot'''
        labels = []
        tokens = []

        self.logger.log("Extracting labels and token")
        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)

        self.logger.log("Creating UMAP")
        reducer = umap.UMAP(verbose=True)
        embedding = reducer.fit_transform(tokens)

        self.graph_utils.plot_umap(embedding, title) 