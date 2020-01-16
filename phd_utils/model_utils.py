import math
import itertools
from datetime import datetime
import keras
import pandas as pd
import pygraphviz as pgv
import numpy as np
import umap
from apyori import apriori as apyori
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from matplotlib import markers
from matplotlib import pyplot as plt
from multiprocessing import Pool
from phd_utils.graph_utils import GraphUtils
from phd_utils.code_converter import CodeConverter
from tqdm import tqdm
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture as BGMM

class ModelUtils():
    def __init__(self, logger):
        self.logger = logger
        self.graph_utils = GraphUtils(logger)

    def _apriori_analysis(self, documents, output_file=None, item_list=None, min_support=0.01, min_confidence=0.8, min_lift = 1.1, directed=True):
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
                if item_list is None and item_0 not in d:
                    d[item_0] = {}

                d[item_0][item_1] = None

        if output_file is not None:
            A = pgv.AGraph(data=d, directed=directed)
            A.node_attr['style']='filled'
            A.node_attr['shape'] = 'circle'
            A.node_attr['fixedsize']='true'
            A.node_attr['fontcolor']='#FFFFFF'
            A.node_attr['height']=4
            A.node_attr['width']=4

            A.draw(str(output_file), prog='fdp')

        return d

    def apriori_analysis(self, documents, output_file=None, item_list=None, min_support=0.01, min_confidence=0.8, min_lift = 1.1, directed=True):
        te = TransactionEncoder()
        te_ary = te.fit(documents).transform(documents)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = apriori(df, min_support=min_support, use_colnames=True)

        rules = association_rules(freq_items, metric='confidence', min_threshold=min_confidence)
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
        rules = rules[(rules['lift'] >= min_lift) & (rules['antecedent_len'] == 1) & (rules['consequent_len'] == 1)]

        rules = rules[['antecedents', 'consequents']].values.tolist()
        d={}
        for rule in rules:
            if rule[0] not in d:
                d[rule[0]] = {}

                d[rule[0]][rule[1]] = None

        if output_file is not None:
            A = pgv.AGraph(data=d, directed=directed)
            A.node_attr['style']='filled'
            A.node_attr['shape'] = 'circle'
            A.node_attr['fixedsize']='true'
            A.node_attr['fontcolor']='#FFFFFF'
            A.node_attr['height']=4
            A.node_attr['width']=4

            A.draw(str(output_file), prog='fdp')

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

    def fp_growth_analysis(self, documents, output_file=None, item_list=None, min_support=0.01, min_confidence=0.8, min_lift = 1.1, directed=True):
        te = TransactionEncoder()
        te_ary = te.fit(documents).transform(documents)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = fpgrowth(df, min_support=min_support, use_colnames=True)

        rules = association_rules(freq_items, metric='confidence', min_threshold=min_confidence)
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
        rules = rules[(rules['lift'] >= min_lift) & (rules['antecedent_len'] == 1) & (rules['consequent_len'] == 1)]

        rules = rules[['antecedents', 'consequents']].values.tolist()
        d={}
        for rule in rules:
            if rule[0] not in d:
                d[rule[0]] = {}

                d[rule[0]][rule[1]] = None

        if output_file is not None:
            A = pgv.AGraph(data=d, directed=directed)
            A.node_attr['style']='filled'
            A.node_attr['shape'] = 'circle'
            A.node_attr['fixedsize']='true'
            A.node_attr['fontcolor']='#FFFFFF'
            A.node_attr['height']=4
            A.node_attr['width']=4

            A.draw(str(output_file), prog='fdp')

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