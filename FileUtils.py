import os
import sys
import atexit
import logging
import math
import numpy as np
import pandas as pd
# from cuml import UMAP as umap
import umap

from datetime import datetime
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if sys.platform == "win32":
    path = 'C:\\Data\\'
else:
    path = '/home/elm/data/'

mbs_header = ['PIN', 'DOS', 'PINSTATE', 'SPR', 'SPR_RSP', 'SPRPRAC', 'SPRSTATE', 'RPR', 'RPRPRAC', 'RPRSTATE', 'ITEM', 'NUMSERV', 'MDV_NUMSERV', 'BENPAID', 'FEECHARGED', 'SCHEDFEE', 'BILLTYPECD', 'INHOSPITAL', 'SAMPLEWEIGHT']
pbs_header = ['PTNT_ID', 'SPPLY_DT', 'ITM_CD', 'PBS_RGLTN24_ADJST_QTY', 'BNFT_AMT', 'PTNT_CNTRBTN_AMT', 'SRT_RPT_IND', 'RGLTN24_IND', 'DRG_TYP_CD', 'MJR_SPCLTY_GRP_CD', 'UNDR_CPRSCRPTN_TYP_CD', 'PRSCRPTN_CNT', 'PTNT_CTGRY_DRVD_CD', 'PTNT_STATE']

def categorical_plot_group(logger, x, y, legend_labels, title, filename):
    logger.log(f"Plotting bar chart: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y)):
        ax.scatter(x[i], y[i], label=legend_labels[i])
    
    # plt.xticks(range(x[0]), (str(i) for i in x[0]))
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ttl = fig.suptitle(title)
    save_plt_fig(logger, fig, filename, (lgd,ttl, ))

def create_boxplot(logger, data, title, filename):
    logger.log(f"Plotting boxplot: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    ax.suptitle(title)
    save_plt_fig(logger, fig, filename)

def create_boxplot_group(logger, data, labels, title, filename):
    logger.log(f"Plotting boxplot group: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    fig.suptitle(title)
    ax.set_xticklabels(labels)
    save_plt_fig(logger, fig, filename)

def create_scatter_plot(logger, data, labels, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
    legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster no.", bbox_to_anchor=(1, 0.5))
    ttl = fig.suptitle(title)

    save_plt_fig(logger, fig, filename, [ttl, legend])
    

def get_best_cluster_size(logger, X, clusters):
    logger.log("Getting best k-means cluster size with average silhouette score")
    avg_sil = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        avg_sil.append(silhouette_score)
        logger.log(f"n = {n}, silhouette score = {silhouette_score}")

    k = clusters[avg_sil.index(max(avg_sil))]
    max_n = math.ceil(max(avg_sil) * 100)
    logger.log(f"Max silhouette score with {k} clusters")

    return (k, max_n)

def get_mbs_files():
    mbs_path = path + 'MBS_Patient_10/'
    files = [mbs_path + f for f in os.listdir(mbs_path) if f.lower().endswith('.parquet')]

    return files

def get_outlier_indices(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    list_of_outlier_indices = []
    for i in range(len(data)):
        if data[i] > q75 + 1.5 * iqr:
            list_of_outlier_indices.append(i)

    return list_of_outlier_indices

def get_pbs_files():
    pbs_path = path + 'PBS_Patient_10/'
    files = [pbs_path + f for f in os.listdir(pbs_path) if f.lower().endswith('.parquet')]

    return files

def save_plt_fig(logger, fig, filename, bbox_extra_artists=None):
    current = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = f"{filename}_{current}"
    if logger != None:
        output_path = logger.output_path / output_path

    if bbox_extra_artists == None:
        fig.savefig(output_path)
    else:
        fig.savefig(output_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

    plt.close(fig)

def tsne_plot(logger, model, perplex, title):
    logger.log("Getting labels and tokens for t-SNE")
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    logger.log(f"Creating TSNE model with perplexity {perplex}")
    tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    logger.log(f"Plotting TSNE figure")
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    for i in range(len(x)):
        ax.scatter(x[i],y[i])
    for i in range(len(x)):
        ax.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    fig.suptitle(title)
    
    name = "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    path = logger.output_path / name
    logger.log(f"Saving TSNE figure to {path}")
    fig.savefig(path)

def umap_plot(logger, model, title):
    labels = []
    tokens = []

    logger.log("Extracting labels and token")
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    logger.log("Creating UMAP")
    reducer = umap.UMAP(verbose=True)
    embedding = reducer.fit_transform(tokens)

    logger.log("Plotting UMAP")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
    # ax.gca().set_aspect('equal', 'datalim')
    fig.suptitle(title)

    name = "UMAP_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    path = logger.output_path / name
    fig.savefig(path)

class code_converter:
    def __init__(self):
        rsp_filename = 'SPR_RSP.csv'
        pbs_item_filename = 'pbs-item-drug-map.csv'

        if not os.path.isfile(rsp_filename):
            raise OSError("Cannot find SPR_RSP.csv - please put it in the same folder as FileUtils")

        if not os.path.isfile(pbs_item_filename):
            raise OSError("Cannot find pbs-item-drug-map.csv - please put it in the same folder as FileUtils")

        self.rsp_table = pd.read_csv(rsp_filename)
        self.pbs_item_table = pd.read_csv(pbs_item_filename, dtype=str, encoding = "latin")
        self.valid_rsp_num_values = self.rsp_table['SPR_RSP'].unique()
        self.valid_rsp_str_values = self.rsp_table['Label'].unique()

    def convert_pbs_code(self, code):

        return self.pbs_item_table.loc[self.pbs_item_table['ITEM_CODE'] == code]

    def convert_rsp_num(self, rsp):
        if int(rsp) not in self.valid_rsp_num_values:
            raise ValueError(f"{rsp} is not a valid SPR_RSP")

        return self.rsp_table.loc[self.rsp_table['SPR_RSP'] == int(rsp)]['Label'].values.tolist()[0]

    def convert_rsp_str(self, rsp):
        if str(rsp) not in self.valid_rsp_str_values:
            raise ValueError(f"{rsp} is not a valid name")

        return self.rsp_table.loc[self.rsp_table['Label'] == str(rsp)]['SPR_RSP'].values.tolist()[0]
    

class logger:
    def __init__(self, name, test_name, copy_path = None):
        self.copy_path = copy_path
        self.test_name = test_name
        self.output_path = self.create_output_folder(test_name)
        self.logger = logging.getLogger(name)
        atexit.register(self.finalise)
        # handler = logging.StreamHandler(stream=sys.stdout)
        # self.logger.addHandler(handler)
        sys.excepthook = self.handle_exception
        self.file_name = self.output_path / f"{test_name}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename = self.file_name, filemode = 'w+')
        self.log(f"Starting {test_name}")

    def finalise(self):
        if self.copy_path != None:
            if type(self.copy_path) != str:
                raise("Copy path must be a string directory")
            
            path = Path(self.copy_path)

            if not path.exists():
                raise(f"Cannot find {self.copy_path}")

            self.log("Copying data folder")
            current = datetime.now().strftime("%Y%m%dT%H%M%S")
            current = f"{self.test_name}_{current}"
            copy_folder = path / current
            os.mkdir(copy_folder)

            copy_tree(self.output_path.absolute().as_posix(), copy_folder.absolute().as_posix())

            self.log("Done")


    def create_output_folder(self, test_name):
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_folder = Path(os.getcwd()) / "Output" / f"{test_name}_{current}"
        os.makedirs(output_folder)

        return output_folder

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        raise(exc_type(exc_value))

    def log(self, line, line_end = '...'):
        print(f"{datetime.now()} {line}{line_end}")
        self.logger.info(line)