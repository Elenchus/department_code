import os
import sys
import logging
import pandas as pd

from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import cluster, metrics
from sklearn.manifold import TSNE

if sys.platform == "win32":
    path = 'C:/Data/'
else:
    path = '/home/elm/data'

mbs_header = ['PIN', 'DOS', 'PINSTATE', 'SPR', 'SPR_RSP', 'SPRPRAC', 'SPRSTATE', 'RPR', 'RPRPRAC', 'RPRSTATE', 'ITEM', 'NUMSERV', 'MDV_NUMSERV', 'BENPAID', 'FEECHARGED', 'SCHEDFEE', 'BILLTYPECD', 'INHOSPITAL', 'SAMPLEWEIGHT']
pbs_header = ['PTNT_ID', 'SPPLY_DT', 'ITM_CD', 'PBS_RGLTN24_ADJST_QTY', 'BNFT_AMT', 'PTNT_CNTRBTN_AMT', 'SRT_RPT_IND', 'RGLTN24_IND', 'DRG_TYP_CD', 'MJR_SPCLTY_GRP_CD', 'UNDR_CPRSCRPTN_TYP_CD', 'PRSCRPTN_CNT', 'PTNT_CTGRY_DRVD_CD', 'PTNT_STATE']

def categorical_plot_group(logger, x, y, legend_labels, title, filename):
    logger.log(f"Plotting bar chart: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(y)):
        ax.plot(x[i], y[i], label=legend_labels[i])
    
    ax.legend()
    fig.suptitle(title)
    fig.savefig(logger.output_path + filename + datetime.now().strftime("%Y%m%dT%H%M%S"))
    plt.close(fig)

def create_boxplot(logger, data, title, filename):
    logger.log(f"Plotting boxplot: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    ax.suptitle(title)
    fig.savefig(logger.output_path + filename + datetime.now().strftime("%Y%m%dT%H%M%S"))
    plt.close(fig)

def create_boxplot_group(logger, data, labels, title, filename):
    logger.log(f"Plotting boxplot group: {title}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    fig.suptitle(title)
    ax.set_xticklabels(labels)
    fig.savefig(logger.output_path + filename + datetime.now().strftime("%Y%m%dT%H%M%S"))
    plt.close(fig)

def get_best_cluster_size(logger, X, clusters):
    logger.log("Getting best k-means cluster sizer with average silhouette score")
    avg_sil = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        avg_sil.append(silhouette_score)
        logger.log(f"n = {n}, silhouette score = {silhouette_score}")

    k = clusters[avg_sil.index(max(avg_sil))]
    logger.log(f"Max silhouette score with {k} clusters")

    return k

def get_mbs_files():
    mbs_path = path + 'MBS_Patient_10/'
    files = [mbs_path + f for f in os.listdir(mbs_path) if f.lower().endswith('.parquet')]

    return files

def get_pbs_files():
    pbs_path = path + 'PBS_Patient_10/'
    files = [pbs_path + f for f in os.listdir(pbs_path) if f.lower().endswith('.parquet')]

    return files

def tsne_plot(logger, model, perplex):
    logger.info(f"Creating TSNE model with perplexity {perplex}")
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    logger.info(f"Plotting TSNE figure")
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
    
    path = logger.output_path + "t-SNE_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    logger.info(f"Saving TSNE figure to {path}")
    fig.savefig(path)

class spr_rsp_converter:
    def __init__(self):
        filename = 'SPR_RSP.csv'
        if not os.path.isfile(filename):
            raise OSError("Cannot find SPR_RSP.csv - please put it in the same folder as FileUtils")

        self.table = pd.read_csv(filename)
        self.valid_values = self.table['SPR_RSP'].unique()

    def convert(self, rsp):
        if int(rsp) not in self.valid_values:
            raise ValueError(f"{rsp} is not a valid SPR_RSP")

        return self.table.loc[self.table['SPR_RSP'] == int(rsp)]['Label'].values.tolist()[0]
    

class logger:
    def __init__(self, name, test_name):
        self.output_path = self.create_output_folder(test_name) + '\\'
        self.logger = logging.getLogger(name)
        # handler = logging.StreamHandler(stream=sys.stdout)
        # self.logger.addHandler(handler)
        sys.excepthook = self.handle_exception
        self.file_name = self.output_path + test_name + '.log'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename = self.file_name, filemode = 'w+')
        self.log(f"Starting {test_name}")

    def create_output_folder(self, test_name):
        path = os.getcwd() + '\\Output\\' + test_name + '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
        os.makedirs(path)

        return path

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def log(self, line, line_end = '...'):
        print(f"{datetime.now()} {line}{line_end}")
        self.logger.info(line)