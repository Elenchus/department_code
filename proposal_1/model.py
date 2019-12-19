import math
import numpy as np
import pandas as pd
from enum import Enum
from gensim.models import KeyedVectors as w2v
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {"input_model": 'synthetic.vec', 'input_data': 'syntetic_proposal_1.csv', "codes_of_interest": ['220'], "project_name": "synthetic data test 1"} 
    # required_params = {"input_model": 'knee_21402_fasttext_cbow_2003_dim_10_epoch_60.vec', 'input_data': 'knee_21402_subset.csv', "codes_of_interest": ['21402'], "project_name": "fasttext cbow knee replacement from anaesthetic 21402"} 
    INITIAL_COLS = ["PIN", "ITEM"]
    FINAL_COLS = INITIAL_COLS
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        # raise AttributeError("Don't use this - use load data instead")
        return data

    def get_test_data(self):
        # raise AttributeError("Don't use this - use load data instead")
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        # input_model = f'prop_1_{self.required_params["code_type"]}_{self.required_params["year"]}_epoch_{self.required_params["epochs"]}_dim_{self.required_params["dimensions"]}_day.vec'
        # input_data = f'{self.required_params["code_type"]}_subset_{self.required_params["year"]}.csv'
        model = w2v.load_word2vec_format(self.required_params['input_model'], binary = False)

        self.log("Creating vectors for patients")
        patient_dict = {}
        with open(self.required_params['input_data'], 'r') as f:
            f.readline()
            for line in f:
                row = line.split(',')
                pid, item = row[1], row[2]
                keys = patient_dict.keys()
                if pid not in keys:
                    patient_dict[pid] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
                else:
                    patient_dict[pid]['Sum'] += model[item]
                    patient_dict[pid]['Average'] = ((patient_dict[pid]['Average'] * patient_dict[pid]['n']) + model[item]) / (patient_dict[pid]['n'] + 1)
                    patient_dict[pid]['n'] += 1

        patient_ids = list(patient_dict.keys())
        sums = [patient_dict[x]['Sum'] for x in patient_ids]
        avgs = [patient_dict[x]['Average'] for x in patient_ids]

        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            # perplex = math.ceil(math.sqrt(math.sqrt(len(model.wv.vocab))))
            # self.log("Creating TSNE plot")
            # FileUtils.tsne_plot(logger, matrix, perplex, f"t-SNE plot of hip replacement patients {name} with perplexity {perplex}")

            # self.log("Creating UMAP plot")
            # FileUtils.umap_plot(logger, matrix, f"UMAP plot of hip replacement patients {name}")

            Y = self.models.pca_2d(matrix)
            kmeans = self.models.k_means_cluster(Y, 16, f"{self.required_params['project_name']} {name} test.", f'{self.required_params["project_name"]}_kmeans_{name}')

            self.log("Calculating distances from k-means clusters")
            all_distances = kmeans.transform(Y) 
            assert len(all_distances) == len(kmeans.labels_)
            num_clusters = kmeans.n_clusters
            cluster_indices = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
            # for i in cluster_indices.keys():
            #     cluster_distances = pd.Series([all_distances[x][i] for x in cluster_indices[i]])

            #     q1 = cluster_distances.quantile(0.25)
            #     q3 = cluster_distances.quantile(0.75)
            #     iqr = q3 - q1

            #     outlier_count = 0
            #     cluster_outlier_file = self.logger.output_path / f"{name}_kmeans_outliers.txt"
            #     for idx, x in enumerate(cluster_distances):
            #         if x >= q3 + (1.5 * iqr):
            #             outlier_count = outlier_count + 1
            #             with open(cluster_outlier_file, 'a') as f:
            #                 pass
            #                 f.write(f'{patient_ids[cluster_indices[i][idx]]}: {x}, cluster: {i}\r\n') 

            # self.log(f"{outlier_count} outliers detected")
            self.models.calculate_BGMM(Y, num_clusters, f'{self.required_params["project_name"]} {name} patient BGMM', f'{name}_patient_BGMM')
            
            cluster_claim_counts = []
            cluster_item_counts = []
            cluster_items_for_interst = {}
            cluster_names = sorted(list(cluster_indices.keys()))
            for cluster_no in cluster_names:
                cluster_patient_indices = cluster_indices[cluster_no]
                cluster_patients = []
                for i in cluster_patient_indices:
                    cluster_patients.append(patient_ids[i])

                df = pd.read_csv(self.required_params['input_data'])
                cluster_items = {}
                cluster_unique_items = {}
                cluster_claims_per_patient = []
                cluster_number_unique_items = []
                for pin in cluster_patients:
                    patient = df.loc[df["PIN"] == int(pin)]
                    items = patient["ITEM"].values.tolist()
                    uniques = patient["ITEM"].unique().tolist()
                    cluster_claims_per_patient.append(len(items))
                    cluster_number_unique_items.append(len(uniques))
                    for item in items:
                        cluster_items[item] = cluster_items.get(item, 0) + 1

                    for unique in uniques:
                        cluster_unique_items[unique] = cluster_unique_items.get(unique, 0) + 1

                cluster_items_for_interst[cluster_no] = cluster_items
                cluster_claim_counts.append(cluster_claims_per_patient)
                cluster_item_counts.append(cluster_number_unique_items)
                cluster_items_items = list(cluster_items.items())
                cluster_unique_items_items = list(cluster_unique_items.items())
                cluster_unique_items_items.sort(key=lambda  x: x[1], reverse=True)
                cluster_items_items.sort(key=lambda  x: x[1], reverse=True)
                self.log(f"Most common items in cluster {cluster_no} for patient vector {name}")
                for i in range(10):
                    self.log(self.code_converter.convert_mbs_code_to_group_labels(cluster_items_items[i][0]))
                    self.log(cluster_items_items[i])

                self.log(f"Most common unique items in cluster {cluster_no} for patient vector {name}")
                for i in range(10):
                    self.log(self.code_converter.convert_mbs_code_to_group_labels(cluster_unique_items_items[i][0]))
                    self.log(cluster_unique_items_items[i])

            self.graphs.create_boxplot_group(cluster_claim_counts, 
                                                cluster_names, 
                                                f"Number of claims per patient per {name} cluster",
                                                f"cluster_{name}_claim_counts",
                                                ("Cluster", "Count"))
            self.graphs.create_boxplot_group(cluster_item_counts,
                                                cluster_names,
                                                f"Number of unique items per patient per {name} cluster",
                                                f"cluster_{name}_item_counts",
                                                ("Cluster", "Count"))

            interest_items_per_cluster = []
            for cluster in cluster_names:
                cluster_list = []
                for item in self.required_params["codes_of_interest"]:
                    cluster_list.append(cluster_items_for_interst[cluster].get(int(item), 0))

                interest_items_per_cluster.append(cluster_list)

            self.graphs.create_grouped_barchart(interest_items_per_cluster, 
                                                cluster_names,
                                                self.required_params["codes_of_interest"],
                                                f"Procedure types per cluster for vector {name}",
                                                f"{name}_procedure_types",
                                                ("Procedure code", "Count"))
            
            # self.log("Calculating unsupervised 1NN distance")
            # {i: Y[np.where(labels == i)] for i in range(kmeans.n_clusters)}
            # nn_calc = kNN(n_neighbors=1)
            # nn_calc.fit(Y) # 2d
            # distances, _ = nn_calc.kneighbors(n_neighbors=1)
            # distances = distances.tolist()
            # distances = [x[0] for x in distances]
            # distances = pd.Series(distances)
            # q1 = distances.quantile(0.25)
            # q3 =  distances.quantile(0.75)
            # iqr = q3 - q1
            # kNN_outlier_file = logger.output_path / f"{name}_1NN_outliers.txt"
            # outlier_count = 0
            # out_labels = [0] * len(Y)
            # with open(kNN_outlier_file, 'w+') as f:
            #     for idx, distance in enumerate(distances):
            #         if distance >= q3 + (1.5 * iqr):
            #             f.write(f'{model.index2word[idx]}: {distance}\r\n')
            #             outlier_count = outlier_count + 1
            #             out_labels[idx] = 1
            # self.log(f"{outlier_count} outliers detected")

            # self.log("Plotting 1NN outliers")
            # FileUtils.create_scatter_plot(logger, Y, out_labels, "1NN cluster and outliers", f"{name}_1NN")

            # self.log("Calculating 1NN cosine-similarity distances from word vector similarity")
            # nearest = {}
            # for word in model.vocab.keys():
            #     nearest[word] = model.most_similar(word)[0][1]

            # values = list(nearest.values())
            # distances = pd.Series(values)
            # q1 = distances.quantile(0.25)
            # q3 = distances.quantile(0.75)
            # iqr = q3 - q1

            # outliers = []
            # keys  = nearest.keys()
            # values, keys = (list(t) for t in zip(*sorted(zip(values, keys))))
            # cosine_outlier_file = logger.output_path / f"{name}_word2vec_cosine_similarity_outliers.txt"
            # outlier_count = 0
            # for i in range(len(keys)):
            #     if values[i] <= q1 - (0.5 * iqr):
            #         outlier_count = outlier_count + 1
            #         with open(cosine_outlier_file, 'a') as f:
            #             f.write(f'{keys[i]}: {values[i]}\r\n')

            # self.log(f"{outlier_count} outliers detected")
