import math
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors as w2v
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {'code_type': 'knee', 'year': 2003, 'dimensions': 20, 'epochs': 100} 
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
        input_model = f'prop_1_{self.required_params["code_type"]}_{self.required_params["year"]}_epoch_{self.required_params["epochs"]}_dim_{self.required_params["dimensions"]}_day.vec'
        input_data = f'{self.required_params["code_type"]}_subset_{self.required_params["year"]}.csv'
        model = w2v.load_word2vec_format(input_model, binary = False)

        self.log("Creating vectors for patients")
        patient_dict = {}
        with open(input_data, 'r') as f:
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

        patient_ids = patient_dict.keys()
        sums = [patient_dict[x]['Sum'] for x in patient_ids]
        avgs = [patient_dict[x]['Average'] for x in patient_ids]

        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            # perplex = math.ceil(math.sqrt(math.sqrt(len(model.wv.vocab))))
            # self.log("Creating TSNE plot")
            # FileUtils.tsne_plot(logger, matrix, perplex, f"t-SNE plot of hip replacement patients {name} with perplexity {perplex}")

            # self.log("Creating UMAP plot")
            # FileUtils.umap_plot(logger, matrix, f"UMAP plot of hip replacement patients {name}")

            Y = self.models.pca_2d(matrix)
            kmeans = self.models.k_means_cluster(Y, 16, f'MCE {self.required_params["code_type"]} replacement k-meanspatients {name} test.', f'mce_{self.required_params["code_type"]}_kmeans_{name}')

            self.log("Calculating distances from k-means clusters")
            all_distances = kmeans.transform(Y) 
            assert len(all_distances) == len(kmeans.labels_)
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
            # self.models.calculate_BGMM(Y, 3, f'{name} patient BGMM', f'mce_{name}_patient_BGMM')
            
            # need to do some stuff
            # -> get clusters from k-means
            # -> for each cluster, calculate box plot for number of unique items, number of claims per patient
            # -> also most common items, etc

            cluster_claim_counts = []
            cluster_item_counts = []
            cluster_names = cluster_indices.keys()
            for cluster_no in cluster_names:
                cluster_patient_indices = cluster_indices[cluster_no]
                cluster_patients = patient_ids[cluster_patient_indices]
                df = input_data
                cluster_items = {}
                cluster_unique_items = {}
                cluster_claims_per_patient = []
                cluster_number_unique_items = []
                for pin in cluster_patients:
                    patient = df[df["PIN" == pin]]
                    items = patient["ITEM"].values.tolist()
                    uniques = patient["ITEM"].unique().values.tolist()
                    cluster_claims_per_patient.append(len(items))
                    cluster_number_unique_items.append(len(uniques))
                    for item in items:
                        cluster_items[item] = cluster_items.get(item, 0) + 1

                    for unique in uniques:
                        cluster_unique_items[unique] = cluster_unique_items.get(unique, 0) + 1

                cluster_claim_counts.append(cluster_claims_per_patient)
                cluster_item_counts.append(cluster_number_unique_items)
                cluster_items_items = cluster_items.items()
                cluster_unique_items_items = cluster_unique_items.items()
                cluster_unique_items_items.sort(key=lambda  x: x[1])
                cluster_items_items.sort(key=lambda  x: x[1])
                self.log(f"Most common items in cluster {cluster_no} for patient vector {name}")
                for i in range(10):
                    self.log(cluster_items_items[i])
                    self.log(self.code_converter.convert_mbs_code_to_group_labels(cluster_items_items[i]))

                self.log(f"Most common unique items in cluster {cluster_no} for patient vector {name}")
                for i in range(10):
                    self.log(cluster_unique_items_items[i])
                    self.log(self.code_converter.convert_mbs_code_to_group_labels(cluster_unique_items_items[i]))

            self.graphs.create_boxplot_group(cluster_claim_counts, 
                                                cluster_names, 
                                                "Number of claims per patient per cluster", "cluster_claim_counts.png")
            self.graphs.create_boxplot_group(cluster_item_counts,
                                                cluster_names,
                                                "Number of unique items per patient per cluster",
                                                "cluster_item_counts.png")

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
