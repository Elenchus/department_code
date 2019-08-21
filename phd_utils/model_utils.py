import numpy as np
from phd_utils.code_converter import CodeConverter
from sklearn.decomposition import PCA

class ModelUtils():
    def __init__(self, logger):
        self.logger = logger

    # def calculate_cosine_similarity(self, model):
    #     self.logger.log("Calculating cosine similarities")
    #     cdv = CodeConverter()
    #     output_file = self.logger.output_path / "Most_similar.csv"
    #     with open(output_file, 'w+') as f:
    #         f.write("provider,Most similar to,Cosine similarity\r\n")
    #         for rsp in list(cdv.valid_rsp_num_values): 
    #             try: 
    #                 y = model.most_similar(str(rsp)) 
    #                 z = y[0][0] 
    #                 f.write(f"{cdv.convert_rsp_num(rsp),cdv.convert_rsp_num(z)},{round(y[0][1], 2)}\r\n") 
    #             except KeyError as err: 
    #                 continue
    #             except Exception:
    #                 raise

    def get_outlier_indices(self, data):
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        list_of_outlier_indices = []
        for i in range(len(data)):
            if data[i] > q75 + 1.5 * iqr:
                list_of_outlier_indices.append(i)

        return list_of_outlier_indices
        
    def pca_2d(self, data):
        self.logger.log("Performing PCA")
        pca2d = PCA(n_components=2)
        pca2d.fit(data)
        output = pca2d.transform(data)

        return output