import keras
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

    def sum_and_average_vectors(self, model, groups):
        item_dict = {}
        for key, group in groups:
            _, group = zip(*list(group))
            for item in group:
                item = str(item)
                if item not in model.wv.vocab:
                    continue
                    
                keys = item_dict.keys()
                if key not in keys:
                    item_dict[key] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
                else:
                    item_dict[key]['Sum'] += model[item]
                    item_dict[key]['Average'] = ((item_dict[key]['Average'] * item_dict[key]['n']) + model[item]) / (item_dict[key]['n'] + 1)
    
        sums = [item_dict[x]['Sum'] for x in item_dict.keys()]
        avgs = [item_dict[x]['Average'] for x in item_dict.keys()]

        return (sums, avgs)
