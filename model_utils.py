import numpy as np

def get_outlier_indices(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    list_of_outlier_indices = []
    for i in range(len(data)):
        if data[i] > q75 + 1.5 * iqr:
            list_of_outlier_indices.append(i)

    return list_of_outlier_indices