import FileUtils
import numpy as np
import pandas as pd

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "network_map", "/mnt/c/data")
    filenames = FileUtils.get_mbs_files()[0]
    # filenames = FileUtils.get_mbs_files()
    cols=['PIN', 'SPR']
    for filename in filenames:
        logger.log(f'Opening {filename}')
        data = pd.read_parquet(filename, columns=cols)

        logger.log("One-hot encoding")
        unique_vals = []
        index = data.index.values.tolist()
        for col in cols:
            unique_vals = unique_vals + data[col].unique().values.tolist()

        one_hot_array = pd.DataFrame(0, index=np.arange(len(index)), columns=unique_vals)
        data = data.values.tolist()

        for idx, row in enumerate(data):
            for val in row:
                one_hot_array.at[index[idx], val] = 1