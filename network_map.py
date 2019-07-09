import FileUtils
import math
import pandas as pd
from gensim.models import Word2Vec

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "network_map", "/mnt/c/data")
    filenames = FileUtils.get_mbs_files()
    cols=['PIN', 'SPR']
    for filename in filenames:
        logger.log(f'Opening {filename}')
        data = pd.read_parquet(filename, columns=cols)

        unique_items = 0
        for col in cols:
            unique_items = unique_items + len(data[col].unique())

        logger.log("Converting to strings")
        for col in cols:
            data[col] = data[col].astype(str)
        
        words = data.values.tolist()

        logger.log("Creating W2V model")
        model = Word2Vec(
            words,
            size=math.sqrt(math.sqrt(unique_items)),
            window= len(cols),
            min_count=1,
            workers=3,
            iter=5)

        FileUtils.umap_plot(logger, model, 'UMAP plot of patient/provider/referrer model')

