import FileUtils
import itertools
import pandas as pd
from sklearn.decomposition import PCA

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, f"proposal_2_one_hot_rsp_item_cluster", "/mnt/c/data")
    filenames = FileUtils.get_mbs_files()

    for filename in filenames:
        logger.log(f'Opening {filename}')
        cols = ["SPR_RSP", "ITEM"]
        data = pd.read_parquet(filename, columns=cols)
        columns = data["ITEM"].unique().tolist()
        rows = data["SPR_RSP"].unique().tolist()
        for i in range(len(cols)):
            assert data.columns[i] == cols[i]

        logger.log("Grouping values")
        groups = sorted(data.values.tolist())
        groups = itertools.groupby(groups, key=lambda x: x[0]) 

        logger.log("One-hot encoding")
        one_hot_table = pd.DataFrame(0, index=rows, columns=columns)
        for rsp, group in groups:
            items = set(x[1] for x in list(group))
            for col in items:
                one_hot_table.loc[rsp, col] = 1

        logger.log("Performing PCA")
        pca2d = PCA(n_components=2)
        pca2d.fit(one_hot_table)
        Y = pca2d.transform(one_hot_table)

        FileUtils.create_scatter_plot(logger, Y, rows, f"RSP clusters", f'RSP_clusters')

        break