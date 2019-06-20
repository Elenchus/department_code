import FileUtils
from functools import partial
import gc
import itertools
from multiprocessing import Pool
import pandas as pd
import re
from FileUtils import mbs_header

def calculate_column_frequency(col, filename):
    year = re.search("_(\d\d\d\d)\.", filename)[1]

    logger.log(f"Loading parquet column {col} from {filename}")
    frequency = pd.read_parquet(filename, columns=[col]).value_counts().tolist()

    return (year, frequency)

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "descriptive_stats")
    filenames = FileUtils.get_mbs_files()

    for col in mbs_header:
        func = partial(calculate_column_frequency, col)
        p = Pool(processes = 3)
        data_map = p.imap(func, filenames)
        p.close()
        p.join()

        data_map.sort()
        years, frequencies = zip(*data_map)
        
        FileUtils.create_boxplot_group(logger, frequencies, years, f"Frequency distribution of {col} {years[0]} - {years[-1]}", f"frequency_{col}")
