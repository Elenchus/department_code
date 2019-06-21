import FileUtils
from functools import partial
import gc
import itertools
from multiprocessing import Pool
import pandas as pd
import re
from FileUtils import mbs_header, pbs_header

def boxplot_all_header_items(logger, mbs_filenames, pbs_filenames):
    for (source, header, filenames) in [('MBS', mbs_header, mbs_filenames), ('PBS', pbs_header, pbs_filenames)]: 
            for col in header:
                years = []
                frequencies = []
                labels = []
                for filename in filenames:
                    year = re.search("_(\d\d\d\d)\.", filename)[1]
                    years.append(year)
                    logger.log(f"Loading {source} column {col} from {year}")
                    data = pd.read_parquet(filename, columns=[col])
                    unique_values = data[col].unique()
                    labels.append(unique_values)
                    no_unique_values = len(unique_values)
                    frequency = data[col].value_counts().tolist()
                    frequencies.append(frequency)
                
                if no_unique_values >= 15:
                    FileUtils.create_boxplot_group(logger, frequencies, years, f"Frequency distribution of {source} {col} {years[0]} - {years[-1]}", f"frequency_{col}")
                else:
                    FileUtils.categorical_plot_group(logger, labels, frequencies, years, f"Occurences of categories in {source} {col} {years[0]} - {years[-1]}", f"{source}_occurrence_{col}")

def correlate_stats(logger):
    pass

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "descriptive_stats")
    mbs_filenames = FileUtils.get_mbs_files()
    pbs_filenames = FileUtils.get_pbs_files()

    boxplot_all_header_items(logger, mbs_filenames, pbs_filenames)


    
