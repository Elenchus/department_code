import FileUtils
import pandas as pd
import re

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "descriptive_stats")
    rsp = FileUtils.spr_rsp_converter()
    filenames = FileUtils.get_mbs_files()
    frequencies_by_year = []
    labels_by_year = []
    years = []
    not_defined = rsp.convert_str("Not Defined")
    for filename in filenames:
        year = re.search("_(\d\d\d\d)\.", filename)[1]
        logger.log(f"Checking {year}")
        df = pd.read_parquet(filename, columns = ['NUMSERV', 'SPR_RSP'])
        rows_of_interest = df.loc[df['SPR_RSP'] == not_defined]
        unique_values = rows_of_interest['NUMSERV'].unique()
        frequencies = rows_of_interest['NUMSERV'].value_counts().tolist()

        frequencies_by_year.append(frequencies)
        labels_by_year.append(unique_values)
        years.append(year)

    FileUtils.categorical_plot_group(logger, labels_by_year, frequencies_by_year, years, "NUMSERV frequencies for Not Defined Provider Specialties", "numserv_not_defined")
    





