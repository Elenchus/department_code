import pandas as pd
from phd_utils import file_utils

mbs_filename = file_utils.get_mbs_files(["2014"])[0]
data = pd.read_parquet(mbs_filename, columns=["PIN", "ITEM", "MDV_NUMSERV", "DOS"])