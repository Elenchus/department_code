import FileUtils
import pandas as pd
from timeit import timeit

filename = FileUtils.get_mbs_files()[0]
col = 'PIN'
x = timeit(pd.read_parquet(filename, columns=[col]))
print(x)