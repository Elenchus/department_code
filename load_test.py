import FileUtils
import pandas
import timeit as tm

filename = FileUtils.get_mbs_files()[0]
col = 'PIN'
x = tm.timeit(lambda: pandas.read_parquet(filename, columns=[col]))
print(x)