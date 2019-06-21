import FileUtils
import pandas
import timeit as tm

filename = FileUtils.get_mbs_files()[0]
col = 'PIN'
timer = tm.Timer("pandas.read_parquet(filename, columns=[col])", globals=globals())
x = timer.timeit(1)
print(x)