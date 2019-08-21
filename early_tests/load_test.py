from phd_utils import file_utils, graph_utils
import pandas
import timeit as tm

filename = file_utils.get_mbs_files()[0]
timer = tm.Timer("pandas.read_parquet(filename, columns=['PIN', 'ITEM', 'INHOSPITAL'])", globals=globals())
x = timer.timeit(1)
print(x)