# may want to use glob for file collection -> can handle regex
import os
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

path = '//ad.unsw.edu.au/OneUNSW/MED/CBDRH/PBS MBS/PBS data/'
new_path = 'H:/Data/'
for filename in [f for f in os.listdir(path) if f.lower().endswith('.csv')]:
    filepath = path + filename
    new_name = new_path + filename[:-3] + 'parquet'
    if os.path.isfile(new_name):
        continue
        
    print(f'Opening {filepath}')
    
    data_list = []
    for chunk in pd.read_csv(filepath, low_memory = False, chunksize = 10000):
        data_list.append(chunk)
    
    csv_data = pd.concat(data_list)

    parquet_data = pa.Table.from_pandas(csv_data)
    print('Writing new file')
    pq.write_table(parquet_data, new_name)
    print('Done')

    del csv_data
    gc.collect()

