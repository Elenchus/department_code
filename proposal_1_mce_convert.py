import FileUtils
import itertools
import pandas as pd
from datetime import datetime as dt

logger = FileUtils.logger(__name__, "proposal_1_mce_conversion", '/mnt/c/data')
output_file = logger.output_path / 'hip_replacement_mce_2003.txt'
data = pd.read_csv('hip_subset_2003.csv', usecols=['PIN', 'ITEM', 'DOS']).values.tolist()
pid, item, dos = zip(*data)
dos = [dt.strptime(x, "%Y-%m-%d").timestamp() for x in dos]

current_pid = pid[0] 
current_date = dos[0]
line = f"{current_pid}, [[{current_date}, [{str(item[0]).lower()}"
for i in range(1, len(pid)):
    if pid[i] != current_pid:
        line = line + ']]]\r\n' 
        with open(output_file, 'a') as f:
            f.write(line)

        current_date = dos[i]
        current_pid = pid[i]
        line = f"{current_pid}, [[{current_date}, [{str(item[i]).lower()}"
    elif dos[i] != current_date:
        line = line + f'], {dos[i]}, [{str(item[i]).lower()}'
        current_date = dos[i]
    else:
        line = line + f', {str(item[i]).lower()}'

line = line + ']]]'
with open(output_file, 'a') as f:
    f.write(line)