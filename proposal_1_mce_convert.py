import FileUtils
import itertools
import pandas as pd
from datetime import datetime as dt

code_type = 'knee'
logger = FileUtils.logger(__name__, f"proposal_1_mce_conversion_{code_type}", '/mnt/c/data')
output_file = logger.output_path / f'{code_type}_replacement_mce_2003.txt'
data = pd.read_csv(f'{code_type}_subset_2003.csv', usecols=['PIN', 'ITEM', 'DOS']).values.tolist()
pid, item, dos = zip(*data)
dos = [dt.strptime(x, "%Y-%m-%d").timestamp() for x in dos]

current_pid = pid[0] 
current_date = dos[0]
line = f"{current_pid}, [[{current_date}, [{item[0]}"
for i in range(1, len(pid)):
    if pid[i] != current_pid:
        line = line + ']]]\r\n' 
        with open(output_file, 'a') as f:
            f.write(line)

        current_date = dos[i]
        current_pid = pid[i]
        line = f"{current_pid}, [[{current_date}, [{item[i]}"
    elif dos[i] != current_date:
        line = line + f'], {dos[i]}, [{item[i]}'
        current_date = dos[i]
    else:
        line = line + f', {item[i]}'

line = line + ']]]'
with open(output_file, 'a') as f:
    f.write(line)