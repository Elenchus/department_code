import FileUtils
import itertools
import pandas as pd
from datetime import datetime as dt

logger = FileUtils.logger(__name__, "proposal_1_mce_conversion", '/mnt/c/data')
output_file = logger.output_path / 'hip_replacement_mce.txt'
data = pd.read_csv('hip_subset.csv').values.tolist()
pid, item, dos = map(data)
dos = [dt.strptime(x, "%d%b%Y").timestamp() for x in dos]

current_pid = pid[0] 
current_date = dos[0]
line = f"{current_pid}, [[{current_pid}, [{item[0]}"
for i in range(1, len(pid)):
    if pid[i] != current_pid:
        line = line + ']]]\r\n' 
        with open(output_file, 'a') as f:
            f.write(line)

        current_date = dos[i]
        current_pid = pid[i]
        line = f"{current_pid}, [[{current_date}, [{item[i]}"

    if dos[i] != current_date:
        line = line + f'], {dos[i]}, [{item[i]}'
    else:
        line = line + f', {item[i]}'

with open(output_file, 'a') as f:
    f.write(line)