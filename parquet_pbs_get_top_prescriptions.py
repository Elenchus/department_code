import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats
from pyarrow import parquet as pq

path = 'H:/Data/PBS_Patient_10/'
filename = ''
filepath = path + filename
data = pq.read_pandas(filepath, columns=['ITM_CD', 'PRSCRPTN_CNT']).to_pandas()

codes = data.groupby('ITM_CD')
dict = {}
counts = []
for (name, group) in codes:
    num = sum(group['PRSCRPTN_CNT'])
    counts.append(num)
    dict[name] = num

counts_mean = stats.mean(counts)
counts_sd = stats.stdev(counts)

print(f'mean: {str(counts_mean)}')
print(f'SD: {str(counts_sd)}')

top_drugs_dict = {}

for key in dict:
    x = dict[key]
    if x >=counts_mean + (6 * counts_sd):
        top_drugs_dict[key] = x

number_of_codes = len(dict)
number_of_top = len(top_drugs_dict)

for num in [number_of_codes, number_of_top, 100*number_of_top/number_of_codes]:
    print(num)

plt.bar(range(len(top_drugs_dict)), list(top_drugs_dict.values()), align='center')
plt.xticks(range(len(top_drugs_dict)), list(top_drugs_dict.keys*()))
plt.show()

data.boxplot()