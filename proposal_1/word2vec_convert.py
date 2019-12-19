from datetime import datetime as dt
from gensim.models import Word2Vec
import pandas as pd

code_type = 'knee'
# input_file = f'knee_21402_subset.csv'
# output_file = f'{code_type}_21402_w2v_2003.txt'
input_file = "simple_syntetic_proposal_1.csv"
output_file = 'simple_synthetic.vec'

data = pd.read_csv(input_file, usecols=['PIN', 'ITEM', 'DOS'])
groups = data.groupby("PIN")
word_list = []
for name, group in groups:
    group = group.sort_values("DOS")
    word_list.append([str(x) for x in group["ITEM"].values.tolist()])

model = Word2Vec(word_list, min_count = 1, size = 10, iter = 60)
model.wv.save_word2vec_format(output_file, binary = False)
