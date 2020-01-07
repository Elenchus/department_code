import numpy as np
import pandas as pd
from biterm.cbtm import oBTM
from biterm.utility import vec_to_biterms
from sklearn.feature_extraction.text import CountVectorizer

input_data = 'synthetic_proposal_1.csv'

data = pd.read_csv(input_data)
patients = data.groupby("PIN")

documents = []
for name, group in patients:
    items = group["ITEM"].values.tolist()
    items = ' '.join([str(item) for item in items])
    documents.append(items)

vec = CountVectorizer()
X = vec.fit_transform(documents)

vocab = np.array(vec.get_feature_names())
biterms = vec_to_biterms(X)
btm =oBTM(num_topics=3, V=vocab)
topics = btm.fit_transform(biterms, iterations=100)