import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, TfidfModel

required_params = {"input_model": 'synthetic_proposal_1.vec', 'input_data': 'synthetic_proposal_1.csv'} 

data = pd.read_csv(required_params["input_data"])
patients = data.groupby("PIN")

documents = []
for name, group in patients:
    items = group["ITEM"].values.tolist()
    items = [str(item) for item in items]
    documents.append(items)


dictionary = Dictionary(documents)
dictionary.filter_extremes(no_above=0.1)
bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
lda_model = LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=2, workers=2)
lda_model.show_topics()