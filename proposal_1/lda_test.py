import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel

# required_params = {'input_data': 'synthetic_proposal_1.csv'} 
required_params = {'input_data': 'synthetic_sentences_prop_1.csv'} 

data = pd.read_csv(required_params["input_data"])
patients = data.groupby("PIN")

documents = []
for name, group in patients:
    items = group["ITEM"].values.tolist()
    items = [str(item) for item in items]
    documents.append(items)


dictionary = Dictionary(documents)
dictionary.filter_extremes()
bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
lda_model = LdaModel(bow_corpus, num_topics=3, id2word=dictionary, passes=2)
print(lda_model.show_topics())