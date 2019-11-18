from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as bp

if __name__ == '__main__':
  # get MBS data for 5 specialties (SPR_RSP):
  # 51: 'Anaesthetics'
  # 52: 'Dermatology'
  # 53: 'Obstetrics and Gynaecology'
  # 54: 'Ophthalmology'
  # 102: 'Dentist (Approved) (OMS)'

  relevant_spr_rsp = [51, 52, 53, 54, 102]
  print('Loading MBS')
  dtype = {'PIN': 'int64',
           'SPR': 'int64',
           'SPR_RSP': 'int64',
           'ITEM': 'int64'}
  # Reduce mbs table
  chunksize = 1000000
  i = 0
  df = pd.read_parquet('/home/elm/data/MBS_Patient_10/MBS_SAMPLE_10PCT_2014.parquet', columns=["PIN", "SPR", "SPR_RSP", "ITEM"])
  df = df[df['SPR_RSP'].isin(relevant_spr_rsp)]
    # save
    # if i == 0:
    #   df.to_csv('mbs_2014_reduced.csv', index=False)
    # else:
    #   df.to_csv('mbs_2014_reduced.csv', mode='a', header=False, index=False)
    # i += 1    
  
  # Load mbs table  
#   df = pd.read_csv('mbs_2014_reduced.csv', usecols=dtype.keys(), dtype=dtype)
#   df = df.drop_duplicates()
  
  print('Filter items')
  df = df.groupby('ITEM').filter(lambda x : len(x)>100)
  print('Filter patients')
  df = df.groupby('PIN').filter(lambda x : len(x)>2)
  
  df.to_pickle('filtered.pkl')
  
  # word2vec
  df = pd.read_pickle('filtered.pkl')
  
  df['ITEM'] = df['ITEM'].astype(str)
  color_df = df.groupby('ITEM')['SPR_RSP'].agg(pd.Series.mode)
  color_dict = color_df.to_dict()
  df_lists = df.groupby(['PIN'])['ITEM'].apply(list)
  
  data = df_lists.values.tolist()
    
  model = Word2Vec(min_count=0, size=10, window=100)
  model.build_vocab(data)

  model.train(data, total_examples=model.corpus_count, epochs=10)
  vectors = np.zeros((len(model.wv.vocab),10))
  color = np.zeros((len(model.wv.vocab)))

  i = 0
  for key in model.wv.vocab.keys():
    vectors[i, :] = model[key]
    if isinstance(color_dict[key], np.int64):
        color[i] = color_dict[key]
    else:
        color[i] = min(color_dict[key])
    if color[i] == 102:
      color[i] = 5
    else:
      color[i] = color[i]-50
    i += 1
  
  vectors_plot = TSNE(n_components=2).fit_transform(vectors)
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  scatter = ax.scatter(vectors_plot[:,0], vectors_plot[:,1], c=color)
  handles, _ = scatter.legend_elements(num=None)
  legend_names = ["Anaesthetics", "Dermatology", "Obstetrics and Gynaecology", "Opthalmology", 'Dentist (Approved) (OMS)']
  legend = ax.legend(handles, legend_names, loc="upper left", title="Legend", bbox_to_anchor=(1, 0.5))
  fig.savefig("out.png")
#   plt.show()
    



