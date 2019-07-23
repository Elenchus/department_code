import gensim

# import pre-trained model
google_model_path = ''
model = gensim.models.Word2Vec.load_word2vec_format(google_model_path, binary=True)

# parse data dictionaries
jh_list = []
livpool_path = []
livpool_icu = []
livpool_general = []

# check sentences for similarity
