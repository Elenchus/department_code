import gensim
import json
import numpy as np
import pandas as pd
# from nltk.corpus import stopwords

# def average_sentence_vectore(words, model):
#             ignore_list = [' ', 'a', ',', '-', '.', '(', ')', "'"]
#     featureVec = np.zeros((len(model[word]),), dtype="float32")
#     nwords = 0
#     for word in sentence:
#         if word not in model.vocab or word in ignore_list:
#             continue

#         nwords = nwords + 1
#         featureVec = np.add(featureVec, model[word])

def simul_score(sentence, match_list, model): 
            phrase_score_list = [] 

            for match in match_list:
                current_score = model.wmdistance(sentence, match)
                phrase_score_list.append((match, current_score)) 
    

            return phrase_score_list

if __name__ == "__main__":
    # parse data dictionaries
    print("Parsing data dictionaries")
    jh_file = '~/data/CHRISP/NUW_JH_variables.csv'
    ed_file = '~/data/CHRISP/CHRISP-ED.csv'
    ad_file = '~/data/CHRISP/CHRISP_patient.csv'
    jh_list = pd.read_csv(jh_file)['Variable'].values.tolist()
    jh_replace = [("_", " "), ("DTTM", "Date Time"), (" ID", " Identifier"), ("DESC", "Description"), (" NO", " Number")]
    for idx in range(len(jh_list)):
        for (before, after) in jh_replace:
            x = jh_list[idx].replace(before, after)
            jh_list[idx] = x

    livpool_path = []
    livpool_ed = pd.read_csv(ad_file)["Variable Name"].values.tolist()
    livpool_ed = [x.replace('\r', '') for x in livpool_ed]

    # import pre-trained model
    print("Importing Google Word2Vec data")
    google_model_path = '~/data/CHRISP/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(google_model_path, binary=True)
    # check sentences for similarity
    print("Checking sentence similarity")
    match_dict = {}
    for sentence in jh_list:
        print(f"Checking  {sentence}")
        phrase_score_list = simul_score(sentence, livpool_ed, model)
        phrase_score_list.sort(key = lambda x:x[1], reverse = True)
        match_dict[sentence] = phrase_score_list

    print("Saving data")
    for key in match_dict.keys():
        match_dict[key].sort(key = lambda x:x[1])

    json_dict = json.dumps(match_dict)
    with open('ad_match.json', 'w+') as f:
        f.write(json_dict)