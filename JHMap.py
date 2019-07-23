import gensim
import json
import numpy as np
import pandas as pd
# from nltk.corpus import stopwords

def simul_score(sentence, match_list, model): 
            phrase_score_list = [] 
            ignore_list = [' ', 'a', ',', '-', '.', '(', ')']
            for match in match_list: 
                score = 0
                word_count = 0
                match_count = 0
                for word in sentence:
                    if word in ignore_list:
                        continue

                    match_score = 0
                    current_match = None
                    for match_word in match:
                        if match_word in ignore_list:
                            continue

                        try:
                            score = score + model.similarity(word, match_word) 
                            match_count = match_count + 1
                        except KeyError as e:
                           print(e) 

                phrase_score_list.append((match, score / match_count)) 
    
            phrase_score_list.sort(key = lambda x:x[1], reverse = True) 

            return phrase_score_list

if __name__ == "__main__":
    # parse data dictionaries
    print("Parsing data dictionaries")
    jh_file = '~/data/CHRISP/NUW_JH_variables.csv'
    ed_file = '~/data/CHRISP/CHRISP-ED.csv'
    jh_list = pd.read_csv(jh_file)['Variable'].values.tolist()
    jh_replace = [("_", " "), ("DTTM", "Date Time"), (" ID", " Identifier"), ("DESC", "Description"), (" NO", " Number")]
    for idx in range(len(jh_list)):
        for (before, after) in jh_replace:
            x = jh_list[idx].replace(before, after)
            jh_list[idx] = x

    livpool_path = []
    livpool_ed = pd.read_csv(ed_file)["Variable Name"].values.tolist()
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
    json_dict = json.dumps(match_dict)
    with open('ed_match.json', 'w+') as f:
        f.write(json_dict)