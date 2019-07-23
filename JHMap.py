import gensim
import json
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

class PhraseVector:
    def __init__(self, phrase):
        self.vector = self.PhraseToVec(phrase)
    # <summary> Calculates similarity between two sets of vectors based on the averages of the sets.</summary>
    # <param>name = "vectorSet" description = "An array of arrays that needs to be condensed into a single array (vector). In this class, used to convert word vecs to phrases."</param>
    # <param>name = "ignore" description = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. In this class, this would be stop words."</param>
    # <returns> The condensed single vector that has the same dimensionality as the other vectors within the vecotSet.</returns>
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
        if len(ignore) == 0: 
            return np.mean(vectorSet, axis = 0)
        else: 
            return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)

    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector=model1[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity=0
        except:
            cosine_similarity=0		
        return cosine_similarity

if __name__ == "__main__":
    # import pre-trained model
    google_model_path = '~/data/CHRISP/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(google_model_path, binary=True)

    # parse data dictionaries
    jh_file = '~/data/CHRISP/NUW_JH_variables.csv'
    ed_file = '.~/data/CHRISP/CHRISP-ED.csv'
    jh_list = pd.read_csv(jh_file)['Variable'].values.tolist()
    jh_replace = [("_", " "), ("DTTM", "Date Time"), (" ID", " Identifier"), ("DESC", "Description"), (" NO", " Number")]
    for idx, var in enumerate(jh_list):
        for (before, after) in jh_replace:
            jh_list[idx] = var.replace(before, after)

    livpool_path = []
    livpool_ed = pd.read_csv(ed_file)["Variable Name"].values.tolist()
    livpool_patient = []

    # check sentences for similarity
    match_dict = {}
    for sentence in jh_list:
        pv_1 = PhraseVector(sentence)
        phrase_score_list = []
        for match in livpool_ed:
            pv_2 = PhraseVector(match)
            score = pv_1.CosineSimilarity(pv_2.vector)
            phrase_score_list.append((pv_2, score))
        
        phrase_score_list.sort(key = lambda x:x[1], reverse = True)
        match_dict[sentence] = phrase_score_list

    json_dict = json.dumps(dict)
    with open('ed_match.csv', 'w+') as f:
        f.write(json_dict)