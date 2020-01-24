import itertools
import math
import unittest
import numpy as np
import pandas as pd
import phd_utils.model_utils as model_utils
from tests.mock_logger import MockLogger
from tests.mock_w2v import MockW2V

def create_mba_test_data():
    documents = []
    for i in range(1000):
        doc = []
        doc.append('0')
        if i < 999:
            doc.append('1')
        if i < 900:
            doc.append('2')
        if i < 899:
            doc.append('3')
        if i < 700:
            doc.append('4')
        if i < 10:
            doc.append('5')
        if i < 9:
            doc.append('6')
        if i < 1:
            doc.append('7')

        documents.append(doc)

    return documents

class ModelUtilsTest(unittest.TestCase):
    def setUp(self):
        self.model = model_utils.ModelUtils(MockLogger())

    def tearDown(self):
        pass

    def test_cartesian_polar(self):
        data = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        polar = self.model.cartesian_to_polar(data)
        for result in polar:
            assert result[0] == math.sqrt(2)

        assert polar[0][1] == 45
        assert polar[1][1] == 135
        assert polar[2][1] == 225
        assert polar[3][1] == 315

    def test_get_best_cluster_size(self):
        arr = None
        no_clusters = 6
        for n in range(no_clusters):
            part_arr = (n * 2) + np.random.rand(10, 2)
            if arr is None:
                arr = part_arr
            else:
                arr = np.concatenate((arr, part_arr))

        (k, max_n) = self.model.get_best_cluster_size(arr, [2, 4, 6, 8])

        assert k == 6
        assert max_n > 0 and max_n <= 100

    def test_get_outlier_indices(self):
        data = [50] * 50 + [40] * 25 + [60] * 25 + [9] + [91]
        x = self.model.get_outlier_indices(data)
        assert len(x) == 2
        assert 101 in x
        assert 100 in x

    def test_fpgrowth(self):
        test_function = self.model.fp_growth_analysis
        documents = create_mba_test_data()

        # test min support
        d = test_function(documents, min_support=0.01, min_conviction=0)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min conviction
        d = test_function(documents, min_support=0.01, min_conviction=1.1)
        cat = [1000,999,900,899,700,10,9,1]
        for i in range(len(cat)):
            for j in range(len(cat)):
                a = cat[i]
                b = cat[j]
                if a == b:
                    continue
                
                supp = min(a,b) / max(cat)
                supp_x = a / max(cat)
                supp_y = b / max(cat)
                conf = supp / supp_x
                conviction = (1 - supp_y) / (1 - conf) if conf != 1 else 9999
                if supp >= 0.01:
                    if conviction < 1.1:
                        if str(i) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

    def test_apriori(self):
        test_function = self.model.apriori_analysis
        documents = create_mba_test_data()
        # test min_support
        d = test_function(documents, min_support=0.01, min_confidence= 0, min_lift=0)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min_confidence
        d = test_function(documents, min_support=0.01, min_confidence=0.9, min_lift=0)
        assert d['0']['1'] == {'color': '#0200fd'}
        assert d['0']['2'] == {'color': '#0200fd'}
        assert len(d['0']) == 2 

        # test min_lift
        d = test_function(documents, min_support=0.01, min_confidence= 0, min_lift=1.1)
        cat = [1000,999,900,899,700,10,9,1]

        for i in range(len(cat)):
            for j in range(len(cat)):
                a = cat[i]
                b = cat[j]
                if a == b:
                    continue
                lift = (min(a,b)/max(cat))/((a/max(cat))*(b/max(cat)))
                supp = min(a,b) / max(cat)
                if supp >= 0.01:
                    if lift < 1.1:
                        if str(a) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

    def test_pairwise_market_basket(self):
        test_function = self.model.pairwise_market_basket
        documents = create_mba_test_data()
        cat = [1000,999,900,899,700,10,9,1]
        names = [str(x) for x in range(8)]
        # test min_support
        d = test_function(names, documents, min_support=0.01, min_confidence= 0, min_lift=0, min_conviction=0)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min_confidence
        d = test_function(names, documents, min_support=0.01, min_confidence=0.9, min_lift=0, min_conviction=0)
        assert d['0']['1'] == None # {'color': '#0200fd'}
        assert d['0']['2'] == None # {'color': '#0200fd'}
        assert len(d['0']) == 2 

        # test min_lift
        d = test_function(names, documents, min_support=0.01, min_confidence= 0, min_lift=1.1, min_conviction=0)

        for i in range(len(cat)):
            for j in range(len(cat)):
                a = cat[i]
                b = cat[j]
                if a == b:
                    continue
                lift = (min(a,b)/max(cat))/((a/max(cat))*(b/max(cat)))
                supp = min(a,b) / max(cat)
                if supp >= 0.01:
                    if lift < 1.1:
                        if str(a) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

        # test min conviction
        d = test_function(names, documents, min_support=0.01, min_conviction=1.1, min_confidence=0, min_lift=0)
        for i in range(len(cat)):
            for j in range(len(cat)):
                a = cat[i]
                b = cat[j]
                if a == b:
                    continue
                
                supp = min(a,b) / max(cat)
                supp_x = a / max(cat)
                supp_y = b / max(cat)
                conf = supp / supp_x
                conviction = (1 - supp_y) / (1 - conf) if conf != 1 else 9999
                if supp >= 0.01:
                    if conviction < 1.1:
                        if str(i) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

        # test min odds ratio - I believe all should go to infinity, denominator will always be 0. not a great test...
        d = test_function(names, documents, min_support=0, min_confidence=0, min_conviction=0, min_lift=0, min_odds_ratio=9999)
        assert len(d) == 8
        for v in d.values():
            assert len(v) == 7

    def test_sum_and_average_vecors(self):
        vocab = {}
        for n in range(1, 5):
            vocab[str(n)] = np.array([10**n, 2 * 10**n])

        model = MockW2V(vocab)
        data = [["A", 1], ["A", 2], ["B", 3], ["B", 4]] # 2d test, should maybe see what happens if I have too many columns
        groups = itertools.groupby(data, key = lambda x: x[0])
        (sums, avgs) = self.model.sum_and_average_vectors(model, groups)

        assert len(sums) == 2
        assert len(avgs) == 2
        assert sums[0][0] == 110
        assert sums[0][1] == 220
        assert sums[1][0] == 11000
        assert sums[1][1] == 22000
        assert avgs[0][0] == 110 / 2
        assert avgs[0][1] == 220 / 2
        assert avgs[1][0] == 11000 / 2
        assert avgs[1][1] == 22000 / 2
        
if __name__ == "__main__":
    unittest.main()