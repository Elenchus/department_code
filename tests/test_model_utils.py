'''Unit tests for ModelUtils'''
import unittest
from tests.mock_graph_utils import MockGraphUtils
from tests.mock_logger import MockLogger
import utilities.model_utils as model_utils
from utilities.code_converter import CodeConverter

def create_mba_test_data():
    '''create test data'''
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
    '''Test cases for ModelUtils'''
    def setUp(self):
        self.model = model_utils.ModelUtils(MockLogger(), MockGraphUtils(), CodeConverter(2014))

    def tearDown(self):
        pass

    def test_fpgrowth(self):
        '''confirm fpgrowth implementation correctly finds association rules'''
        test_function = self.model.fp_growth_analysis
        documents = create_mba_test_data()

        # test min support
        d = test_function(documents, min_support=0.01, min_conviction=0)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min conviction
        d = test_function(documents, min_support=0.01, min_conviction=1.1)
        cat = [1000, 999, 900, 899, 700, 10, 9, 1]
        for i, a in enumerate(cat):
            for j, b in enumerate(cat):
                if a == b:
                    continue

                supp = min(a, b) / max(cat)
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
        '''confirm apriori implementation correctly finds association rules'''
        test_function = self.model.apriori_analysis
        documents = create_mba_test_data()
        # test min_support
        d = test_function(documents, min_support=0.01, min_confidence=0, min_lift=0)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min_confidence
        d = test_function(documents, min_support=0.01, min_confidence=0.9, min_lift=0)
        assert d['0']['1'] == {'color': '#0200fd'}
        assert d['0']['2'] == {'color': '#0200fd'}
        assert len(d['0']) == 2

        # test min_lift
        d = test_function(documents, min_support=0.01, min_confidence=0, min_lift=1.1)
        cat = [1000, 999, 900, 899, 700, 10, 9, 1]

        for i, a in enumerate(cat):
            for j, b in enumerate(cat):
                if a == b:
                    continue
                lift = (min(a, b)/max(cat))/((a/max(cat))*(b/max(cat)))
                supp = min(a, b) / max(cat)
                if supp >= 0.01:
                    if lift < 1.1:
                        if str(a) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

    def test_pairwise_market_basket(self):
        '''confirm pairwise implementation correctly finds association rules'''
        test_function = self.model.mba.pairwise_market_basket
        documents = create_mba_test_data()
        cat = [1000, 999, 900, 899, 700, 10, 9, 1]
        names = [str(x) for x in range(8)]
        # test min_support
        d = test_function(names, documents, min_support=0.01)
        assert len(d) == 6
        for i in d:
            assert len(d[i]) == 5

        # test min_confidence
        filters = {'confidence': {'value': 0.9}}
        self.model.mba.update_filters(filters)
        d = test_function(names, documents, min_support=0.01)
        assert '1' in d['0'] # {'color': '#0200fd'}
        assert '2' in d['0'] # {'color': '#0200fd'}
        assert len(d['0']) == 2

        # test min_lift
        filters = {
            'confidence': {'value': 0.0},
            'lift': {'value': 1.1}
        }
        self.model.mba.update_filters(filters)
        d = test_function(names, documents, min_support=0.01)

        for i, a in enumerate(cat):
            for j, b in enumerate(cat):
                if a == b:
                    continue
                lift = (min(a, b)/max(cat))/((a/max(cat))*(b/max(cat)))
                supp = min(a, b) / max(cat)
                if supp >= 0.01:
                    if lift < 1.1:
                        if str(a) in d:
                            assert str(j) not in d[str(i)]
                    else:
                        assert str(j) in d[str(i)]

        # test min conviction
        filters = {
            'conviction': {'value': 1.1},
            'lift': {'value': 0}
        }
        self.model.mba.update_filters(filters)
        d = test_function(names, documents, min_support=0.01)
        for i, a in enumerate(cat):
            for j, b in enumerate(cat):
                if a == b:
                    continue

                supp = min(a, b) / max(cat)
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
        filters = {
            'conviction': {'value': 0},
            'odds_ratio': {'value': 9999}
        }
        self.model.mba.update_filters(filters)
        d = test_function(names, documents, min_support=0)
        assert len(d) == 8
        for v in d.values():
            assert len(v) == 7

if __name__ == "__main__":
    unittest.main()
