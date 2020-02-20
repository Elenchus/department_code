import unittest
import phd_utils.graph_utils as graph_utils
from tests.mock_logger import MockLogger

class ModelUtilsTest(unittest.TestCase):
    def setUp(self):
        self.graphs = graph_utils.GraphUtils(MockLogger())

    def tearDown(self):
        pass

    def test_graph_edit_distance(self):
        model = {
            1: {2: {}},
            3: {4: {}},
            4: {5: {}, 6: {}, 1: {}},
            5: {4: {}},
            6: {1: {}}
        }
        tests = [
            ({2: {}}, 0),
            ({3: {}}, 12),
            ({3: {7: {}}}, 14),
            ({7: {3: {}}}, 14),
            ({7: {}, 3: {}}, 13),
            ({7: {3: {}}, 8: {7: {}}}, 16),
            ({7: {3: {}}, 4: {7: {}}}, 14)
        ]
        for test, val in tests:
            ged = self.graphs.graph_edit_distance(model, test)
            assert ged == val
