import unittest
import phd_utils.graph_utils as graph_utils
from tests.mock_logger import MockLogger

class ModelUtilsTest(unittest.TestCase):
    def setUp(self):
        self.graphs = graph_utils.GraphUtils(MockLogger())

    def tearDown(self):
        pass

    def test_graph_conversion(self):
        model = {
            1: {2: {}},
            3: {4: {}},
            4: {5: {}, 6: {}, 1: {}},
            5: {4: {}},
            6: {1: {}},
            7: {6: {}}
        }

        am = self.graphs.convert_graph_to_adjacency_matrix(model)
        assert am.at[1, 2] == 1
        assert am.loc[1].sum() == 1
        assert am[1].sum() == 2

        graph = self.graphs.convert_adjacency_matrix_to_graph(am)
        assert len(model.keys()) == len(graph.keys())
        for ante in model.keys():
            for con in model[ante].keys():
                assert graph[ante][con] is None

    def test_component_id(self):
        model = {
            1: {2: {}},
            3: {4: {}},
            4: {5: {}},
            5: {3: {}}
        }

        components = self.graphs.graph_component_finder(model)
        assert len(components) == 2
        assert len(components[0]) == 2 or len(components[0]) == 3
        assert len(components[1]) == 3 or len(components[1]) == 2
        assert len(components[0]) != len(components[1])

    def test_graph_edit_distance(self):
        model = {
            1: {2: {}},
            3: {4: {}},
            4: {5: {}, 6: {}, 1: {}},
            5: {4: {}},
            6: {1: {}},
            7: {6: {}}
        }
        tests = [
            ({2: {}}, 0),
            ({3: {}}, 12),
            ({3: {7: {}}}, 14),
            ({7: {3: {}}}, 14),
            ({7: {}, 3: {}}, 13),
            ({7: {3: {}}, 8: {7: {}}}, 16),
            ({7: {3: {}}, 4: {7: {}}}, 14),
            ({1: {}, 2: {}}, 1)
        ]
        for test, val in tests:
            ged, _, _ = self.graphs.graph_edit_distance(model, test)
            assert ged == val
