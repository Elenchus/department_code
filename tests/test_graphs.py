'''Test cases for GraphUtils'''
import unittest
import utilities.graph_utils as graph_utils
from mock_logger import MockLogger

class GraphUtilsTest(unittest.TestCase):
    '''Test cases for GraphUtils'''
    def setUp(self):
        self.graphs = graph_utils.GraphUtils(MockLogger())

    def tearDown(self):
        pass

    def test_graph_conversion(self):
        '''Confirm graph and adjacency matrices are correctly converted'''
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
        for ante in model:
            for con in model[ante].keys():
                assert graph[ante][con] is None

    def test_component_id(self):
        '''Confirm separate components are correctly found'''
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
        '''Confirm graph edit distance scoring is correct'''
        model = {
            1: {2: {}},
            3: {4: {}},
            4: {5: {}, 6: {}, 1: {}},
            5: {4: {}},
            6: {1: {}},
            7: {6: {}}
        }
        tests = [
            ({2: {}}, 0, 0),
            ({3: {}}, 0, 12),
            ({3: {7: {}}}, 1, 13),
            ({7: {3: {}}}, 1, 13),
            ({7: {}, 3: {}}, 0, 13),
            ({7: {3: {}}, 8: {7: {}}}, 3, 13),
            ({7: {3: {}}, 4: {7: {}}}, 2, 12),
            ({1: {}, 2: {}}, 0, 1)
        ]
        for test, plus_val, minus_val in tests:
            (plus_ged, minus_ged), _, _ = self.graphs.graph_edit_distance(model, test, edge_distance_costs=True)
            assert plus_ged == plus_val
            assert minus_ged == minus_val
