import os
import unittest
import pandas as pd
from tests.mock_graph_utils import MockGraphUtils
from tests.mock_logger import MockLogger

class SimpleProviderPatientItemClusters(unittest.TestCase):
    def setUp(self):
        test_file = __import__(f"proposal_2.simple_provider-patient-item_clusters", fromlist=['TestCase'])
        mock_params = {'size': 9, 'INHOSPITAL': 'N', 'RSPs': ['Ophthalmology', 'Anaesthetics', 'Obstetrics and Gynaecology', 'Dermatology', 'Dentist (Approved) (OMS)']}
        mock_logger = MockLogger()
        self.test_case = test_file.TestCase(mock_logger, mock_params)
        self.test_case.graphs = MockGraphUtils()

    def tearDown(self):
        pass

    def test_get_test_data(self):
        mock_params = {'size': 9, 'INHOSPITAL': 'N', 'RSPs': ['Ophthalmology', 'Anaesthetics', 'Obstetrics and Gynaecology', 'Dermatology', 'Dentist (Approved) (OMS)']}
        self.test_case.required_params = mock_params
        mock_data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["PIN", "SPR", "ITEM"])
        self.test_case.processed_data = mock_data
        self.test_case.get_test_data()
        test_data = self.test_case.test_data
        assert len(test_data) == 0
        mock_data = pd.DataFrame([[1, 2, 3], [1, 5, 6], [1, 8, 9]], columns=["PIN", "SPR", "ITEM"])
        self.test_case.processed_data = mock_data
        self.test_case.get_test_data()
        test_data = self.test_case.test_data
        assert len(test_data) == 0
        mock_data = pd.DataFrame([[1, 2, 3], [1, 2, 6], [1, 2, 9]], columns=["PIN", "SPR", "ITEM"])
        self.test_case.processed_data = mock_data
        self.test_case.get_test_data()
        test_data = self.test_case.test_data
        assert len(test_data) == 1
        assert len(test_data[0]) == 3
        assert '3' in test_data[0]
        assert '9' in test_data[0]
        assert '6' in test_data[0]

    def test_run_test(self):
        sentences = []

if __name__ == "__main__":
    unittest.main()