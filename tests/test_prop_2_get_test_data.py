import unittest
import pandas as pd
from mock_logger import MockLogger
from phd_utils import base_proposal_test

class GetTestDataTest(unittest.TestCase):
    def __init__(self):
        self.mock_logger = MockLogger()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_provider_patient_item_clusters(self):
        test_file = __import__(f"proposal_2.simple_provider-patient-item_clusters.py", fromlist=['TestCase'])
        mock_params = {'size': 9, 'INHOSPITAL': 'N', 'RSPs': ['Ophthalmology', 'Anaesthetics', 'Obstetrics and Gynaecology', 'Dermatology', 'Dentist (Approved) (OMS)']}
        test_case = test_file.TestCase(self.mock_logger, mock_params)
        mock_data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_case.processed_data = mock_data
        test_case.get_test_data()
        test_data = test_case.test_data
        assert len(test_data) == 0
        mock_data = pd.DataFrame([[1, 2, 3], [1, 5, 6], [1, 8, 9]])
        test_case.processed_data = mock_data
        test_case.get_test_data()
        test_data = test_case.test_data
        assert len(test_data) == 0
        mock_data = pd.DataFrame([[1, 2, 3], [1, 2, 6], [1, 2, 9]])
        test_case.processed_data = mock_data
        test_case.get_test_data()
        test_data = test_case.test_data
        assert len(test_data) == 1
        assert test_data[0][0] == 3
        assert test_data[0][1] == 6
        assert test_data[0][2] == 9

if __name__ == "__main__":
    unittest.main()