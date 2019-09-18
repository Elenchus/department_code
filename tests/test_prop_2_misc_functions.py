import unittest
import pandas as pd
import proposal_2
from tests.mock_logger import MockLogger

class MiscFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MockLogger()

    def tearDown(self):
        pass

    def test_item_labels(self):
        test_file = __import__(f"proposal_2.simple_provider-patient-item_clusters", fromlist=['TestCase'])
        mock_params = {'size': 9, 'INHOSPITAL': 'N', 'RSPs': ['Ophthalmology', 'Anaesthetics', 'Obstetrics and Gynaecology', 'Dermatology', 'Dentist (Approved) (OMS)']}
        test_case = test_file.TestCase(self.mock_logger, mock_params)
        mock_data = pd.DataFrame([[1, 7], [1, 5], [2, 7], [2, 6], [2, 7], [3, 9], [3, 9], [3, 9]], columns=["ITEM", "SPR_RSP"])
        test_case.processed_data = mock_data
        vocab = ["1", "2", "3"]
        (labels, frequencies) = test_case.get_item_labels(vocab)
        assert len(labels) == len(vocab)
        assert labels['1'] == "Mixed"
        assert labels['2'] == "Mixed"
        assert labels['3'] == "Neurology"
        assert frequencies[0] == 0.5
        assert frequencies[1] == 0.7
        assert frequencies[2] == 1

if __name__ == "__main__":
    unittest.main()