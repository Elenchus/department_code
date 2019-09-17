import math
import unittest
import phd_utils.model_utils as model_utils
from mock_logger import MockLogger

class ModelUtilsTest(unittest.TestCase):
    def __init__(self):
        self.model = model_utils.ModelUtils(MockLogger())

    def setUp(self):
        pass

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

if __name__ == "__main__":
    unittest.main()