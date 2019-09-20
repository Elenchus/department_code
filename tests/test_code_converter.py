import unittest
from phd_utils.code_converter import CodeConverter

class TestCodeConverter(unittest.TestCase):
    def setUp(self):
        self.cdv = CodeConverter()

    def tearDown(self):
        pass

    def test_convert_mbs_code(self):
        assert self.cdv.convert_mbs_code(113) == "1 - A3 - None"
        assert self.cdv.convert_mbs_code("32046") == "3 - T8 - 2"

    def test_convert_pbs_code(self):
        row = self.cdv.convert_pbs_code("00043G")
        assert len(row) == 4
        assert row[0] == "00043G"
        assert row[1] == "EXTEMPORANEOUSLY PREPARED"
        assert row[2] == "ointments, waxes"
        assert row[3] == "Z"

    def test_convert_rsp_num(self):
        expected = {0: "Not Defined", 4: "Cardiology", 160: "Interns", 416: "College Trainee - Palliative Medicine"}
        for key, val in expected.items():
            result = self.cdv.convert_rsp_num(key)
            assert result == val

    def test_convert_rsp_str(self):
        expected = {0: "Not Defined", 4: "Cardiology", 160: "Interns", 416: "College Trainee - Palliative Medicine"}
        for key, val in expected.items():
            result = self.cdv.convert_rsp_str(val)
            assert result == key
