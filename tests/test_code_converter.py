import unittest
from utilities.code_converter import CodeConverter

class TestCodeConverter(unittest.TestCase):
    def setUp(self):
        self.cdv = CodeConverter(2019)

    def tearDown(self):
        pass

    def test_convert_mbs_code(self):
        for (func, test, expected_result) in [
                                        (self.cdv.convert_mbs_code_to_group_numbers, 113, ['1', 'A3', None]),
                                        (self.cdv.convert_mbs_code_to_group_numbers, "32046", ['3', 'T8', '2']),
                                        (self.cdv.convert_mbs_code_to_description, 50608, "SCOLIOSIS OR KYPHOSIS, in a child or adolescent, treatment by segmental instrumentation and fusion of the spine, not being a service to which item51011 to 51171 applies (Anaes.) (Assist.)\n"),
                                        (self.cdv.convert_mbs_code_to_description, "11004", "Electroencephalography, ambulatory or video, prolonged recording of at least 3 hours duration up to 24 hours duration, recording on the first day, not being a service: (a) associated with a service to which item 11000,11003, 11005, 11006 or 11009 applies; or (b) involving quantitative topographic mapping using neurometrics or similar devices\n"),
                                        (self.cdv.convert_mbs_code_to_group_labels, 24130, ["THERAPEUTIC PROCEDURES", "Relative Value Guide For Anaesthesia - Medicare Benefits Are Only Payable For Anaesthesia Performed In Association With An Eligible Service", 'Anaesthesia/Perfusion Time Units']),
                                        (self.cdv.convert_mbs_code_to_group_labels, "47", ["PROFESSIONAL ATTENDANCES", "General Practitioner Attendances To Which No Other Item Applies"])
                                        ]:
            actual_result = func(test)
            if isinstance(expected_result, list):
                assert len(expected_result) == len(actual_result)
                for i in range(len(expected_result)):
                    assert expected_result[i] == actual_result[i]
            else:
                assert expected_result == actual_result

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
