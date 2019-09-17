from pathlib import Path

class MockLogger():
    output_path = Path('./Output/unit_tests')
    test_name = 'Mock'

    def log(self, line, line_end):
        pass
