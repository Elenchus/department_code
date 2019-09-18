from pathlib import Path

class MockLogger():
    output_path = Path('./Output/')
    test_name = 'Mock'

    def log(self, line, line_end=None):
        pass
