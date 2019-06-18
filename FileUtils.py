import os
import sys
import logging

from datetime import datetime

class logger:
    def __init__(self, name, test_name):
        self.output_path = self.create_output_folder(test_name) + '\\'
        self.logger = logging.getLogger(name)
        # handler = logging.StreamHandler(stream=sys.stdout)
        # self.logger.addHandler(handler)
        sys.excepthook = self.handle_exception
        self.file_name = self.output_path + test_name + '.log'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename = self.file_name, filemode = 'w+')
        # with open(self.file_name, 'w+'):
        #     pass

    def create_output_folder(self, test_name):
        path = os.getcwd() + '\\Output\\' + test_name + '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
        os.makedirs(path)

        return path

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def log(self, line, line_end = '...'):
        print(f"{datetime.now()} {line}{line_end}")
        self.logger.info(line)