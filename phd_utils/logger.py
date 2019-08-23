'''Logging functions for MBS and PBS tests'''
# import atexit
import logging
import os
import subprocess
import sys
from datetime import datetime
from distutils.dir_util import copy_tree
from pathlib import Path


class Logger:
    '''Logging functions and output path'''
    def __init__(self, test_name, copy_path=None):
        # self.name = name
        self.test_name = test_name
        self.copy_path = copy_path

    def __enter__(self):
        class LoggingStructure:
            def __init__(self, test_name, copy_path=None):
                self.copy_path = copy_path
                self.test_name = test_name
                self.output_path = self.create_output_folder(test_name)
                sys.excepthook = self.handle_exception
                self.file_name = self.output_path / f"{test_name}.log"
                self.logger = logging.getLogger()
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s - %(message)s',
                                    filename=self.file_name)
                # atexit.register(self.finalise)
                self.log(f"Starting {test_name}")
                self.log(subprocess.check_output(["git", "describe", "--always"]).strip())

            def finalise(self):
                '''Copy directory and finish log at test end'''
                self.log("Finalising")
                handlers = self.logger.root.handlers.copy()
                for handler in handlers:
                    self.logger.root.removeHandler(handler)
                    handler.flush()
                    handler.close()

                if self.copy_path is not None:
                    if not isinstance(self.copy_path, str):
                        raise "Copy path must be a string directory"

                    path = Path(self.copy_path)

                    if not path.exists():
                        raise f"Cannot find {self.copy_path}"

                    # self.log("Copying data folder")
                    current = datetime.now().strftime("%Y%m%dT%H%M%S")
                    current = f"{self.test_name}_{current}"
                    copy_folder = path / current
                    os.mkdir(copy_folder)

                    copy_tree(self.output_path.absolute().as_posix(), copy_folder.absolute().as_posix())

            @classmethod
            def create_output_folder(cls, test_name):
                '''create an output folder for the log and any test results'''
                current = datetime.now().strftime("%Y%m%dT%H%M%S")
                output_folder = Path(os.getcwd()) / "Output" / f"{test_name}_{current}"
                os.makedirs(output_folder)

                return output_folder

            def handle_exception(self, exc_type, exc_value, exc_traceback):
                '''log exceptions to file'''
                if issubclass(exc_type, KeyboardInterrupt):
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                    return

                self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
                raise exc_type(exc_value)

            def log(self, line, line_end='...'):
                '''add to log file'''
                print(f"{datetime.now()} {line}{line_end}")
                self.logger.info(line)

        self.logger = LoggingStructure(self.test_name, self.copy_path)

        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.finalise()
