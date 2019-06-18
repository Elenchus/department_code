import FileUtils
import sys
if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "test.log")
    raise RuntimeError("Test unhandled")