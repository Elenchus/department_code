import file_utils
import sys
if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "test.log")
    raise RuntimeError("Test unhandled")