import logging
import os
import sys

MAIN_LOGGER_NAME = "RUPQ"


def set_logger(name=MAIN_LOGGER_NAME, verbosity_level=logging.INFO):
    # setting up logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  - %(message)s")
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(verbosity_level)


def set_logger_logdir(logdir, name=MAIN_LOGGER_NAME):
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(os.path.join(logdir, ".log"))
    logger.addHandler(file_handler)


def get_logger():
    return logging.getLogger(MAIN_LOGGER_NAME)
