import logging
from cryptomart.globals import LOGGING_FORMATTER

from .core import Client
from .globals import STUDY_START_DATE, BLACKLISTED_SYMBOLS

root_logger = logging.getLogger("app")
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())
root_logger.handlers[0].setFormatter(LOGGING_FORMATTER)