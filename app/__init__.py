import logging

from . import bq_util, core, enums, feeds, globals, vbt_backtest
from .bq_util import get_bid_ask_spread, get_order_book_stats
from .globals import (
    BLACKLISTED_SYMBOLS,
    LOGGING_FORMATTER,
    STUDY_END_DATE,
    STUDY_INST_TYPES,
    STUDY_START_DATE,
    STUDY_TIME_RANGE,
)

root_logger = logging.getLogger("app")
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())
root_logger.handlers[0].setFormatter(LOGGING_FORMATTER)
