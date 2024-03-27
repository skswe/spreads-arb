import logging

import dotenv
from tqdm import tqdm

from . import data_prep, enums, errors, feeds, globals, pandas_backtest, plotting, vbt_backtest, util

root_logger = logging.getLogger("app")
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())
root_logger.handlers[0].setFormatter(globals.LOGGING_FORMATTER)

dotenv.load_dotenv()
tqdm.pandas()
