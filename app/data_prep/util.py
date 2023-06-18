import glob
import os

import pandas as pd
from tqdm import tqdm


def create_exchange_dataframe(include_inst_type=False):
    if not include_inst_type:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["exchange", "symbol"]))
    else:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], []], names=["exchange", "inst_type", "symbol"]))


def get_cryptomart_data_iterator(base_path, filter_list=None, show_progress=False):
    """Iterates through directory structure `base_path/exchange/symbol` and returns exchange, symbol, filepath"""
    glob_path = os.path.join(base_path, "*", "*")
    filepaths = glob.glob(glob_path)

    def filter_filepaths(filepath):
        *_, exchange, filename = filepath.split(os.path.sep)
        symbol = os.path.splitext(filename)[0]
        if (exchange, symbol) in filter_list:
            return True
        else:
            return False

    if filter_list is not None:
        filepaths = list(filter(filter_filepaths, filepaths))

    if show_progress:
        filepaths = tqdm(filepaths)

    for filepath in filepaths:
        *_, exchange, filename = filepath.split(os.path.sep)
        symbol = os.path.splitext(filename)[0]
        yield exchange, symbol, filepath
