"""This module contains utility functions for data preparation
"""

import glob
import os

import pandas as pd
from tqdm import tqdm


def create_exchange_dataframe(include_inst_type=False):
    """Create a DataFrame with the MultiIndex (exchange, symbol) or (exchange, inst_type, symbol)

    Args:
        include_inst_type: Whether to include the inst_type key. Defaults to False.

    Returns:
        Empty DataFrame with the MultiIndex (exchange, symbol) or (exchange, inst_type, symbol)
    """
    if not include_inst_type:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["exchange", "symbol"]))
    else:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], []], names=["exchange", "inst_type", "symbol"]))


def get_cryptomart_data_iterator(base_path, filter_list=None, show_progress=False):
    """Iterates through directory structure `base_path/exchange/symbol` and returns exchange, symbol, filepath

    Args:
        base_path: base path containing the exchange directories
        filter_list: (Optional) list of symbols to filter the iterator by. Defaults to None.
        show_progress: Shows progress bar if true. Defaults to False.

    Yields:
        Tuple containing (exchange, symbol, filepath) for each file in the directory structure
    """
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
