"""This module contains a function to load ohlcv data
"""

import pickle

import pandas as pd

import cryptomart as cm

from ..enums import Exchange
from ..errors import NotSupportedError
from ..globals import BLACKLISTED_SYMBOLS, STUDY_INST_TYPES
from ..util import cached


@cached("/tmp/cache/all_ohlcv", refresh=False)
def all_ohlcv(start, end, interval, **cache_kwargs) -> pd.DataFrame:
    """Get the OHLCV timeseries for all instruments

    Args:
        start: start time for desired timeseries
        end: end time for desired timeseries
        interval: interval for desired timeseries

    Returns:
        DataFrame with keys (exchange, inst_type, symbol) and column [ohlcv, rows, missing_rows, earliest_time, latest_time, gaps]
    """
    cm_client = cm.Client(quiet=True, instrument_cache_kwargs={"refresh": True})
    ohlcvs = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], []], names=["exchange", "inst_type", "symbol"]))

    for exchange in Exchange:
        for inst_type in STUDY_INST_TYPES:
            instruments = cm_client.instrument_info(exchange, inst_type)
            instruments = instruments[~instruments["cryptomart_symbol"].isin(BLACKLISTED_SYMBOLS)]
            for symbol in instruments["cryptomart_symbol"]:
                try:
                    data = cm_client.ohlcv(
                        exchange,
                        symbol,
                        inst_type,
                        start,
                        end,
                        interval,
                        cache_kwargs={"disabled": False},
                    )
                except NotSupportedError:
                    # Instrument not supported by exchange
                    continue
                ohlcvs.at[(exchange, inst_type, symbol), "ohlcv"] = pickle.dumps(data)
                ohlcvs.at[(exchange, inst_type, symbol), "rows"] = len(data.valid_rows)
                ohlcvs.at[(exchange, inst_type, symbol), "missing_rows"] = len(data.missing_rows)
                ohlcvs.at[(exchange, inst_type, symbol), "earliest_time"] = data.earliest_time
                ohlcvs.at[(exchange, inst_type, symbol), "latest_time"] = data.latest_time
                ohlcvs.at[(exchange, inst_type, symbol), "gaps"] = data.gaps

    return ohlcvs
