"""This module contains a function to load funding rate data
"""

import pickle

import cryptomart as cm
import pandas as pd
from ..util import cached

from ..enums import Exchange
from ..errors import NotSupportedError
from ..globals import BLACKLISTED_SYMBOLS, STUDY_INST_TYPES


@cached("/tmp/cache/all_funding_rates", refresh=False)
def all_funding_rates(start, end, **cache_kwargs) -> pd.DataFrame:
    """Get the funding rate timeseries for all instruments

    Args:
        start: start time for desired timeseries
        end: end time for desired timeseries

    Returns:
        DataFrame with keys (exchange, inst_type, symbol) and column [funding_rate]
    """
    cm_client = cm.Client(quiet=True)
    funding_rates = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([[], [], []], names=["exchange", "inst_type", "symbol"])
    )

    for exchange in Exchange:
        for inst_type in STUDY_INST_TYPES:
            instruments = cm_client.instrument_info(exchange, inst_type)
            instruments = instruments[~instruments["cryptomart_symbol"].isin(BLACKLISTED_SYMBOLS)]
            for symbol in instruments["cryptomart_symbol"]:
                try:
                    fr = cm_client.funding_rate(exchange, symbol, start, end, cache_kwargs={"disabled": False})
                except NotSupportedError:
                    # Instrument not supported by exchange
                    continue
                funding_rates.at[(exchange, inst_type, symbol), "funding_rate"] = pickle.dumps(fr)

    return funding_rates
