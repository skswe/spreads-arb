import pickle

import cryptomart as cm
import pandas as pd
from ..util import cached

from ..enums import Exchange
from ..errors import NotSupportedError
from ..globals import BLACKLISTED_SYMBOLS, STUDY_INST_TYPES


@cached("/tmp/cache/all_ohlcv", refresh=False)
def all_ohlcv(start, end, interval, **cache_kwargs) -> pd.DataFrame:
    """Get OHLCV for all instruments"""
    cm_client = cm.Client(quiet=True)
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
