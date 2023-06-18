import pickle
import warnings

import cryptomart as cm
import pandas as pd
import pyutil

from ..enums import Exchange
from ..globals import BLACKLISTED_SYMBOLS, STUDY_INST_TYPES


@pyutil.cache.cached("/tmp/cache/all_bid_ask_spreads", refresh=False)
def all_bid_ask_spreads(start, end, **cache_kwargs) -> pd.DataFrame:
    """Get the bid ask spread timeseries for all instruments in the Order Book data mart"""
    cm_client = cm.Client(quiet=True)
    ba_spreads = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], [], []], names=["exchange", "inst_type", "symbol"]))

    for exchange in Exchange:
        for inst_type in STUDY_INST_TYPES:
            instruments = cm_client.instrument_info(exchange, inst_type)
            instruments = instruments[~instruments["cryptomart_symbol"].isin(BLACKLISTED_SYMBOLS)]
            for symbol in instruments["cryptomart_symbol"]:
                raise NotImplementedError
                ba_spreads.at[(exchange, inst_type, symbol), "bid_ask_spread"] = pickle.dumps(ba_spread)

    return ba_spreads


def dummy_bid_ask_spreads(ohlcvs, default_slippage=0.02, force_default=False):
    old_slippages = pd.read_pickle("data/old_slippages.pkl")
    old_slippages_symbol = old_slippages.groupby(level="symbol").mean()

    ohlcvs = ohlcvs.copy().sort_index()

    def applyfunc(serialized_df):
        df = pickle.loads(serialized_df)
        if force_default:
            slip = default_slippage
        else:
            try:
                slip = old_slippages.at[df.exchange_name, df.symbol].iloc[0]
            except KeyError:
                try:
                    slip = old_slippages_symbol.at[df.symbol]
                except KeyError:
                    slip = default_slippage

        bas = (
            (df.set_index("open_time").close * slip)
            .reset_index()
            .rename(columns={"open_time": "date", "close": "bid_ask_spread"})
        )
        return pickle.dumps(bas)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bas = ohlcvs.ohlcv.apply(applyfunc).to_frame("bid_ask_spread")
        bas["avg_slippage"] = bas.bid_ask_spread.apply(
            lambda x: pickle.loads(x).bid_ask_spread.mean()
        ) / ohlcvs.ohlcv.apply(lambda x: pickle.loads(x).close.mean())
        return bas
