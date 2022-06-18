from typing import Union
from cryptomart.enums import InstrumentType
import numpy as np
import pandas as pd
from cryptomart.feeds import FeedBase, OHLCVFeed
from IPython.display import display


class OHLCVColumn:
    open_time = "open_time"
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    volume = "volume"
    funding_rate = "funding_rate"
    returns = "returns"


class SpreadColumn(OHLCVColumn):
    zscore = "zscore"


class Spread(FeedBase):
    _metadata = ["ohlcv_a", "ohlcv_b"]

    def __init__(self, data=None, a: OHLCVFeed = None, b: OHLCVFeed = None):
        if a is not None and b is not None:
            self.ohlcv_a = a
            self.ohlcv_b = b

        if (np.array([a.inst_type, b.inst_type]) == InstrumentType.PERPETUAL).all():
            # Append funding rate column
            pass
    
        super().__init__(data=data)

    @classmethod
    def from_ohlcv(cls, ohlcv_a: OHLCVFeed, ohlcv_b: OHLCVFeed, z_score_period: int = 30):
        merged = ohlcv_b._df.merge(ohlcv_a._df, on=OHLCVColumn.open_time, suffixes=("_b", "_a"))
        data = pd.DataFrame()
        data[SpreadColumn.open_time] = merged[OHLCVColumn.open_time]
        data[SpreadColumn.open] = merged[OHLCVColumn.open + "_b"] - merged[OHLCVColumn.open + "_a"]
        data[SpreadColumn.high] = merged[OHLCVColumn.high + "_b"] - merged[OHLCVColumn.high + "_a"]
        data[SpreadColumn.low] = merged[OHLCVColumn.low + "_b"] - merged[OHLCVColumn.low + "_a"]
        data[SpreadColumn.close] = merged[OHLCVColumn.close + "_b"] - merged[OHLCVColumn.close + "_a"]
        data[SpreadColumn.volume] = (merged[OHLCVColumn.volume + "_b"] + merged[OHLCVColumn.volume + "_a"]) / 2
        data[SpreadColumn.returns] = merged[OHLCVColumn.returns + "_b"] - merged[OHLCVColumn.returns + "_a"]

        start_time = max(ohlcv_a.earliest_time, ohlcv_b.earliest_time)
        end_time = min(ohlcv_a.latest_time, ohlcv_b.latest_time)

        data = data[data[SpreadColumn.open_time].between(start_time, end_time)]
        metric_column = data[OHLCVColumn.close]
        data[SpreadColumn.zscore] = (
            metric_column - metric_column.rolling(z_score_period).mean()
        ) / metric_column.rolling(z_score_period).std()

        return cls(data=data, a=ohlcv_a, b=ohlcv_b)

    @property
    def underlyings(self):
        a = self.ohlcv_a[np.isin(self.ohlcv_a[OHLCVColumn.open_time], self[SpreadColumn.open_time])].reset_index(
            drop=True
        )
        b = self.ohlcv_b[np.isin(self.ohlcv_b[OHLCVColumn.open_time], self[SpreadColumn.open_time])].reset_index(
            drop=True
        )
        return pd.concat([a._df, b._df], keys=[a._underlying_info, b._underlying_info], axis=1)

    def underlying_col(self, column_name=SpreadColumn.close):
        index = self[SpreadColumn.open_time]
        df = self.underlyings.droplevel(0, axis=1)[column_name].set_axis(
            self.underlyings.columns.get_level_values(0).unique(), axis=1
        )
        df.index = index
        return df

    @property
    def volatility(self):
        DAYS_IN_YEAR = 365
        return self[OHLCVColumn.returns].std() * np.sqrt(DAYS_IN_YEAR)

    @property
    def _underlying_info(self):
        return f"{self.ohlcv_b._underlying_info} - {self.ohlcv_a._underlying_info}"

    def __str__(self):
        return super().__str__() + self._underlying_info + "\n"

    def profile(self):
        self.show()
        display(self.underlyings)
        self.ohlcv_a.show()
        display(self.ohlcv_a.orig_starttime, self.ohlcv_a.orig_endtime)
        self.ohlcv_b.show()
        display(self.ohlcv_b.orig_starttime, self.ohlcv_b.orig_endtime)
