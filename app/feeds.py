from typing import List

import cryptomart as cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cryptomart.feeds import FundingRateFeed, OHLCVFeed, TSFeedBase
from IPython.display import display

from .enums import Exchange, InstrumentType, Interval, OHLCVColumn, SpreadColumn, Symbol

client = cm.Client(quiet=True)


class Spread(TSFeedBase):
    _metadata = TSFeedBase._metadata + [
        "ohlcv_list",
        "funding_rate",
        "bid_ask_spread",
    ]

    def __init__(self, data=None, a: OHLCVFeed = None, b: OHLCVFeed = None):
        timedelta = a.timedelta if a is not None else None
        super().__init__(
            data=data, time_column=SpreadColumn.open_time, value_column=SpreadColumn.open, timedelta=timedelta
        )

        self.ohlcv_list = [a, b]

    @classmethod
    def from_api(
        cls,
        symbol: Symbol,
        exchanges: List[Exchange],
        starttime: tuple[int],
        endtime: tuple[int],
        inst_types: List[InstrumentType] = [InstrumentType.PERPETUAL, InstrumentType.PERPETUAL],
        interval: Interval = Interval.interval_1d,
        cache_kwargs: dict = {},
    ):
        assert len(exchanges) == 2, "Must provide two exchanges"
        assert len(inst_types) == 2, "Must provide two inst_types"
        ohlcv_a = client.ohlcv(
            exchanges[0], symbol, inst_types[0], starttime, endtime, interval, cache_kwargs=cache_kwargs
        )
        ohlcv_b = client.ohlcv(
            exchanges[1], symbol, inst_types[1], starttime, endtime, interval, cache_kwargs=cache_kwargs
        )
        spread = cls.create_ohlcv(ohlcv_a, ohlcv_b)
        return cls(data=spread, a=ohlcv_a, b=ohlcv_b)

    @classmethod
    def from_ohlcv(
        cls,
        a: OHLCVFeed,
        b: OHLCVFeed,
    ):
        return cls(cls.create_ohlcv(a, b), a, b)

    @staticmethod
    def create_ohlcv(a: OHLCVFeed, b: OHLCVFeed):
        def aggregate(row):
            return pd.Series(
                {
                    SpreadColumn.open_time: row.open_time,
                    SpreadColumn.open: row.open_y - row.open_x,
                    SpreadColumn.high: row.high_y - row.high_x,
                    SpreadColumn.low: row.low_y - row.low_x,
                    SpreadColumn.close: row.close_y - row.close_x,
                    SpreadColumn.volume: (row.volume_y + row.volume_x) / 2,
                }
            )

        return a.merge(b, on="open_time").apply(aggregate, axis=1).fillna(np.nan).reset_index(drop=True)

    def add_bid_ask_spread(self, bid_ask_spread_a: FundingRateFeed, bid_ask_spread_b: FundingRateFeed, fillna=True):
        """Add bid ask spread feeds to underlying OHLCVFeeds"""
        bid_ask_spread_a = bid_ask_spread_a.resample(self.ohlcv_list[0].timedelta, on="date").median()
        bid_ask_spread_b = bid_ask_spread_b.resample(self.ohlcv_list[1].timedelta, on="date").median()

        self.ohlcv_list[0] = (
            self.ohlcv_list[0]
            .drop(columns="bid_ask_spread", errors="ignore")
            .set_index(self.ohlcv_list[0].time_column)
            .join(bid_ask_spread_a, how="left")
            .reset_index()
        )
        self.ohlcv_list[1] = (
            self.ohlcv_list[1]
            .drop(columns="bid_ask_spread", errors="ignore")
            .set_index(self.ohlcv_list[1].time_column)
            .join(bid_ask_spread_b, how="left")
            .reset_index()
        )

        if fillna:
            for x in range(len(self.ohlcv_list)):
                self.ohlcv_list[x].bid_ask_spread.fillna(
                    self.ohlcv_list[x].bid_ask_spread.expanding(1).mean(), inplace=True
                )
                self.ohlcv_list[x].bid_ask_spread.fillna(self.ohlcv_list[x].bid_ask_spread.mean(), inplace=True)

    def add_funding_rate(self, funding_rate_a: FundingRateFeed, funding_rate_b: FundingRateFeed, fillna=True):
        """Add funding rate feeds to underlying OHLCVFeeds"""
        funding_rate_a = funding_rate_a.resample(self.ohlcv_list[0].timedelta, on="timestamp").median()
        funding_rate_b = funding_rate_b.resample(self.ohlcv_list[1].timedelta, on="timestamp").median()

        self.ohlcv_list[0] = (
            self.ohlcv_list[0]
            .drop(columns="funding_rate", errors="ignore")
            .set_index(self.ohlcv_list[0].time_column)
            .join(funding_rate_a, how="left")
            .reset_index()
        )
        self.ohlcv_list[1] = (
            self.ohlcv_list[1]
            .drop(columns="funding_rate", errors="ignore")
            .set_index(self.ohlcv_list[1].time_column)
            .join(funding_rate_a, how="left")
            .reset_index()
        )

        if fillna:
            self.ohlcv_list[0].funding_rate.fillna(self.ohlcv_list[0].funding_rate.expanding(1).mean(), inplace=True)
            self.ohlcv_list[1].funding_rate.fillna(self.ohlcv_list[1].funding_rate.expanding(1).mean(), inplace=True)

    def returns(self, column=SpreadColumn.close):
        ohlcv_a, ohlcv_b = iter(self.ohlcv_list)
        return ohlcv_b.returns(column) - ohlcv_a.returns(column)

    def zscore(self, column=SpreadColumn.close, period=30):
        return (
            ((self[column] - self[column].rolling(period).mean()) / self[column].rolling(period).std())
            .rename("zscore")
            .set_axis(self[self.time_column])
        )

    def volatility(self, column=SpreadColumn.close):
        DAYS_IN_YEAR = 365
        return self.returns(column).std() * np.sqrt(DAYS_IN_YEAR)

    def value_at_risk(self, percentile=5):
        return tuple(
            x.dropna().returns().quantile(percentile / 100, interpolation="midpoint") / 100 for x in self.ohlcv_list
        )

    @property
    def underlyings(self):
        return pd.concat(
            [df.set_index(SpreadColumn.open_time) for df in self.ohlcv_list],
            keys=[df._underlying_info for df in self.ohlcv_list],
            axis=1,
        )

    def underlying_col(self, column=SpreadColumn.close):
        return self.underlyings.loc[:, (slice(None), column)]

    @property
    def _underlying_info(self):
        return f"Spread({self.ohlcv_list[1]._underlying_info} - {self.ohlcv_list[0]._underlying_info})"

    def plot(self, *args, columns="all", kind="line", **kwargs):
        if kind == "ohlcv":
            return go.Figure(
                data=go.Candlestick(
                    x=self[SpreadColumn.open_time],
                    open=self[SpreadColumn.open],
                    high=self[SpreadColumn.high],
                    low=self[SpreadColumn.low],
                    close=self[SpreadColumn.close],
                ),
                layout=go.Layout(title=self._underlying_info),
            )
        else:
            if columns == "all":
                columns = list(set(SpreadColumn._values()) - {SpreadColumn.open_time})
            elif type(columns) == str:
                columns = [columns]
            fig: go.Figure = pd.DataFrame(self.set_index(SpreadColumn.open_time)[columns]).plot(
                *args, kind=kind, title=self._underlying_info, backend="plotly", **kwargs
            )
            fig.update_layout(
                xaxis={
                    "rangeslider": {
                        "visible": True,
                    },
                    "type": "date",
                },
            )
            return fig
