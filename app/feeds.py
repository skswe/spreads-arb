import datetime
from typing import List

import cryptomart as cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cryptomart.feeds import FundingRateFeed, OHLCVFeed, TSFeedBase
from IPython.display import display

from .enums import Exchange, InstrumentType, Interval, OHLCVColumn, SpreadColumn

client = cm.Client(quiet=True)


class QuotesFeed(TSFeedBase):
    _metadata = TSFeedBase._metadata + [
        "exchange_name",
        "symbol",
        "inst_type",
        "interval",
        "orig_starttime",
        "orig_endtime",
    ]

    @staticmethod
    def preprocess(df, start, end, freq):
        df = df.reindex(pd.date_range(start, end, freq=freq)[:-1]).reset_index().rename(columns={"index": "timestamp"})
        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
        return df

    def __init__(
        self,
        data=None,
        exchange_name: str = "",
        symbol: str = None,
        inst_type: InstrumentType = None,
        interval: Interval = None,
        timedelta: datetime.timedelta = None,
        starttime: datetime.datetime = None,
        endtime: datetime.datetime = None,
        **kwargs,
    ):
        if symbol is not None:
            data = self.preprocess(data, starttime, endtime, timedelta)

        super().__init__(data=data, time_column="timestamp", value_column="mid_price", timedelta=timedelta, **kwargs)

        self.exchange_name = exchange_name
        self.symbol = symbol
        self.inst_type = inst_type
        self.interval = interval
        self.orig_starttime = starttime
        self.orig_endtime = endtime

    def returns(self, column="mid_price"):
        return (
            (((self[column] - self[column].shift(1)) / self[column].shift(1)) * 100)
            .rename("returns")
            .set_axis(self[self.time_column])
        )

    @property
    def _underlying_info(self):
        return f"ohlcv.{self.exchange_name}.{self.inst_type}.{self.symbol}"

    def zscore(self, column="mid_price"):
        s = self.set_index(self.time_column)[column]
        return (
            ((s - s.rolling("30D").mean()) / s.rolling("30D").std())
            .rename("zscore")
            .set_axis(self[self.time_column])
        )


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
        self.resample_to_days = lambda: self

    @classmethod
    def from_api(
        cls,
        symbol: str,
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
        x = a.merge(b, on="open_time")
        x[SpreadColumn.open] = x.eval("open_y - open_x")
        x[SpreadColumn.high] = x.eval("high_y - high_x")
        x[SpreadColumn.low] = x.eval("low_y - low_x")
        x[SpreadColumn.close] = x.eval("close_y - close_x")
        x[SpreadColumn.volume] = x.eval("(volume_y + volume_x) / 2")
        x = x[
            [
                "open_time",
                SpreadColumn.open,
                SpreadColumn.high,
                SpreadColumn.low,
                SpreadColumn.close,
                SpreadColumn.volume,
            ]
        ]
        x = x.fillna(np.nan)
        x = x.reset_index(drop=True)
        return x

    @classmethod
    def from_quotes(cls, a: QuotesFeed, b: QuotesFeed):
        x = a.merge(b, on="timestamp")
        x["bid_price"] = x.eval("bid_price_y - bid_price_x")
        x["ask_price"] = x.eval("ask_price_y - ask_price_x")
        x["mid_price"] = x.eval("mid_price_y - mid_price_x")
        x["bid_amount"] = x.eval("(bid_amount_y + bid_amount_x) / 2")
        x["ask_amount"] = x.eval("(ask_amount_y + ask_amount_x) / 2")
        x = x[
            [
                "timestamp",
                "bid_price",
                "ask_price",
                "bid_amount",
                "ask_amount",
                "mid_price",
            ]
        ]
        x = x.fillna(np.nan)
        x = x.reset_index(drop=True)
        obj = cls(x, a, b)
        obj.time_column = "timestamp"
        obj.value_column = "mid_price"
        obj.resample_to_days = lambda: obj.set_index(obj.time_column).resample("1d").last().reset_index()
        return obj

    def add_bid_ask_spread(
        self, bid_ask_spread_a: FundingRateFeed, bid_ask_spread_b: FundingRateFeed, fillna=True, type="sum"
    ):
        """Add bid ask spread feeds to underlying OHLCVFeeds"""
        assert type in ["sum", "mean"]

        if type == "mean":
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
        elif type == "sum":
            self.ohlcv_list[0] = (
                self.ohlcv_list[0]
                .drop(columns=["bas", "bid_amount", "ask_amount"], errors="ignore")
                .set_index(self.ohlcv_list[0].time_column)
                .join(bid_ask_spread_a, how="left")
                .reset_index()
            )
            self.ohlcv_list[1] = (
                self.ohlcv_list[1]
                .drop(columns=["bas", "bid_amount", "ask_amount"], errors="ignore")
                .set_index(self.ohlcv_list[1].time_column)
                .join(bid_ask_spread_b, how="left")
                .reset_index()
            )

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
        ohlcv_a = ohlcv_a.set_index(self.time_column).resample("1d").last().reset_index()
        ohlcv_b = ohlcv_b.set_index(self.time_column).resample("1d").last().reset_index()
        return ohlcv_b.returns(column) - ohlcv_a.returns(column)

    def zscore(self, column=SpreadColumn.close, period=30):
        s = self.set_index(self.time_column)[column]
        return (
            ((s - s.rolling(period).mean()) / s.rolling(period).std())
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

    def var(self, column, percentile=5):
        return abs(
            self.ohlcv_list[column].dropna().returns().quantile(percentile / 100, interpolation="midpoint") / 100
        )

    def long_var(self):
        # Long means column 0 is short and column 1 is long
        # Column 0 VAR should be 95th percentile (positive return)
        # Column 1 VAR should be 5th percentile (negative return)
        return [self.var(0, 95), self.var(1, 5)]

    def short_var(self):
        # Short means column 0 is long and column 0 is short
        # Column 0 VAR should be 5th percentile (negative return)
        # Column 1 VAR should be 95th percentile (positive return)
        return [self.var(0, 5), self.var(1, 95)]

    @property
    def underlyings(self):
        return pd.concat(
            [df.set_index(self.time_column) for df in self.ohlcv_list],
            keys=[df._underlying_info for df in self.ohlcv_list],
            axis=1,
        )

    def underlying_col(self, column=SpreadColumn.close):
        return self.underlyings.loc[:, (slice(None), column)]

    @property
    def _underlying_info(self):
        return f"Spread({self.ohlcv_list[1]._underlying_info} - {self.ohlcv_list[0]._underlying_info})"

    def plot(self, *args, columns="all", kind="line", **kwargs):
        df = self.resample_to_days()
        if kind == "ohlcv":
            return go.Figure(
                data=go.Candlestick(
                    x=self[self.time_column],
                    open=self[SpreadColumn.open],
                    high=self[SpreadColumn.high],
                    low=self[SpreadColumn.low],
                    close=self[SpreadColumn.close],
                ),
                layout=go.Layout(title=self._underlying_info),
            )
        else:
            if columns == "all":
                columns = list(set(SpreadColumn._values()) - {self.time_column})
            elif type(columns) == str:
                columns = [columns]
            fig: go.Figure = pd.DataFrame(df.set_index(self.time_column)[columns]).plot(
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
