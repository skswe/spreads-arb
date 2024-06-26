"""This module contains DataFrame wrappers for backtesting via pandas
"""

import numpy as np
import pandas as pd
from IPython.display import display

from .. import plotting
from ..data_prep import get_fee_info

fee_info = get_fee_info().sort_index()


def fast_groupby(res):
    """Wrap output from a groupby-apply where the result is a dictionary"""
    if not isinstance(res, pd.DataFrame):
        return pd.DataFrame(res.to_list())
    else:
        return None


class LazyDataFrameHolder:
    """Wraps a parquet file and provides methods to get and plot the data."""

    def __init__(self, filepath, name=""):
        self.filepath = filepath
        self.name = name

    def __repr__(self):
        return f"ldfh({self.filepath})"

    @staticmethod
    def _get_between(df, start=None, end=None, padding=0):
        start = start or pd.Timestamp("2000-01-01")
        end = end or pd.Timestamp("2099-12-31")
        freq = df.timestamp.diff().median()
        return df[df.timestamp.between(start - freq * padding, end + freq * padding)]

    def get(self, start=None, end=None, padding=0):
        return self._get_between(pd.read_parquet(self.filepath), start, end, padding)

    def plot(self, start=None, end=None, padding=0, df=None):
        df = df or self.get(start, end, padding)
        plotting.plot_df_subplots(
            df=df,
            x="timestamp",
            y=[
                ["price_signal"],
                ["bid_amount", "ask_amount"],
                ["filled"],
            ],
            row_heights=[0.5, 0.2, 0.2],
            update_layout_kwargs={"title": self.name},
        )


class LazySpreadHolder:
    """Wraps two parquet files which form a spread and provides methods to get and plot the data."""

    def __init__(
        self,
        leg_x: LazyDataFrameHolder,
        leg_y: LazyDataFrameHolder,
        name: str,
        z_score_period=pd.Timedelta(days=30),
        z_score_thresh=1.0,
        min_trade_time=None,
    ):
        self.leg_x = leg_x
        self.leg_y = leg_y
        self.name = name
        self.z_score_period = z_score_period
        self.z_score_thresh = z_score_thresh
        self.min_trade_time = min_trade_time

    @staticmethod
    def _get_between(df, start=None, end=None, padding=0):
        start = start or pd.Timestamp("2000-01-01")
        end = end or pd.Timestamp("2099-12-31")
        freq = df.timestamp.diff().median()
        return df[df.timestamp.between(start - freq * padding, end + freq * padding)]

    def get_x(self, start=None, end=None, padding=0):
        return self._get_between(self.leg_x.get(), start, end, padding)

    def get_y(self, start=None, end=None, padding=0):
        return self._get_between(self.leg_y.get(), start, end, padding)

    def get(self, start=None, end=None, padding=0):
        spread = self.get_x().merge(self.get_y(), on="timestamp")
        spread["bid_price"] = spread.bid_price_y - spread.bid_price_x
        spread["ask_price"] = spread.ask_price_y - spread.ask_price_x
        spread["price_signal"] = spread.price_signal_y - spread.price_signal_x
        # spread["price_signal"] = np.where(spread["filled_x"] | spread["filled_y"], np.nan, spread["price_signal"])
        spread["bid_amount"] = (spread.bid_amount_y + spread.bid_amount_x) / 2
        spread["ask_amount"] = (spread.ask_amount_y + spread.ask_amount_x) / 2

        data_frequency = spread.timestamp.diff().median()
        row_period = self.z_score_period / data_frequency
        # print(f"Row period: {row_period}")
        assert row_period == int(row_period), f"z_score_period must be a multiple of data_frequency ({data_frequency})"

        rolling_window = spread["price_signal"].rolling(int(row_period) + 1, min_periods=1)
        spread["zscore"] = (spread["price_signal"] - rolling_window.mean()) / rolling_window.std()
        spread["zscore"] = np.where(spread.index <= int(row_period), np.nan, spread["zscore"])

        # Create two seperate crossed signals so they don't interfere with each other
        # when there is a minimum trade time
        long_crossed = ((spread["zscore"] * spread["zscore"].shift(1)) <= 0).astype(int)
        short_crossed = ((spread["zscore"] * spread["zscore"].shift(1)) <= 0).astype(int)
        long_signal = (spread["zscore"] < -self.z_score_thresh).astype(int)
        short_signal = (spread["zscore"] > self.z_score_thresh).astype(int)

        # Compute entries with crossed signals
        # Hold signal goes to 1 when there is a signal and stays 1 until the signal is crossed
        long_hold = long_signal.groupby(long_crossed.cumsum()).cummax()
        short_hold = short_signal.groupby(short_crossed.cumsum()).cummax()
        spread["long_entry"] = (long_hold.diff() == 1).astype(int)
        spread["short_entry"] = (short_hold.diff() == 1).astype(int)

        if self.min_trade_time is not None:
            trade_timeout_period = int(self.min_trade_time / data_frequency)
            short_blocked = (spread["short_entry"].rolling(trade_timeout_period, min_periods=1).sum() > 0).shift(1)
            long_blocked = (spread["long_entry"].rolling(trade_timeout_period, min_periods=1).sum() > 0).shift(1)

            # Disable long entries and short exits for `min_trade_time` after short entries
            long_signal.loc[short_blocked == 1] = 0
            short_crossed.loc[short_blocked == 1] = 0
            # Disable short entries and long exits for `min_trade_time` after long entries
            short_signal.loc[long_blocked == 1] = 0
            long_crossed.loc[long_blocked == 1] = 0

            # Recompute hold signals
            long_hold = long_signal.groupby(long_crossed.cumsum()).cummax()
            short_hold = short_signal.groupby(short_crossed.cumsum()).cummax()

            spread["long_entry"] = (long_hold.diff() == 1).astype(int)
            spread["short_entry"] = (short_hold.diff() == 1).astype(int)

        spread["long_exit"] = (long_hold.diff() == -1).astype(int)
        spread["short_exit"] = (short_hold.diff() == -1).astype(int)

        long_spread_trade = (spread.long_entry | spread.short_exit).astype(bool)
        short_spread_trade = (spread.short_entry | spread.long_exit).astype(bool)

        # Long spread means leg_x is sold (bid) and leg_y is bought (ask)
        spread.loc[long_spread_trade, "real_price_y"] = spread.loc[long_spread_trade, "ask_price_y"]
        spread.loc[long_spread_trade, "real_price_x"] = spread.loc[long_spread_trade, "bid_price_x"]
        # short spread means leg_x is bought (ask) and leg_y is sold (bought)
        spread.loc[short_spread_trade, "real_price_y"] = spread.loc[short_spread_trade, "bid_price_y"]
        spread.loc[short_spread_trade, "real_price_x"] = spread.loc[short_spread_trade, "ask_price_x"]

        spread["real_price"] = spread["real_price_y"] - spread["real_price_x"]

        # apply padding to the amount columns
        long_spread_trade = long_spread_trade | long_spread_trade.shift(1) | long_spread_trade.shift(-1)
        short_spread_trade = short_spread_trade | short_spread_trade.shift(1) | short_spread_trade.shift(-1)

        spread.loc[long_spread_trade, "real_amount_y"] = spread.loc[long_spread_trade, "ask_amount_y"]
        spread.loc[long_spread_trade, "real_amount_x"] = spread.loc[long_spread_trade, "bid_amount_x"]
        spread.loc[short_spread_trade, "real_amount_y"] = spread.loc[short_spread_trade, "bid_amount_y"]
        spread.loc[short_spread_trade, "real_amount_x"] = spread.loc[short_spread_trade, "ask_amount_x"]

        spread["real_amount"] = spread[["real_amount_y", "real_amount_x"]].min(axis=1)

        spread.loc[long_spread_trade, "real_amount"] = spread.loc[
            long_spread_trade, ["ask_amount_y", "bid_amount_x"]
        ].min(axis=1)
        spread.loc[short_spread_trade, "real_amount"] = spread.loc[
            short_spread_trade, ["ask_amount_x", "bid_amount_y"]
        ].min(axis=1)

        spread.drop(
            columns=[
                x
                for x in spread.columns
                if (x.endswith("_x") or x.endswith("_y")) and not x.startswith("filled") and not x.startswith("real")
            ],
            inplace=True,
        )

        sort_cols = ["timestamp", "price_signal", "zscore", "bid_price", "bid_amount", "ask_price", "ask_amount"]
        spread = spread[sort_cols + [x for x in spread.columns if x not in sort_cols]]

        return self._get_between(spread, start, end, padding)

    def plot(self, start=None, end=None, padding=0, spread=None, **update_layout_kwargs):
        spread = spread or self.get(start, end, padding)

        plotting.plot_df_subplots(
            df=spread,
            x="timestamp",
            y=[
                ["price_signal", "zscore"],
                ["real_amount"],
                ["real_price", "long_entry", "long_exit", "short_entry", "short_exit"],
                ["filled_x", "filled_y"],
            ],
            row_heights=[0.5, 0.2, 0.2, 0.2],
            multi_y=[["zscore"], [], ["real_price"], []],
            scatter_kwargs=[
                [{}, {"opacity": 0.4}],
            ],
            update_layout_kwargs={"title": self.name, **update_layout_kwargs},
        )

        self.leg_x.plot(start, end, padding)
        self.leg_y.plot(start, end, padding)

    def get_trades(self):
        def group_trade(g, side="long"):
            if len(g) < 2:
                return {}

            side_multiplier = 1 if side == "long" else -1

            entry_price = g.real_price.iat[0]
            exit_price = g.real_price.iat[1]
            amount = g.real_amount.min()
            entry_time = g.timestamp.iat[0]
            exit_time = g.timestamp.iat[1]
            duration = exit_time - entry_time
            entry_zscore = g.zscore.iat[0]
            exit_zscore = g.zscore.iat[1]
            z_score_movement = abs(exit_zscore - entry_zscore)
            raw_profit = side_multiplier * (exit_price - entry_price) * amount

            entry_price_x = g.real_price_x.iat[0]
            entry_price_y = g.real_price_y.iat[0]
            entry_amount_x = g.real_amount_x.iat[0]
            entry_amount_y = g.real_amount_y.iat[0]
            exit_price_x = g.real_price_x.iat[1]
            exit_price_y = g.real_price_y.iat[1]
            exit_amount_x = g.real_amount_x.iat[1]
            exit_amount_y = g.real_amount_y.iat[1]

            transacted_value = (entry_price_x + entry_price_y + exit_price_x + exit_price_y) * amount

            symbol, exch_0, exch_1 = self.name.split(".")
            fee_pct_0 = fee_info.loc[exch_0, symbol].iloc[0].fee_pct
            fee_pct_1 = fee_info.loc[exch_1, symbol].iloc[0].fee_pct
            fees_paid = ((entry_price_x + exit_price_x) * amount * fee_pct_0) + (
                (entry_price_y + exit_price_y) * amount * fee_pct_1
            )
            profit = raw_profit - fees_paid

            return {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "amount": amount,
                "transacted_value": transacted_value,
                "raw_profit": raw_profit,
                "fees_paid": fees_paid,
                "profit": profit,
                "invalid": g.filled_x.any() or g.filled_y.any(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration": duration,
                "entry_zscore": entry_zscore,
                "exit_zscore": exit_zscore,
                "z_score_movement": z_score_movement,
                "entry_price_x": entry_price_x,
                "entry_price_y": entry_price_y,
                "entry_amount_x": entry_amount_x,
                "entry_amount_y": entry_amount_y,
                "exit_price_x": exit_price_x,
                "exit_price_y": exit_price_y,
                "exit_amount_x": exit_amount_x,
                "exit_amount_y": exit_amount_y,
            }

        spread = self.get()
        long_filter = spread[(spread.long_entry | spread.long_exit).astype(bool)]
        short_filter = spread[(spread.short_entry | spread.short_exit).astype(bool)]

        long_trades = fast_groupby(
            long_filter.groupby(long_filter.reset_index(drop=True).index // 2).apply(group_trade, side="long")
        )

        short_trades = fast_groupby(
            short_filter.groupby(short_filter.reset_index(drop=True).index // 2).apply(group_trade, side="short")
        )

        trades = pd.concat([long_trades, short_trades]).sort_values("entry_time").reset_index(drop=True)
        return trades

    def get_trades_summary(self):
        trades = self.get_trades()
        return pd.Series(
            {
                "avg_profit": trades.profit.mean(),
                "best_profit": trades.profit.max(),
                "worst_profit": trades.profit.min(),
                "winning_trades": (trades.profit > 0).sum(),
                "losing_trades": (trades.profit <= 0).sum(),
                "win_pct": (trades.profit > 0).sum() / len(trades),
                "avg_duration": trades.duration.mean(),
                "avg_transacted_value": trades.transacted_value.mean(),
                "avg_fees_paid": trades.fees_paid.mean(),
                "highest_z_score_movement": trades.z_score_movement.max(),
                "lowest_z_score_movement": trades.z_score_movement.min(),
                "avg_z_score_movement": trades.z_score_movement.mean(),
            }
        )


class Trades(pd.DataFrame):
    """Wrapper for a DataFrame containing trades data"""

    _metadata = ["other_granularity_refs"]

    str_to_timedelta = {
        "1s": pd.Timedelta(seconds=1),
        "3s": pd.Timedelta(seconds=3),
        "10s": pd.Timedelta(seconds=10),
        "1m": pd.Timedelta(minutes=1),
        "10m": pd.Timedelta(minutes=10),
    }

    @classmethod
    def _pandas_constructor(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @property
    def _constructor(self):
        return self._pandas_constructor

    def __init__(self, data=None, other_granularity_refs=None, **kwargs):
        super().__init__(data=data, **kwargs)
        if other_granularity_refs:
            self.other_granularity_refs = other_granularity_refs

    def top_profit(self, n=10, valid_only=False):
        df = self
        if valid_only:
            df = df[df.invalid == False]
        return df.sort_values("profit", ascending=False).head(n)

    def switch_granularity(self, granularity):
        if isinstance(granularity, str):
            granularity = self.str_to_timedelta[granularity]
        return Trades(self.other_granularity_refs[granularity], other_granularity_refs=self.other_granularity_refs)

    def get(self, idx, padding=0):
        """Get the spread during the trade"""
        trade = self.loc[idx]
        return self.loc[idx].lsh.get(trade.entry_time, trade.exit_time, padding)

    def plot(self, idx, padding=5, **kwargs):
        """Get the spread during the trade"""
        trade = self.loc[idx]
        trade_title = f"{trade.symbol}, ({trade.exchange_y} - {trade.exchange_x}) {idx} ({trade.entry_time} - {trade.exit_time}), profit: {trade.profit:.2f}, z_score_movement {trade.z_score_movement:.2f}"
        display(trade.to_frame().T)
        self.loc[idx].lsh.plot(
            start=trade.entry_time, end=trade.exit_time, padding=padding, title=trade_title, **kwargs
        )

        display(f"leg_0 ({trade.exchange_y})")
        display(trade.lsh.get_x(trade.entry_time, trade.exit_time))
        display(f"leg_1 ({trade.exchange_x})")
        display(trade.lsh.get_y(trade.entry_time, trade.exit_time))

    def plot_granularity(self, idx, granularity, padding=5, **kwargs):
        trade = self.loc[idx]
        try:
            self.other_granularity_refs[self.str_to_timedelta[granularity]].pipe(
                lambda df: df[
                    (df.symbol == trade.symbol)
                    & (df.exchange_x == trade.exchange_x)
                    & (df.exchange_y == trade.exchange_y)
                ]
            ).iloc[0].lsh.plot(start=trade.entry_time, end=trade.exit_time, padding=padding)
        except KeyError:
            raise KeyError(f"Symbol {trade.symbol} not found at granularity {granularity}")

    def stats(self, valid_only=False):
        if valid_only:
            df = self[self.invalid == False]
        else:
            df = self
        return pd.Series(
            {
                "total_trades": df.shape[0],
                "winning_trades": (df.profit > 0).sum(),
                "losing_trades": (df.profit < 0).sum(),
                "win_pct": (df.profit > 0).mean(),
                "average_duration": df.duration.mean(),
                "best_trade": df.profit.max(),
                "worst_trade": df.profit.min(),
                "avg_profit": df.profit.mean(),
                "total_profit": df.profit.sum(),
                "total_transacted_value": df.transacted_value.sum(),
            }
        )

    def in_range(self, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        return self[(self["entry_time"] >= start) & (self["entry_time"] <= end)]
