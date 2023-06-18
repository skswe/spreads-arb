import pandas as pd
from IPython.display import display


class Trades(pd.DataFrame):
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
