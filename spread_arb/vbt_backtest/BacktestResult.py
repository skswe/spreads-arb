"""This module provides an API for analyzing individual spread backtest results. For
chained backtest results see `ChainedBacktestResult`
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
from IPython.display import display

from ..feeds import Spread
from ..util import BashFormatter


class BacktestResult:
    def __init__(self, portfolio: vbt.Portfolio, feed: Spread):
        self.portfolio = portfolio
        self.feed = feed

    def slippage(self) -> pd.DataFrame:
        """Return slippage paid by day.

        Returns dataframe with datetime index and columns [exchange_0, exchange_1]
        """
        close_prices = self.feed.underlying_col("close").droplevel(1, axis=1).rename(columns=lambda c: c.split(".")[1])
        filled_prices = (
            self.portfolio.orders.records_readable.groupby(["Timestamp", "Column"])
            .first()
            .Price.unstack()
            .rename(columns=lambda c: c.split(".")[1])
        )
        sizes = (
            self.portfolio.orders.records_readable.groupby(["Timestamp", "Column"])
            .sum()
            .Size.unstack("Column")
            .rename(columns=lambda c: c.split(".")[1])
        )
        return abs(close_prices - filled_prices) * sizes

    def plot(self) -> go.FigureWidget:
        """Return FigureWidget with plots for Spread, Zscore, Price, PnL, Returns, Slippage"""
        portfolio = self.portfolio
        feed = self.feed

        df = portfolio.trades.records.sort_values(["entry_idx", "col"])
        # For column 0, a short means we are long on the spread and vice versa
        df = df[df.col == 0]

        # direction 1 = short
        # direction 0 = long
        long_trades = df[df.direction == 1]
        short_trades = df[df.direction == 0]

        temp_signals = np.zeros(len(feed))
        temp_signals[long_trades.entry_idx] = True
        long_entries = pd.Series(index=feed[feed.time_column], data=temp_signals).astype(bool)

        temp_signals = np.zeros(len(feed))
        temp_signals[long_trades.exit_idx] = True
        long_exits = pd.Series(index=feed[feed.time_column], data=temp_signals).astype(bool)

        temp_signals = np.zeros(len(feed))
        temp_signals[short_trades.entry_idx] = True
        short_entries = pd.Series(index=feed[feed.time_column], data=temp_signals).astype(bool)

        temp_signals = np.zeros(len(feed))
        temp_signals[short_trades.exit_idx] = True
        short_exits = pd.Series(index=feed[feed.time_column], data=temp_signals).astype(bool)

        fig = vbt.make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        spread = feed.set_index(feed.time_column).close
        zscore = feed.zscore()

        spread.vbt.plot(add_trace_kwargs=dict(row=1, col=1), fig=fig, title=feed._underlying_info)
        zscore.vbt.plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)

        # Plot entry and exit markers on z-score
        short_entries.vbt.signals.plot_as_exit_markers(
            zscore,
            add_trace_kwargs=dict(row=2, col=1),
            trace_kwargs=dict(marker=dict(opacity=0.4, size=12, color="green"), name="short_entry"),
            fig=fig,
        )
        short_exits.vbt.signals.plot_as_entry_markers(
            zscore,
            add_trace_kwargs=dict(row=2, col=1),
            trace_kwargs=dict(marker=dict(opacity=0.4, size=12, color="red"), name="short_exit"),
            fig=fig,
        )
        long_entries.vbt.signals.plot_as_entry_markers(
            zscore,
            add_trace_kwargs=dict(row=2, col=1),
            trace_kwargs=dict(marker=dict(opacity=0.8), name="long_entry"),
            fig=fig,
        )
        long_exits.vbt.signals.plot_as_exit_markers(
            zscore,
            add_trace_kwargs=dict(row=2, col=1),
            trace_kwargs=dict(marker=dict(opacity=0.8), name="long_exit"),
            fig=fig,
        )

        # Plot individual close prices
        feed.underlying_col("close").droplevel(1, axis=1).rename(
            columns=lambda c: c.split(".")[1] + " close price"
        ).vbt.plot(add_trace_kwargs=dict(row=3, col=1), fig=fig)

        # Plot daily returns
        (portfolio.returns() * 100).rename("% returns").vbt.scatterplot(add_trace_kwargs=dict(row=4, col=1), fig=fig)

        # Plot order entry and exit prices
        orders = portfolio.orders.records_readable
        if len(orders) > 0:
            orders["Side"] = orders["Side"].replace({"Sell": -1, "Buy": 1})
            orders["Price"] = orders["Price"] * orders["Side"]
            orders["When"] = pd.concat(
                [pd.Series(["entry", "exit"]).repeat(2)] * int(np.ceil(len(orders) / 4)), ignore_index=True
            ).values[: len(orders)]
            orders = orders.set_index(["Timestamp", "Column", "When"])
            orders = orders.unstack(["Column", "When"]).Price.rename(
                columns=lambda c: c.split(".")[1] + " fill price", level=0
            )
            orders.vbt.scatterplot(add_trace_kwargs=dict(row=5, col=1), fig=fig)

        # Plot daily PnL
        self.portfolio.trades.records_readable.sort_values("Entry Timestamp").groupby(
            "Entry Timestamp"
        ).PnL.sum().reindex(self.feed.open_time).fillna(0).rename("PnL").vbt.scatterplot(
            add_trace_kwargs=dict(row=6, col=1), fig=fig
        )

        # Plot cumulative returns
        (portfolio.cumulative_returns() * 100).rename("cumulative returns").vbt.plot(
            add_trace_kwargs=dict(row=7, col=1), fig=fig
        )

        # Plot slippage
        self.slippage().vbt.scatterplot(add_trace_kwargs=dict(row=8, col=1), fig=fig)

        fig.update_layout(height=1200, width=1800, hovermode="x unified", hoverlabel={"namelength": -1}, legend=None)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y2",
            x0=0,
            y0=1,
            x1=1,
            y1=-1,
            fillcolor="gray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

        return fig

    def analyze(self):
        """Display plots, session stats, trades, orders, underlying prices"""
        display(self.plot())
        display(f"slippage: {self.slippage().sum().sum()}")
        display(self.aggregate_stats())
        display(self.portfolio.stats())
        display(self.portfolio.trades.records_readable.sort_values("Entry Timestamp").head(10))
        display(self.portfolio.orders.records_readable.sort_values("Timestamp").head(20))
        display(self.feed.underlyings)

    def get_leg_trades(self) -> pd.DataFrame:
        """Return DataFrame with additional stats per leg for each trade"""
        portfolio = self.portfolio
        orders = portfolio.orders.records_readable.sort_values("Timestamp")
        orders["col"] = portfolio.orders.records.sort_values("idx").col
        try:
            orders["When"] = pd.concat(
                [pd.Series(["entry", "exit"]).repeat(2)] * int(np.ceil(len(orders) / 4)), ignore_index=True
            ).values[: len(orders)]
        except ValueError:
            orders["When"] = pd.Series()
        orders = orders.rename(columns={"Price": "filled_price", "Size": "size"})

        close_prices = (
            self.feed.underlying_col("close")
            .droplevel(1, axis=1)
            .set_axis([0, 1], axis=1)
            .melt(var_name="col", value_name="close", ignore_index=False)
            .reset_index()
            .rename(columns={"open_time": "Timestamp"})
        )
        slippage = orders.merge(close_prices)

        slippage["slippage"] = abs(slippage["close"] - slippage["filled_price"]) * slippage["size"]
        slippage = slippage[["Timestamp", "col", "slippage", "When"]]
        entry_slippage = (
            slippage[slippage.When == "entry"]
            .rename(columns={"slippage": "entry_slippage", "Timestamp": "Entry Timestamp"})
            .drop(columns="When")
        )
        exit_slippage = (
            slippage[slippage.When == "exit"]
            .rename(columns={"slippage": "exit_slippage", "Timestamp": "Exit Timestamp"})
            .drop(columns="When")
        )

        leg_trades = portfolio.trades.records_readable.sort_values("Entry Timestamp")
        leg_trades["col"] = portfolio.trades.records.sort_values("entry_idx").col
        leg_trades = leg_trades.merge(
            self.feed.zscore(period=self.feed.zscore_period).rename("entry_zscore"),
            left_on="Entry Timestamp",
            right_index=True,
        )
        leg_trades = leg_trades.merge(
            self.feed.zscore(period=self.feed.zscore_period).rename("exit_zscore"),
            left_on="Exit Timestamp",
            right_index=True,
        )
        leg_trades = leg_trades.merge(
            portfolio.cash().rename("entry_cash"), left_on="Entry Timestamp", right_index=True
        )
        leg_trades = leg_trades.merge(portfolio.cash().rename("exit_cash"), left_on="Exit Timestamp", right_index=True)
        leg_trades = leg_trades.merge(entry_slippage)
        leg_trades = leg_trades.merge(exit_slippage, how="left")  # Left merge because last trade may not be closed
        leg_trades = leg_trades.merge(close_prices.rename(columns={"Timestamp": "Entry Timestamp"})).rename(
            columns={"close": "entry_price"}
        )
        leg_trades = leg_trades.merge(close_prices.rename(columns={"Timestamp": "Exit Timestamp"})).rename(
            columns={"close": "exit_price"}
        )

        leg_trades["slippage"] = leg_trades["entry_slippage"] + leg_trades["exit_slippage"]
        leg_trades["fees"] = leg_trades["Entry Fees"] + leg_trades["Exit Fees"]
        leg_trades.rename(columns={"PnL": "PnL w fees"}, inplace=True)
        leg_trades["raw PnL"] = leg_trades["PnL w fees"] + leg_trades["fees"]
        leg_trades["PnL w slip"] = leg_trades["raw PnL"] - leg_trades["slippage"]
        leg_trades["PnL w slip + fees"] = leg_trades["PnL w fees"] - leg_trades["slippage"]

        return leg_trades

    @staticmethod
    def group_trades(g: pd.DataFrame, sum_legs=False) -> pd.DataFrame:
        """Apply method for aggregating leg trades grouped by Entry Timestamp"""
        out = pd.DataFrame()
        entry_time = g["Entry Timestamp"].iloc[0]
        exit_time = g["Exit Timestamp"].iloc[0]
        names = g.Column.to_list()
        status = g.Status.to_list()
        directions = g.Direction.replace("Short", -1).replace("Long", 1).to_list()
        spread_dir = 1 if directions[0] == -1 else -1

        if (g.Status == "Closed").all():
            overall_status = "Closed"
        elif (g.Status == "Open").any():
            overall_status = "Open"
        elif (g.status == "Rejected").any():
            overall_status = "Rejected"
        else:
            overall_status = f"{g.Status.iloc[0]} / {g.Status.iloc[1]}"

        entry_fees = g["Entry Fees"].to_list()
        exit_fees = g["Exit Fees"].to_list()
        total_fees = (g["Entry Fees"] + g["Exit Fees"]).to_list()

        entry_slippage = g["entry_slippage"].to_list()
        exit_slippage = g["exit_slippage"].to_list()
        total_slippage = (g["entry_slippage"] + g["exit_slippage"]).to_list()

        raw_pnl = g["raw PnL"].to_list()
        pnl_w_fees = g["PnL w fees"].to_list()
        pnl_w_slip = g["PnL w slip"].to_list()
        pnl_w_slip_fees = g["PnL w slip + fees"].to_list()

        entry_prices = g["entry_price"].to_list()
        exit_prices = g["exit_price"].to_list()
        entry_zscore = g["entry_zscore"].iloc[0]
        exit_zscore = g["exit_zscore"].iloc[0]
        filled_entry_prices = g["Avg Entry Price"].to_list()
        filled_exit_prices = g["Avg Exit Price"].to_list()
        sizes = g["Size"].to_list()
        entry_spread = entry_prices[1] - entry_prices[0]
        exit_spread = exit_prices[1] - exit_prices[0]
        filled_entry_spread = filled_entry_prices[1] - filled_entry_prices[0]
        filled_exit_spread = filled_exit_prices[1] - filled_exit_prices[0]

        spread_change = (exit_spread - entry_spread) * spread_dir
        filled_spread_change = (filled_exit_spread - filled_entry_spread) * spread_dir

        start_cash = g["entry_cash"].iloc[0] + sum(entry_fees) + sum(entry_slippage)
        end_cash = g["exit_cash"].iloc[0]

        def add_wide_column(df: pd.DataFrame, name: str, values: list):
            if sum_legs:
                try:
                    return df.assign(**{name: sum(values)})
                except TypeError:
                    pass
            return df.assign(**{name: [values]})

        out = add_wide_column(out, "names", names)
        out = add_wide_column(out, "status", status)
        out = add_wide_column(out, "directions", directions)
        out = add_wide_column(out, "entry_fees", entry_fees)
        out = add_wide_column(out, "exit_fees", exit_fees)
        out = add_wide_column(out, "total_fees", total_fees)
        out = add_wide_column(out, "entry_slippage", entry_slippage)
        out = add_wide_column(out, "exit_slippage", exit_slippage)
        out = add_wide_column(out, "total_slippage", total_slippage)
        out = add_wide_column(out, "raw_pnl", raw_pnl)
        out = add_wide_column(out, "pnl_w_fees", pnl_w_fees)
        out = add_wide_column(out, "pnl_w_slip", pnl_w_slip)
        out = add_wide_column(out, "pnl_w_slip_fees", pnl_w_slip_fees)
        out = add_wide_column(out, "entry_prices", entry_prices)
        out = add_wide_column(out, "exit_prices", exit_prices)
        out = add_wide_column(out, "filled_entry_prices", filled_entry_prices)
        out = add_wide_column(out, "filled_exit_prices", filled_exit_prices)
        out = add_wide_column(out, "sizes", sizes)

        out["entry_time"] = entry_time
        out["exit_time"] = exit_time
        out["entry_zscore"] = entry_zscore
        out["exit_zscore"] = exit_zscore
        out["spread_dir"] = spread_dir
        out["overall_status"] = overall_status
        out["entry_spread"] = entry_spread
        out["exit_spread"] = exit_spread
        out["filled_entry_spread"] = filled_entry_spread
        out["filled_exit_spread"] = filled_exit_spread
        out["spread_change"] = spread_change
        out["filled_spread_change"] = filled_spread_change
        out["start_cash"] = start_cash
        out["end_cash"] = end_cash
        return out

    @staticmethod
    def format_trade(t: pd.Series) -> str:
        """Format trade string using an aggregated trade created by `self.group_trades`"""
        exchanges = [x.split(".")[1] for x in t.names]
        symbol = t.names[0].split(".")[-1]
        starttime = t.entry_time.strftime("%Y-%m-%d")
        endtime = t.exit_time.strftime("%Y-%m-%d")

        # Rounding factors
        spread_rf = 4 if (abs(t.entry_spread) < 1) else 2
        price_rf = 4 if (sum([abs(x) for x in t.entry_prices]) / len(t.entry_prices)) < 1 else 2
        size_rf = 3 if (sum(t.sizes) / len(t.sizes)) < 1 else 2

        # direction symbols
        d_sym = {-1: "\\", 1: "/"}

        # direction strings
        d_str = {-1: "Short", 1: "Long"}

        entry_spread = t.entry_spread
        filled_entry_spread = t.filled_entry_spread
        entry_prices = [(x) for x in t.entry_prices]
        filled_entry_prices = [(x) for x in t.filled_entry_prices]

        entry_zscore = t.entry_zscore
        exit_zscore = t.exit_zscore

        exit_spread = t.exit_spread
        filled_exit_spread = t.filled_exit_spread
        exit_prices = [(x) for x in t.exit_prices]
        filled_exit_prices = [(x) for x in t.filled_exit_prices]

        spread_change = t.spread_change
        spread_change_pct = 100 * spread_change / abs(entry_spread)
        filled_spread_change = t.filled_spread_change
        filled_spread_change_pct = 100 * filled_spread_change / abs(filled_entry_spread)

        trade_value = sum(size * price for size, price in zip(t.sizes, t.entry_prices))
        sizes = [(x) for x in t.sizes]
        raw_pnl = sum(t.raw_pnl)
        raw_returns = 100 * sum(t.raw_pnl) / trade_value
        pnl_w_slip_fees = sum(t.pnl_w_slip_fees)
        returns_w_slip_fees = 100 * sum(t.pnl_w_slip_fees) / trade_value

        slippage = sum(t.entry_slippage) + sum(t.exit_slippage)
        slippage_pct = 100 * slippage / trade_value
        entry_slippage = sum(t.entry_slippage)
        exit_slippage = sum(t.exit_slippage)

        fees = sum(t.total_fees)
        fee_pct = 100 * fees / trade_value
        entry_fees = sum(t.entry_fees)
        exit_fees = sum(t.exit_fees)

        formatter = BashFormatter()

        def add_color(x: str, x_float: float):
            if x_float > 0:
                return formatter.format(x, "green", "black", "bold")
            elif x_float < 0:
                return formatter.format(x, "red", "black", "bold")
            else:
                return formatter.format(x, "orange", "black", "bold")

        def add_secondary_color(x: str, x_float: float):
            if x_float > 0:
                return formatter.color(x, "light_green")
            elif x_float < 0:
                return formatter.color(x, "light_red")
            else:
                return formatter.color(x, "light_orange")

        margin = 25
        trade_info = f"""
        {'-'*130}
        {'Info:':{margin}} {d_str[t.spread_dir]} {d_sym[t.spread_dir]} | {symbol} | ({exchanges[1]} - {exchanges[0]}) | {t.overall_status} | {starttime} --> {endtime} ({(t.exit_time - t.entry_time).days} days)
        {'Cash:':{margin}} {f'Start[ {t.start_cash:7.2f}]':20} End[ {t.end_cash:7.2f}]
        {'Entry Spread:':{margin}} {f'Inital[ {entry_spread: .{spread_rf}f} ({entry_prices[1]: .{spread_rf}f} - {entry_prices[0]: .{spread_rf}f})]':40} Filled[ {filled_entry_spread: .{spread_rf}f} ({filled_entry_prices[1]: .{spread_rf}f} - {filled_entry_prices[0]: .{spread_rf}f})]
        {'Exit Spread:':{margin}} {f'Inital[ {exit_spread: .{spread_rf}f} ({exit_prices[1]: .{spread_rf}f} - {exit_prices[0]: .{spread_rf}f})]':40} Filled[ {filled_exit_spread: .{spread_rf}f} ({filled_exit_prices[1]: .{spread_rf}f} - {filled_exit_prices[0]: .{spread_rf}f})]
        {'Spread change:':{margin}} {add_secondary_color(f'Inital[ {spread_change: .{spread_rf}f} ({spread_change_pct:.2f}%)]', spread_change):40} {add_secondary_color(f'Filled[ {filled_spread_change:.{spread_rf}f} ({filled_spread_change_pct:.2f}%)]', filled_spread_change)}
        {'Trade Value:':{margin}} {trade_value:.2f} ({sizes[0]:.{size_rf}f}x{filled_entry_prices[0]:.{price_rf}f} + {sizes[1]:.{size_rf}f}x{filled_entry_prices[1]:.{price_rf}f})
        {'Raw PnL:':{margin}} {add_secondary_color(f'{raw_pnl:8.2f} ({raw_returns: .2f}%)', raw_pnl)}
        {'PnL (w/ slip & fees):':{margin}} {add_color(f'{pnl_w_slip_fees:8.2f} ({returns_w_slip_fees: .2f}%)', pnl_w_slip_fees)}
        {'Z Score:':{margin}} {f'Entry[ {entry_zscore:7.2f}]':20} Exit[ {exit_zscore:7.2f}]
        {'Slippage:':{margin}} {slippage:8.2f} ({slippage_pct:.2f}%) {f'Entry[ {entry_slippage:7.2f}]':20} Exit[ {exit_slippage:7.2f}]
        {'Fees:':{margin}} {fees:8.2f} ({fee_pct:.2f}%) {f'Entry[ {entry_fees:7.2f}]':20} Exit[ {exit_fees:7.2f}]
        
        {'Exchange:':{margin}} {exchanges[0]:>10}   {exchanges[1]:>10}
        {'Entry Price: ':{margin}} {f'{entry_prices[0]:.{price_rf}f}':>10}   {f'{entry_prices[1]:.{price_rf}f}':>10}
        {'Entry Price (filled): ':{margin}} {f'{filled_entry_prices[0]:.{price_rf}f}':>10}   {f'{filled_entry_prices[1]:.{price_rf}f}':>10}
        {'Exit Price: ':{margin}} {f'{exit_prices[0]:.{price_rf}f}':>10}   {f'{exit_prices[1]:.{price_rf}f}':>10}
        {'Exit Price (filled): ':{margin}} {f'{filled_exit_prices[0]:.{price_rf}f}':>10}   {f'{filled_exit_prices[1]:.{price_rf}f}':>10}
        {'Trade Size: ':{margin}} {f'{sizes[0]:.{size_rf}f}':>10}   {f'{sizes[1]:.{size_rf}f}':>10}
        {'Trade Value: ':{margin}} {f'{sizes[0] * filled_entry_prices[0]:.{price_rf}f}':>10}   {f'{sizes[1] * filled_entry_prices[1]:.{price_rf}f}':>10}
        {'Raw PnL: ':{margin}} {add_secondary_color(f'{t.raw_pnl[0]:.2f}', t.raw_pnl[0]):>10}   {add_secondary_color(f'{t.raw_pnl[1]:.2f}', t.raw_pnl[1]):>10}
        {'slippage: ':{margin}} {f'{t.entry_slippage[0] + t.exit_slippage[0]:.2f}':>10}   {f'{t.entry_slippage[1] + t.exit_slippage[1]:.2f}':>10}
        {'fees:':{margin}} {f'{t.total_fees[0]:.2f}':>10}   {f'{t.total_fees[1]:.2f}':>10}
        {'PnL (w/ slip only):':{margin}} {f'{t.pnl_w_slip[0]:.2f}':>10}   {f'{t.pnl_w_slip[1]:.2f}':>10}
        {'PnL (w/ fees only):':{margin}} {f'{t.pnl_w_fees[0]:.2f}':>10}   {f'{t.pnl_w_fees[1]:.2f}':>10}
        {'PnL (w/ slip & fees):':{margin}} {f'{t.pnl_w_slip_fees[0]:.2f}':>10}   {f'{t.pnl_w_slip_fees[1]:.2f}':>10}
        {'-'*130}
        """

        return trade_info

    def print_all_trades(self):
        """Print formatted trades to stdout"""
        leg_trades = self.get_leg_trades()
        if len(leg_trades) == 0:
            print("No trades to show")
        else:
            grouped_trades = (
                leg_trades.sort_values(["Entry Timestamp", "col"])
                .groupby("Entry Timestamp", as_index=False)
                .apply(self.group_trades)
                .reset_index(level=1, drop=True)
            )
            for idx, trade in grouped_trades.iterrows():
                print(self.format_trade(trade))

    def aggregate_stats(self):
        stats = pd.Series()
        leg_trades = self.get_leg_trades()
        if len(leg_trades) == 0:
            stats["avg_fees"] = np.nan
            stats["avg_slippage"] = np.nan
            stats["avg_pnl"] = np.nan
            stats["profitable_trades"] = 0
            stats["losing_trades"] = 0
            stats["total_trades"] = 0
            return stats

        grouped_trades = (
            leg_trades.sort_values(["Entry Timestamp", "col"])
            .groupby("Entry Timestamp", as_index=False)
            .apply(self.group_trades, sum_legs=True)
            .reset_index(level=1, drop=True)
        )
        stats["avg_fees"] = grouped_trades["total_fees"].mean()
        stats["avg_slippage"] = grouped_trades["total_slippage"].mean()
        stats["avg_pnl"] = grouped_trades["pnl_w_slip_fees"].mean()
        stats["profitable_trades"] = (grouped_trades["pnl_w_slip_fees"] > 0).sum()
        stats["losing_trades"] = (grouped_trades["pnl_w_slip_fees"] < 0).sum()
        stats["total_trades"] = len(grouped_trades[grouped_trades.overall_status == "Closed"])
        return stats
