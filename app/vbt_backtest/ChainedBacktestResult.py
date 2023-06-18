import pickle
from functools import cached_property

import app
import numpy as np
import pandas as pd
import pyutil
import vectorbt as vbt
from app import data_prep
from app.feeds import Spread
from IPython.display import display

from .. import data_prep
from . import BacktestResult


def alternating(a, b, n):
    return pd.Series([a, b] * (n // 2) + [a] * (n % 2)).values


class ChainedBacktestResult(BacktestResult):
    def __init__(self, portfolio: vbt.Portfolio, all_spreads: Spread, z_score_period: int = 30):
        self.portfolio = portfolio
        self.all_spreads = all_spreads
        self.z_score_period = z_score_period

    @cached_property
    def column_map(self) -> pd.DataFrame:
        """Returns DataFrame with schema: [Column, col]"""
        return pd.DataFrame(
            {"Column": self.portfolio.assets().columns, "col": range(len(self.portfolio.assets().columns))}
        )

    @cached_property
    def timestamp_map(self) -> pd.DataFrame:
        """Returns DataFrame with schema: [open_time, time_idx]"""
        return pd.DataFrame({"open_time": self.portfolio.cash().index, "time_idx": range(len(self.portfolio.cash()))})

    @cached_property
    def close_prices(self) -> pd.DataFrame:
        """Returns DataFrame with schema: [col, time_idx, close]"""
        df = pd.concat(
            [
                pickle.loads(self.all_spreads.iloc[i].spread)
                .underlying_col("close")
                .droplevel(1, axis=1)
                .set_axis([i, i + len(self.all_spreads)], axis=1)
                .melt(var_name="col", value_name="close", ignore_index=False)
                .reset_index()
                for i in range(len(self.all_spreads))
            ],
            axis=0,
        )
        df = df.merge(self.timestamp_map)
        df = df.drop(columns="open_time")
        df = df.sort_values(["col", "time_idx"])
        df = df.reset_index(drop=True)
        df = df[["col", "time_idx", "close"]]
        return df

    @cached_property
    def zscores(self) -> pd.DataFrame:
        """Returns DataFrame with schema [col, time_idx, zscore]"""
        df = pd.concat(
            [
                pickle.loads(self.all_spreads.iloc[i % len(self.all_spreads)].spread)
                .zscore(period=self.z_score_period)
                .to_frame()
                .assign(col=i)
                for i in range(2 * len(self.all_spreads))
            ],
            axis=0,
        )
        df = df.reset_index()
        df = df.merge(self.timestamp_map)
        df = df.drop(columns="open_time")
        df = df.sort_values(["col", "time_idx"])
        df = df.reset_index(drop=True)
        df = df[["col", "time_idx", "zscore"]]
        return df

    @cached_property
    def orders(self) -> pd.DataFrame:
        """Returns DataFrame with schema [id, col, idx, size, filled_price, fees, side, type, price, zscore, slippage]"""
        df = self.portfolio.orders.records.sort_values(["idx", "id"])
        df = df.groupby("col").apply(lambda df: df.assign(type=alternating("entry", "exit", len(df))))
        df = df.merge(self.close_prices, left_on=["col", "idx"], right_on=["col", "time_idx"])
        df = df.merge(self.zscores)
        df = df.drop(columns="time_idx")
        df = df.rename(columns={"price": "filled_price", "close": "price"})
        df["slippage"] = abs(df.filled_price - df.price) * df["size"]

        assert len(df) == len(self.portfolio.orders.records)

        return df

    @cached_property
    def leg_trades(self) -> pd.DataFrame:
        """Return DataFrame with additional stats per leg for each trade"""

        orders = self.orders.drop(columns=["id", "size", "side"])

        entry_orders = (
            orders[orders.type == "entry"]
            .drop(columns="type")
            .rename(columns=lambda c: "entry_" + c if c != "col" else c)
        )
        exit_orders = (
            orders[orders.type == "exit"]
            .drop(columns="type")
            .rename(columns=lambda c: "exit_" + c if c != "col" else c)
        )

        trades = self.portfolio.trades.records.drop(
            columns=["entry_price", "exit_price", "entry_fees", "exit_fees", "parent_id", "id"]
        )
        trades = trades.merge(entry_orders, on=["col", "entry_idx"], how="left")
        trades = trades.merge(exit_orders, on=["col", "exit_idx"], how="left")
        trades = trades.merge(self.timestamp_map.rename(columns={"open_time": "entry_time", "time_idx": "entry_idx"}))
        trades = trades.merge(self.timestamp_map.rename(columns={"open_time": "exit_time", "time_idx": "exit_idx"}))
        trades = trades.merge(self.column_map.rename(columns={"Column": "name"}))
        trades = trades.replace({"direction": {0: "long", 1: "short"}})
        trades = trades.replace({"status": {0: "open", 1: "closed"}})

        trades["slippage"] = trades["entry_slippage"] + trades["exit_slippage"]
        trades["fees"] = trades["entry_fees"] + trades["exit_fees"]

        common_columns = [c for c in trades.columns if not c.startswith("entry_") and not c.startswith("exit_")]
        entry_columns = [c for c in trades.columns if c.startswith("entry_")]
        exit_columns = [c for c in trades.columns if c.startswith("exit_")]

        trades = trades[common_columns + entry_columns + exit_columns]

        trades.rename(columns={"pnl": "pnl_w_fees"}, inplace=True)

        trades["raw_pnl"] = trades["pnl_w_fees"] + trades["fees"]
        trades["pnl_w_slip"] = trades["raw_pnl"] - trades["slippage"]
        trades["pnl_w_slip_fees"] = trades["pnl_w_fees"] - trades["slippage"]

        trades = trades.sort_values(["entry_idx", "col"])

        assert len(trades) == len(self.portfolio.trades.records)

        return trades

    @cached_property
    def trades(self) -> pd.DataFrame:
        trades = (
            self.leg_trades.groupby("entry_idx", as_index=False)
            .apply(self.group_trades)
            .reset_index(level=1, drop=True)
        )
        pnl_w_slip_fees = pd.Series(np.stack(trades["pnl_w_slip_fees"]).sum(axis=1))
        trades["start_cash"] = self.portfolio.init_cash + pnl_w_slip_fees.cumsum().shift(1).fillna(0)
        trades["end_cash"] = self.portfolio.init_cash + pnl_w_slip_fees.cumsum()
        return trades

    @cached_property
    def trades_summed(self) -> pd.DataFrame:
        trades = (
            self.leg_trades.groupby("entry_idx", as_index=False)
            .apply(self.group_trades, sum_legs=True)
            .reset_index(level=1, drop=True)
        )
        trades["start_cash"] = self.portfolio.init_cash + trades["pnl_w_slip_fees"].cumsum().shift(1).fillna(0)
        trades["end_cash"] = self.portfolio.init_cash + trades["pnl_w_slip_fees"].cumsum()
        return trades

    @staticmethod
    def group_trades(g: pd.DataFrame, sum_legs=False) -> pd.DataFrame:
        """Apply method for aggregating leg trades grouped by Entry Timestamp"""
        out = pd.DataFrame()
        entry_time = g["entry_time"].iloc[0]
        exit_time = g["exit_time"].iloc[0]
        names = g["name"].to_list()
        status = g.status.to_list()
        directions = g.direction.replace("short", -1).replace("long", 1).to_list()
        spread_dir = 1 if directions[0] == -1 else -1

        if (g.status == "closed").all():
            overall_status = "closed"
        elif (g.status == "open").any():
            overall_status = "open"
        else:
            overall_status = f"{g.status.iloc[0]} / {g.status.iloc[1]}"

        entry_fees = g["entry_fees"].to_list()
        exit_fees = g["exit_fees"].to_list()
        total_fees = (g["entry_fees"] + g["exit_fees"]).to_list()

        entry_slippage = g["entry_slippage"].to_list()
        exit_slippage = g["exit_slippage"].to_list()
        total_slippage = (g["entry_slippage"] + g["exit_slippage"]).to_list()

        raw_pnl = g["raw_pnl"].to_list()
        pnl_w_fees = g["pnl_w_fees"].to_list()
        pnl_w_slip = g["pnl_w_slip"].to_list()
        pnl_w_slip_fees = g["pnl_w_slip_fees"].to_list()

        entry_prices = g["entry_price"].to_list()
        exit_prices = g["exit_price"].to_list()
        entry_zscore = g["entry_zscore"].iloc[0]
        exit_zscore = g["exit_zscore"].iloc[0]
        entry_filled_prices = g["entry_filled_price"].to_list()
        exit_filled_prices = g["exit_filled_price"].to_list()
        sizes = g["size"].to_list()
        entry_spread = entry_prices[1] - entry_prices[0]
        exit_spread = exit_prices[1] - exit_prices[0]
        entry_filled_spread = entry_filled_prices[1] - entry_filled_prices[0]
        exit_filled_spread = exit_filled_prices[1] - exit_filled_prices[0]

        spread_change = (exit_spread - entry_spread) * spread_dir
        filled_spread_change = (exit_filled_spread - entry_filled_spread) * spread_dir

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
        out = add_wide_column(out, "entry_filled_prices", entry_filled_prices)
        out = add_wide_column(out, "exit_filled_prices", exit_filled_prices)
        out = add_wide_column(out, "sizes", sizes)

        out["entry_time"] = entry_time
        out["exit_time"] = exit_time
        out["entry_zscore"] = entry_zscore
        out["exit_zscore"] = exit_zscore
        out["spread_dir"] = spread_dir
        out["overall_status"] = overall_status
        out["entry_spread"] = entry_spread
        out["exit_spread"] = exit_spread
        out["entry_filled_spread"] = entry_filled_spread
        out["exit_filled_spread"] = exit_filled_spread
        out["spread_change"] = spread_change
        out["filled_spread_change"] = filled_spread_change
        return out

    @staticmethod
    def format_trade(t: pd.Series) -> str:
        """Format trade string using an aggregated trade created by `self.group_trades`"""
        exchanges = [x.split(".")[1] for x in t.names]
        symbol = t.names[0].split(".")[-1]
        starttime = t.entry_time.strftime("%Y-%m-%d %H:%M")
        endtime = t.exit_time.strftime("%Y-%m-%d %H:%M")

        # Rounding factors
        spread_rf = 4 if (abs(t.entry_spread) < 1) else 2
        price_rf = 4 if (sum([abs(x) for x in t.entry_prices]) / len(t.entry_prices)) < 1 else 2
        size_rf = 3 if (sum(t.sizes) / len(t.sizes)) < 1 else 2

        # direction symbols
        d_sym = {-1: "\\", 1: "/"}

        # direction strings
        d_str = {-1: "Short", 1: "Long"}

        entry_spread = t.entry_spread
        entry_filled_spread = t.entry_filled_spread
        entry_prices = [x for x in t.entry_prices]
        entry_filled_prices = [x for x in t.entry_filled_prices]

        entry_zscore = t.entry_zscore
        exit_zscore = t.exit_zscore

        exit_spread = t.exit_spread
        exit_filled_spread = t.exit_filled_spread
        exit_prices = [x for x in t.exit_prices]
        exit_filled_prices = [x for x in t.exit_filled_prices]

        spread_change = t.spread_change
        spread_change_pct = 100 * spread_change / abs(entry_spread)
        filled_spread_change = t.filled_spread_change
        filled_spread_change_pct = 100 * filled_spread_change / abs(entry_filled_spread)

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

        formatter = pyutil.io.BashFormatter()

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
        {'Info:':{margin}} {d_str[t.spread_dir]} {d_sym[t.spread_dir]} | {symbol} | ({exchanges[1]} - {exchanges[0]}) | {t.overall_status} | {starttime} --> {endtime} ({(t.exit_time - t.entry_time)})
        {'Cash:':{margin}} {f'Start[ {t.start_cash:7.2f}]':20} End[ {t.end_cash:7.2f}]
        {'Entry Spread:':{margin}} {f'Inital[ {entry_spread: .{spread_rf}f} ({entry_prices[1]: .{spread_rf}f} - {entry_prices[0]: .{spread_rf}f})]':40} Filled[ {entry_filled_spread: .{spread_rf}f} ({entry_filled_prices[1]: .{spread_rf}f} - {entry_filled_prices[0]: .{spread_rf}f})]
        {'Exit Spread:':{margin}} {f'Inital[ {exit_spread: .{spread_rf}f} ({exit_prices[1]: .{spread_rf}f} - {exit_prices[0]: .{spread_rf}f})]':40} Filled[ {exit_filled_spread: .{spread_rf}f} ({exit_filled_prices[1]: .{spread_rf}f} - {exit_filled_prices[0]: .{spread_rf}f})]
        {'Spread change:':{margin}} {add_secondary_color(f'Inital[ {spread_change: .{spread_rf}f} ({spread_change_pct:.2f}%)]', spread_change):40} {add_secondary_color(f'Filled[ {filled_spread_change:.{spread_rf}f} ({filled_spread_change_pct:.2f}%)]', filled_spread_change)}
        {'Trade Value:':{margin}} {trade_value:.2f} ({sizes[0]:.{size_rf}f}x{entry_filled_prices[0]:.{price_rf}f} + {sizes[1]:.{size_rf}f}x{entry_filled_prices[1]:.{price_rf}f})
        {'Raw PnL:':{margin}} {add_secondary_color(f'{raw_pnl:8.2f} ({raw_returns: .2f}%)', raw_pnl)}
        {'PnL (w/ slip & fees):':{margin}} {add_color(f'{pnl_w_slip_fees:8.2f} ({returns_w_slip_fees: .2f}%)', pnl_w_slip_fees)}
        {'Z Score:':{margin}} {f'Entry[ {entry_zscore:7.2f}]':20} Exit[ {exit_zscore:7.2f}]
        {'Slippage:':{margin}} {slippage:8.2f} ({slippage_pct:.2f}%) {f'Entry[ {entry_slippage:7.2f}]':20} Exit[ {exit_slippage:7.2f}]
        {'Fees:':{margin}} {fees:8.2f} ({fee_pct:.2f}%) {f'Entry[ {entry_fees:7.2f}]':20} Exit[ {exit_fees:7.2f}]
        
        {'Exchange:':{margin}} {exchanges[0]:>10}   {exchanges[1]:>10}
        {'Entry Price: ':{margin}} {f'{entry_prices[0]:.{price_rf}f}':>10}   {f'{entry_prices[1]:.{price_rf}f}':>10}
        {'Entry Price (filled): ':{margin}} {f'{entry_filled_prices[0]:.{price_rf}f}':>10}   {f'{entry_filled_prices[1]:.{price_rf}f}':>10}
        {'Exit Price: ':{margin}} {f'{exit_prices[0]:.{price_rf}f}':>10}   {f'{exit_prices[1]:.{price_rf}f}':>10}
        {'Exit Price (filled): ':{margin}} {f'{exit_filled_prices[0]:.{price_rf}f}':>10}   {f'{exit_filled_prices[1]:.{price_rf}f}':>10}
        {'Trade Size: ':{margin}} {f'{sizes[0]:.{size_rf}f}':>10}   {f'{sizes[1]:.{size_rf}f}':>10}
        {'Trade Value: ':{margin}} {f'{sizes[0] * entry_filled_prices[0]:.{price_rf}f}':>10}   {f'{sizes[1] * entry_filled_prices[1]:.{price_rf}f}':>10}
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
        if len(self.trades) == 0:
            print("No trades to show")
        else:
            for idx, trade in self.trades.iterrows():
                print(self.format_trade(trade))

    def aggregate_stats(self):
        stats = pd.Series()
        if len(self.leg_trades) == 0:
            stats["avg_fees"] = np.nan
            stats["avg_slippage"] = np.nan
            stats["avg_pnl"] = np.nan
            stats["profitable_trades"] = 0
            stats["losing_trades"] = 0
            stats["total_trades"] = 0
            return stats

        stats["avg_fees"] = self.trades_summed["total_fees"].mean()
        stats["avg_slippage"] = self.trades_summed["total_slippage"].mean()
        stats["avg_pnl"] = self.trades_summed["pnl_w_slip_fees"].mean()
        stats["profitable_trades"] = (self.trades_summed["pnl_w_slip_fees"] > 0).sum()
        stats["losing_trades"] = (self.trades_summed["pnl_w_slip_fees"] < 0).sum()
        stats["total_trades"] = len(self.trades_summed[self.trades_summed.overall_status == "closed"])

        stats["longest_trade"] = (self.trades_summed["exit_time"] - self.trades_summed["entry_time"]).max()
        stats["shortest_trade"] = (self.trades_summed["exit_time"] - self.trades_summed["entry_time"]).min()
        stats["average_trade_length"] = (self.trades_summed["exit_time"] - self.trades_summed["entry_time"]).mean()

        stats["end_cash"] = self.trades_summed["end_cash"].dropna().iloc[-1]
        stats["return"] = 100 * (stats["end_cash"] - self.portfolio.init_cash) / self.portfolio.init_cash
        stats["avg_trade_return"] = stats["return"] / stats["total_trades"]
        stats["return_per_timedelta"] = stats["return"] / (len(self.timestamp_map) - self.z_score_period)

        stats["best_trade"] = pickle.dumps(self.trades.iloc[self.trades_summed["pnl_w_slip_fees"].argmax()])
        stats["worst_trade"] = pickle.dumps(self.trades.iloc[self.trades_summed["pnl_w_slip_fees"].argmin()])

        return stats

    def batch_run(self, slippages=[1e-9, 1e-4, 5e-4, 1e-3, 5e-3, 7e-3, 8e-3], identifier="spreads-arb-v2"):
        results = []
        for slippage in slippages:
            ohlcvs = data_prep.all_ohlcv(
                "2022-02-01", "2023-05-04", "interval_1h", refresh=False, identifiers=[identifier]
            )
            ohlcvs = ohlcvs[ohlcvs.missing_rows <= 0]
            ba_spreads = data_prep.dummy_bid_ask_spreads(ohlcvs, slippage, force_default=True)
            fee_info = data_prep.get_fee_info(refresh=False, identifiers=[identifier])
            spreads = data_prep.create_spreads(ohlcvs, fee_info, bas=ba_spreads)

            res = app.BacktestRunner(
                log_dir=None,
                use_slippage=True,
                use_funding_rate=False,
                profitable_only=True,
                z_score_thresholds=(0, 1),
                z_score_period=500,
            ).run_chained(spreads)

            res = res.aggregate_stats()
            res["slippage"] = slippage

            results.append(res)

        stats = pd.concat(results, axis=1).T
        start_columns = ["slippage", "end_cash", "return", "avg_trade_return", "return_per_timedelta"]
        end_columns = ["best_trade", "worst_trade"]
        stats = stats.reindex(
            start_columns + list(set(stats.columns) - set(start_columns) - set(end_columns)) + end_columns, axis=1
        )
        return stats

    def analyze(self):
        """Display plots, session stats, trades, orders, underlying prices"""
        display(self.aggregate_stats())
        display(self.portfolio.stats())
        display(self.portfolio.trades.records_readable.sort_values("Entry Timestamp").head(10))
        display(self.portfolio.orders.records_readable.sort_values("Timestamp").head(20))
