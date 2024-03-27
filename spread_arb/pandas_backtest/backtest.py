import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import cryptomart as cm

from ..data_prep import create_exchange_dataframe, get_cryptomart_data_iterator
from ..errors import NotSupportedError
from ..util import cached
from .wrappers import LazyDataFrameHolder, LazySpreadHolder, Trades


def log_function_call(message):
    """Print `message` before and after calling the decorated function"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            print(message)
            ret = func(*args, **kwargs)
            print(message + " done")
            return ret

        return wrapper

    return decorator


def get_freq_str(td: pd.Timedelta) -> str:
    """Returns pd.Timedelta as a string. i.e. (15m, 1d, 4h, 3s, etc)"""
    return [str(v) + k[0] for k, v in td.components._asdict().items() if v != 0][0]


class DataPipeline:
    def __init__(
        self,
        base_data_path="data",
        frequency=pd.Timedelta(seconds=5),
        start_time="2023-04-10",
        end_time="2023-05-10",
        z_score_period=pd.Timedelta(days=5),
    ):
        self.base_data_path = base_data_path
        self.frequency = frequency
        self.start_time = start_time
        self.end_time = end_time
        self.z_score_period = z_score_period
        self.cm_client = cm.Client(quiet=True)

    @log_function_call("Aggregating tick quotes")
    def aggregate_quotes(
        self,
        group_threshold=None,
        parallel=False,
        overwrite_existing=False,
        workers=4,
        keep_symbols=None,
    ):
        """Aggregates tick quotes to a given frequency and saves them to disk"""

        def wrapper(df):
            if df.empty:
                return pd.Series(
                    {
                        "bid_price": np.nan,
                        "bid_amount": np.nan,
                        "ask_price": np.nan,
                        "ask_amount": np.nan,
                    },
                )

            if group_threshold is not None:
                start = df.name
                threshold = df[df.timestamp <= start + group_threshold]
                if threshold.empty:
                    return None
                df = threshold

            return pd.Series(
                {
                    "bid_price": df.bid_price.iloc[0],
                    "bid_amount": df.bid_amount.max(),
                    "ask_price": df.ask_price.iloc[0],
                    "ask_amount": df.ask_amount.max(),
                },
            )

        freq_str = get_freq_str(self.frequency)
        in_path = os.path.join(self.base_data_path, "tick_quotes")
        out_path = os.path.join(self.base_data_path, f"tick_quotes_{freq_str}_agg")

        def single_job(*args):
            args = args[0] if len(args) == 1 else args
            exchange, symbol, filepath = args
            outpath = os.path.join(out_path, exchange, symbol + ".parquet")
            if not overwrite_existing:
                if os.path.exists(outpath):
                    return
            df = pd.read_parquet(filepath)
            df = df[df.timestamp >= self.start_time]

            grouper = pd.Grouper(key="timestamp", freq=self.frequency)
            df = df.groupby(grouper).progress_apply(wrapper)
            df = df.reindex(
                pd.date_range(start=self.start_time, end=self.end_time, freq=self.frequency, name="timestamp")[:-1]
            ).reset_index()
            df["filled"] = df.bid_price.isna().astype(int)
            df = df.ffill().bfill()
            os.makedirs(os.path.join(out_path, exchange), exist_ok=True)
            df.to_parquet(outpath)

        if parallel:
            params = list(get_cryptomart_data_iterator(in_path, keep_symbols))
            with ThreadPoolExecutor(workers) as executor:
                progress_bar = tqdm(total=len(params))

                for _ in executor.map(single_job, params):
                    progress_bar.update(1)

                progress_bar.close()
        else:
            for exchange, symbol, filepath in get_cryptomart_data_iterator(in_path, keep_symbols, show_progress=True):
                single_job(exchange, symbol, filepath)

    @log_function_call("Computing aggregated quote stats")
    def get_agg_quote_stats(self):
        """Returns a DataFrame containing stats about the aggregated quotes for the given frequency"""
        freq_str = get_freq_str(self.frequency)
        in_path = os.path.join(self.base_data_path, f"tick_quotes_{freq_str}_agg")

        quotes_stats = create_exchange_dataframe()
        for exchange, symbol, filepath in get_cryptomart_data_iterator(in_path, show_progress=False):
            df = pd.read_parquet(filepath)
            quotes_stats.at[(exchange, symbol), "earliest_time"] = df.timestamp.min()
            quotes_stats.at[(exchange, symbol), "latest_time"] = df.timestamp.max()
            quotes_stats.at[(exchange, symbol), "total_rows"] = df.shape[0]
            quotes_stats.at[(exchange, symbol), "invalid_rows"] = df.filled.sum()
            quotes_stats.at[(exchange, symbol), "valid_rows"] = (1 - df.filled).sum()

        return quotes_stats.sort_values("invalid_rows")

    @log_function_call("Getting daily close")
    def get_daily_close(self):
        """Pulls daily close data from the cryptomart API and saves it to disk"""
        freq_str = get_freq_str(self.frequency)
        in_path = os.path.join(self.base_data_path, f"tick_quotes_{freq_str}_agg")
        out_path = os.path.join(self.base_data_path, f"daily_close")

        for exchange, symbol, filepath in get_cryptomart_data_iterator(in_path, show_progress=False):
            outfile = os.path.join(out_path, exchange, symbol + ".parquet")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            try:
                ohlcv = self.cm_client.ohlcv(
                    exchange, symbol, "perpetual", self.start_time, self.end_time, "interval_1d"
                )
            except NotSupportedError:
                continue
            daily_close = ohlcv.set_index("open_time")[["close"]]
            daily_close["filled"] = daily_close.close.isna().astype(int)
            daily_close = daily_close.ffill().bfill()
            daily_close.to_parquet(outfile)

    @log_function_call("Computing daily close stats")
    def get_daily_close_stats(self):
        """Returns a DataFrame containing stats about the daily close data for the given frequency"""
        in_path = os.path.join(self.base_data_path, f"daily_close")

        daily_close_stats = create_exchange_dataframe()
        for exchange, symbol, filepath in get_cryptomart_data_iterator(in_path, show_progress=False):
            df = pd.read_parquet(filepath)
            daily_close_stats.at[(exchange, symbol), "earliest_time"] = df.index.min()
            daily_close_stats.at[(exchange, symbol), "latest_time"] = df.index.max()
            daily_close_stats.at[(exchange, symbol), "total_rows"] = df.shape[0]
            daily_close_stats.at[(exchange, symbol), "invalid_rows"] = df.filled.sum()
            daily_close_stats.at[(exchange, symbol), "valid_rows"] = (1 - df.filled).sum()

        return daily_close_stats.sort_values("invalid_rows")

    @log_function_call("Creating price signals")
    def create_price_signals(self, keep_symbols=None):
        """Creates price signals from the aggregated quotes and daily close data and saves them to disk"""
        freq_str = get_freq_str(self.frequency)
        quotes_path = os.path.join(self.base_data_path, f"tick_quotes_{freq_str}_agg")
        daily_close_path = os.path.join(self.base_data_path, "daily_close")
        out_path = os.path.join(self.base_data_path, f"price_signal_{freq_str}")

        for (exchange, symbol, quotes_fn), (_, __, daily_close_fn) in zip(
            get_cryptomart_data_iterator(quotes_path, keep_symbols),
            get_cryptomart_data_iterator(daily_close_path, keep_symbols, show_progress=False),
        ):
            try:
                order_book_multi = (
                    self.cm_client._exchange_instance_map[exchange]
                    ._get_interface("order_book", "perpetual")
                    .multipliers[symbol]
                )
            except KeyError:
                continue
            quotes = pd.read_parquet(quotes_fn)
            quotes["ask_amount"] = quotes.ask_amount * order_book_multi
            quotes["bid_amount"] = quotes.bid_amount * order_book_multi
            daily_close = pd.read_parquet(daily_close_fn)
            quotes["price_signal"] = (quotes.ask_price + quotes.bid_price) / 2
            daily_close = (
                daily_close.reindex(
                    pd.date_range(daily_close.index.min(), quotes.timestamp.min(), freq=self.frequency)[:-1]
                )
                .ffill()
                .bfill()
                .rename(columns={"close": "price_signal"})
                .astype({"filled": int})
                .rename_axis("timestamp")
                .reset_index()
            )
            price_signal = pd.concat([daily_close, quotes])
            outfile = os.path.join(out_path, exchange, symbol + ".parquet")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            price_signal.to_parquet(outfile)

    def load_price_signals(self):
        """Loads the price signals from disk and returns them as a DataFrame"""
        freq_str = get_freq_str(self.frequency)
        in_path = os.path.join(self.base_data_path, f"price_signal_{freq_str}")
        price_signal_dfs = create_exchange_dataframe()
        for exchange, symbol, filepath in get_cryptomart_data_iterator(in_path, show_progress=False):
            price_signal_dfs.at[(exchange, symbol), "ldfh"] = LazyDataFrameHolder(
                filepath, name=f"{exchange}.{symbol}"
            )
        return price_signal_dfs

    def load_spreads(self, price_signals):
        """Loads the price signals and returns a DataFrame containing the spreads between each pair of exchanges"""
        ps_cross_product = (
            price_signals.reset_index()
            .merge(price_signals.reset_index(), how="cross", suffixes=("_x", "_y"))
            .pipe(lambda df: df[(df.exchange_x < df.exchange_y) & (df.symbol_x == df.symbol_y)])
            .drop(columns="symbol_y")
            .rename(columns={"symbol_x": "symbol"})
            .reset_index(drop=True)
        )[["symbol", "exchange_x", "exchange_y", "ldfh_x", "ldfh_y"]].set_index(["symbol", "exchange_x", "exchange_y"])

        freq = price_signals.iloc[0].ldfh.get().timestamp.diff().median()
        if freq < pd.Timedelta(seconds=30):
            min_trade_time = pd.Timedelta(seconds=30)
        else:
            min_trade_time = None

        spreads = ps_cross_product.apply(
            lambda row: LazySpreadHolder(
                row.ldfh_x,
                row.ldfh_y,
                name=f"{row.name[0]}.{row.name[1]}.{row.name[2]}",
                min_trade_time=min_trade_time,
                z_score_period=self.z_score_period,
            ),
            axis=1,
        )

        return spreads

    def run(self, keeps_file=None):
        """Runs the entire data pipeline for the spread arbitrage strategy

        1. Aggregate tick quotes + retrieve daily close for the given frequency, start and end times.
        2. Filters out symbols with too many invalid rows, unless `keeps_file` is provided in which case it uses the symbols in the file.
        3. Create the price signals from the aggregated quotes and daily close data.
        4. Load the spreads between each pair of exchanges.
        5. Get all trades for each spread.
        6. Get all trades summary.

        Returns the spreads DataFrame.
        """
        if keeps_file:
            with open(keeps_file, "rb") as f:
                keep_symbols = pickle.load(f)
        else:
            keep_symbols = None

        self.aggregate_quotes(
            group_threshold=None,
            parallel=False,
            overwrite_existing=False,
            workers=8,
            keep_symbols=keep_symbols,
        )
        self.get_daily_close()

        if not keeps_file:
            agg_quote_stats = self.get_agg_quote_stats()
            daily_close_stats = self.get_daily_close_stats()
            keep_symbols = (
                daily_close_stats[daily_close_stats.invalid_rows == 0]
                .join(
                    agg_quote_stats[agg_quote_stats.invalid_rows < (30 * (pd.Timedelta(hours=1) / self.frequency))],
                    lsuffix="_x",
                    rsuffix="_y",
                    how="inner",
                )
                .index
            )
        self.create_price_signals(keep_symbols)

        return self.load_spreads(self.load_price_signals())


class BacktestRunner:
    """Class for running the backtest pipeline on a DataFrame of spreads"""

    def __init__(self, spreads):
        self.spreads = spreads

    @log_function_call("Getting all trades_summary")
    @cached("/tmp/cache/get_all_trades_summary")
    def get_all_trades_summary(self, **cache_kwargs):
        """Returns a DataFrame containing aggregated trade stats for each spread"""
        return self.spreads.progress_apply(lambda x: x.get_trades_summary())

    @log_function_call("Getting all trades")
    @cached("/tmp/cache/get_all_trades")
    def get_all_trades(self, **cache_kwargs):
        """Returns a DataFrame containing all trades for each spread"""
        dfs = []
        for idx, row in tqdm(self.spreads.reset_index().iterrows(), total=len(self.spreads)):
            trades = row[0].get_trades()
            trades = trades.assign(symbol=row.symbol, exchange_x=row.exchange_x, exchange_y=row.exchange_y, lsh=row[0])
            dfs.append(trades)
        trades = pd.concat(dfs).sort_values(["entry_time", "exit_time"])
        trades = trades.groupby("entry_time", as_index=False).apply(
            lambda g: g[g.entry_zscore.abs() == g.entry_zscore.abs().max()]
        )
        trades = trades[trades.entry_time <= trades.shift(1).exit_time]
        return Trades(trades.reset_index(drop=True))

    def save_subset_of_symbols(self, criteria="avg_profit", n=20, name="keep_symbols.pkl"):
        """Saves the top `n` symbols based on the given `criteria` applied to the trades summary"""
        keep_symbols = self.get_all_trades_summary().sort_values(criteria).head(n).index
        keep_symbols = (
            keep_symbols.to_frame()
            .reset_index(drop=True)
            .melt(id_vars="symbol")
            .drop(columns="variable")
            .drop_duplicates()[["value", "symbol"]]
            .values
        )
        keep_symbols = [tuple(x) for x in keep_symbols]
        with open(os.path.join(self.base_data_path, name), "wb") as f:
            pickle.dump(keep_symbols, f)

    def get_chained_summary(self, all_trades):
        """Returns a Series containing aggregated stats for all trades"""
        return pd.Series(
            {
                "total_trades": all_trades.shape[0],
                "winning_trades": (all_trades.profit > 0).sum(),
                "losing_trades": (all_trades.profit < 0).sum(),
                "win_pct": (all_trades.profit > 0).mean(),
                "best_trade": all_trades.profit.max(),
                "worst_trade": all_trades.profit.min(),
                "avg_profit": all_trades.profit.mean(),
                "total_profit": all_trades.profit.sum(),
                "total_transacted_value": all_trades.transacted_value.sum(),
            }
        )

    def run(self):
        """Runs the entire backtest pipeline and returns a dictionary containing the results"""
        all_trades = self.get_all_trades()
        all_trades_summary = self.get_all_trades_summary()
        chained_summary = self.get_chained_summary(all_trades)

        return {
            "all_trades": all_trades,
            "all_trades_summary": all_trades_summary,
            "chained_summary": chained_summary,
        }
