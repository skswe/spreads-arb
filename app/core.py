import datetime
import logging
import os
from collections import namedtuple
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import cryptomart
import numpy as np
import pandas as pd
import pyutil
import vectorbt as vbt
from cryptomart.enums import Exchange, Instrument, InstrumentType, Interval, Symbol
from IPython.display import display
from pyutil.io import redirect_stdout

from .feeds import Spread
from .globals import BLACKLISTED_SYMBOLS
from .vbt_backtest import from_order_func_wrapper

logger = logging.getLogger(__name__)


class BtResults:
    def __init__(self, feed: Spread, portfolio: vbt.Portfolio):
        self.feed = feed
        self.portfolio = portfolio


def spread_details(spreads: List[Spread]):
    """Format list of Spreads as a dataframe with additional info"""
    return pd.DataFrame(
        {
            "exchange_a": map(lambda e: e.ohlcv_a.exchange_name, spreads),
            "exchange_b": map(lambda e: e.ohlcv_b.exchange_name, spreads),
            "symbol": map(lambda e: e.ohlcv_b.symbol, spreads),
            "instType": map(lambda e: e.ohlcv_a.instType, spreads),
            "volatility": map(lambda e: e.volatility, spreads),
            "missing_rows": map(lambda e: len(e.missing_rows), spreads),
            "total_rows": map(lambda e: len(e), spreads),
            "earliest_time": map(lambda e: e.earliest_time, spreads),
            "latest_time": map(lambda e: e.latest_time, spreads),
            "gaps": map(lambda e: e.gaps, spreads),
            "alias": map(
                lambda e: f"{e.ohlcv_a.exchange_name}_{e.ohlcv_a.instType}_{e.ohlcv_b.exchange_name}_{e.ohlcv_b.instType}_{e.ohlcv_b.symbol}",
                spreads,
            ),
        }
    )


def plot_spread_and_zscore(result: BtResults):
    portfolio = result.portfolio
    feed = result.feed

    df = portfolio.trades.records.sort_values(["entry_idx", "col"])
    # For column 0, a short means we are long on the spread and vice versa
    df = df[df.col == 0]

    # direction 1 = short
    # direction 0 = long
    long_trades = df[df.direction == 1]
    short_trades = df[df.direction == 0]

    temp_signals = np.zeros(len(feed))
    temp_signals[long_trades.entry_idx] = True
    long_entries = pd.Series(index=feed.index, data=temp_signals).astype(bool)

    temp_signals = np.zeros(len(feed))
    temp_signals[long_trades.exit_idx] = True
    long_exits = pd.Series(index=feed.index, data=temp_signals).astype(bool)

    temp_signals = np.zeros(len(feed))
    temp_signals[short_trades.entry_idx] = True
    short_entries = pd.Series(index=feed.index, data=temp_signals).astype(bool)

    temp_signals = np.zeros(len(feed))
    temp_signals[short_trades.exit_idx] = True
    short_exits = pd.Series(index=feed.index, data=temp_signals).astype(bool)

    fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    spread = feed.close
    zscore = feed.zscore

    spread.vbt.plot(add_trace_kwargs=dict(row=1, col=1), fig=fig, title=feed._underlying_info)
    zscore.vbt.plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)

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

    fig.update_layout(height=400, width=1800)
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


def value_at_risk(df: pd.DataFrame, percentile=5):
    temp = df["returns"]
    temp = temp[temp.notna()]
    temp = temp.sort_values()
    index = (percentile / 100) * (len(temp) + 1)
    floor_index = int(np.floor(index))
    ceiling_index = int(np.ceil(index))
    if floor_index != index:
        value_at_risk = (temp.iloc[floor_index] + temp.iloc[ceiling_index]) / 2
    else:
        value_at_risk = temp.iloc[floor_index]
    return value_at_risk


class FeeInfo:
    NamedTupleGenerator = namedtuple("FeeInfo", ["fee_pct", "fee_fixed", "slippage", "init_margin", "maint_margin"])

    def __init__(
        self,
        fee_pct: float = 0,
        fee_fixed: float = 0,
        slippage: float = 0,
        init_margin: float = 0,
        maint_margin: float = 0.25,
    ):
        self.fee_pct = fee_pct
        self.fee_fixed = fee_fixed
        self.slippage = slippage
        self.init_margin = init_margin
        self.maint_margin = maint_margin

    def to_namedtuple(self) -> Tuple:
        return self.NamedTupleGenerator(
            self.fee_pct, self.fee_fixed, self.slippage, self.init_margin, self.maint_margin
        )


exchange_data = {
    Exchange.BINANCE: {
        "fee_info": FeeInfo(fee_pct=0.0004),
    },
    Exchange.BITMEX: {
        "fee_info": FeeInfo(fee_pct=0.0005),
    },
    Exchange.BYBIT: {
        "fee_info": FeeInfo(fee_pct=0.0006),
    },
    Exchange.COINFLEX: {
        "fee_info": FeeInfo(fee_pct=0.0005),
    },
    Exchange.FTX: {
        "fee_info": FeeInfo(fee_pct=0.0007),
    },
    Exchange.GATEIO: {
        "fee_info": FeeInfo(fee_pct=0.00075),
    },
    Exchange.KUCOIN: {
        "fee_info": FeeInfo(fee_pct=0.0006),
    },
    Exchange.OKEX: {
        "fee_info": FeeInfo(fee_pct=0.0005),
    },
}


class Client:
    def __init__(self):
        self.cm = cryptomart.Client()

    @pyutil.cache.cached("/tmp/cache/spread", is_method=True, log_level="INFO")
    def get_spread(
        self,
        exchange_a: Exchange,
        exchange_b: Exchange,
        symbol: Symbol,
        instType: InstrumentType = InstrumentType.PERPETUAL,
        interval: Interval = Interval.interval_1d,
        starttime: Union[datetime, Tuple[int]] = None,
        endtime: Union[datetime, Tuple[int]] = None,
        z_score_period: int = 30,
        cache_kwargs: Dict = {},
    ) -> Spread:

        """Get a single spread

        Args:
            exchange_a (Exchange): exchange_a
            exchange_b (Exchange): exchange_b
            symbol (Symbol): symbol
            instType (InstrumentType, optional): instrument type. Defaults to InstrumentType.PERPETUAL.
            interval (Interval, optional): interval. Defaults to Interval.interval_1d.
            starttime (Union[datetime, Tuple[int]], optional): starttime of ohlcv data. Defaults to None.
            endtime (Union[datetime, Tuple[int]], optional): endtime of ohlcv data. Defaults to None.
            z_score_period (int, optional): rolling window for spread z-score. Defaults to 30.
            cache_kwargs (Dict, optional): kwargs to control cache. Defaults to {}.

        Returns:
            Spread: spread
        """
        instance_a: cryptomart.ExchangeAPIBase = getattr(self.cm, exchange_a)
        instance_b: cryptomart.ExchangeAPIBase = getattr(self.cm, exchange_b)

        ohlcv_a = instance_a.ohlcv(symbol, instType, interval, starttime, endtime)
        ohlcv_b = instance_b.ohlcv(symbol, instType, interval, starttime, endtime)

        return Spread.from_ohlcv(ohlcv_a, ohlcv_b, z_score_period)

    def get_spreads(
        self,
        exchanges: List[Exchange] = Exchange._values(),
        symbol: Union[Symbol, List[Symbol]] = Symbol._values(),
        instType: Union[InstrumentType, List[InstrumentType]] = InstrumentType.PERPETUAL,
        interval: Interval = Interval.interval_1d,
        starttime: Union[datetime, Tuple[int]] = None,
        endtime: Union[datetime, Tuple[int]] = None,
        z_score_period: int = 30,
        good_spreads: bool = True,
        minimum_rows: int = 180,
        cache_kwargs={},
    ) -> List[Spread]:
        """Pull all spreads from cryptomart and export as array of Spread objects

        Args:
            z_score_period (int, optional): What time-period (rows) should the z-score be cmoputed with. Defaults to 90.
            good_spreads (bool, optional): Filter out bad quality spreads. Defaults to True.
            minimum_rows (int, optional): Minimum rows requirement for good_spreads filter. Defaults to 180.

        Returns:
            List: List of Spreads
        """
        logger.warning(
            f"get_spreads: z_score_period={z_score_period}, good_spreads={good_spreads}, minmum_rows={minimum_rows}"
        )
        if not isinstance(instType, list):
            instType = [instType]
        if not isinstance(symbol, list):
            symbol = [symbol]

        all_instruments = pd.DataFrame()
        for exchange in exchanges:
            instance: cryptomart.ExchangeAPIBase = getattr(self.cm, exchange)
            exchange_instruments = instance.active_instruments.assign(exchange=exchange)[
                [Instrument.instType, Instrument.symbol, "exchange"]
            ]
            all_instruments = pd.concat([all_instruments, exchange_instruments], ignore_index=True)

        # Create cartesian product of all exchange combinations with the same instType and symbol
        instrument_pairs = all_instruments.merge(
            all_instruments, on=[Instrument.instType, Instrument.symbol], suffixes=("_a", "_b")
        )

        # Filter out duplicates
        instrument_pairs = instrument_pairs[instrument_pairs.exchange_a > instrument_pairs.exchange_b].reset_index(
            drop=True
        )

        # Filter out blacklisted sybmols
        instrument_pairs = instrument_pairs[~np.isin(instrument_pairs.symbol, BLACKLISTED_SYMBOLS)]

        spread_list = []
        for idx, row in instrument_pairs.iterrows():
            spread = self.get_spread(
                row.exchange_a,
                row.exchange_b,
                row.symbol,
                instType=row.instType,
                interval=interval,
                starttime=starttime,
                endtime=endtime,
                z_score_period=z_score_period,
                cache_kwargs=cache_kwargs,
            )
            spread_list.append(spread)

        spreads = pd.DataFrame({"feed": spread_list})
        details = spread_details(spread_list)

        spreads = pd.concat([spreads, details], axis=1)

        if good_spreads:
            spreads = spreads[
                (spreads.missing_rows == 0) & (spreads.total_rows > minimum_rows) & (spreads.gaps == 0)
            ].reset_index(drop=True)
        logger.warning(f"get_spreads: done. Fetched {len(spreads)} spreads")
        return spreads

    def run_strategy(
        self,
        feed: Spread,
        initial_cash=150000,
        trade_value=10000,
        z_score_thresh=1,
        vbt_function: Callable = from_order_func_wrapper,
        logging=True,
    ):
        """Run vectorbt strategy `vbt_function` on `feed`

        Args:
            feed (Spread): spread
            initial_cash (int, optional): initial cash for strategy. Defaults to 150000.
            trade_value (int, optional): value of trade for each leg. Defaults to 10000.
            z_score_thresh (int, optional): z-score above this threshold will trigger a trade. Defaults to 1.
            vbt_function (Callable, optional): vectorbt strategy function. Defaults to from_order_func_wrapper.
            logging (bool, optional): If True, logs written to disk in `./logs`. Defaults to True.

        Returns:
            Portfolio: results of backtest (portfolio)
        """
        assert vbt_function is not None
        fee_info_a = exchange_data[feed.ohlcv_a.exchange_name]["fee_info"].to_namedtuple()
        fee_info_b = exchange_data[feed.ohlcv_b.exchange_name]["fee_info"].to_namedtuple()
        fee_info = (fee_info_a, fee_info_b)
        var = (
            (value_at_risk(feed.ohlcv_a, percentile=5) / 100, value_at_risk(feed.ohlcv_a, percentile=95) / 100),
            (value_at_risk(feed.ohlcv_b, percentile=5) / 100, value_at_risk(feed.ohlcv_b, percentile=95) / 100),
        )
        close_prices = feed.underlying_col("close").to_numpy()

        feed = feed.set_index("open_time", drop=False)

        func_args = (
            feed.zscore.to_numpy(),  # z-score
            trade_value,  # target trade value
            z_score_thresh,  # entry zscore thresh
            0,  # exit zscore thresh
            fee_info,  # fee info
            var,  # value at risk
            logging,  # logging
        )

        if logging:
            # Create log file path
            log_dir = "logs"
            day_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
            hour_dir = os.path.join(day_dir, datetime.datetime.now().strftime("%I"))
            minute_dir = os.path.join(hour_dir, datetime.datetime.now().strftime("%M"))
            filename = feed._underlying_info.replace(" ", "").replace(".", "_") + ".log"
            out_path = os.path.join(minute_dir, filename)

            i = 2
            while os.path.exists(out_path):
                filename = feed._underlying_info.replace(" ", "").replace(".", "_") + f"_{i}.log"
                i += 1
                out_path = os.path.join(minute_dir, filename)

            vbt_function = redirect_stdout(out_path)(vbt_function)
        portfolio = vbt_function(close_prices, func_args, initial_cash)

        portfolio = portfolio.replace(
            wrapper=portfolio.wrapper.replace(index=feed.index, columns=feed.underlying_col("close").columns)
        )

        return BtResults(feed, portfolio)

    def run_strategy_batch(
        self,
        spreads: pd.DataFrame,
        initial_cash=150000,
        trade_value=10000,
        z_score_thresh=1,
        vbt_function: Callable = from_order_func_wrapper,
        logging=False,
    ) -> pd.Series:
        """Runs strategy in parallel on dataframe of spreads. Saves results as a Series of BtResults objects in self.bt_results

        Args:
            spreads (pd.DataFrame): dataframe of spread elements
            initial_cash (int, optional): initial cash for strategy . Defaults to 150000.
            trade_value (int, optional): value of trade for each leg. Defaults to 10000.
            z_score_thresh (int, optional): z-score above this threshold will trigger a trade. Defaults to 1.
            vbt_function (Callable, optional): vectorbt strategy function. Defaults to from_order_func_wrapper.
            logging (bool, optional): If True, logs written to disk in `./logs`. Defaults to False.

        Returns:
            pd.Series: Series of BtResults
        """
        logger.warning(
            f"run_strategy_batch: initial_cash={initial_cash}, trade_value={trade_value}, z_score_thresh={z_score_thresh}, logging={logging}"
        )

        @pyutil.profiling.timed(return_time=True)
        def run():
            return spreads.apply(
                lambda r: self.run_strategy(
                    r.feed, initial_cash, trade_value, z_score_thresh, vbt_function, logging=logging
                ),
                axis=1,
            )

        res, elapsed_time = run()
        self.bt_results: pd.Series[BtResults] = res
        logger.warning(f"Backtest results saved in self.bt_results. Total runtime={elapsed_time}s")
        return self.bt_results

    def backtest_stats(self) -> pd.DataFrame:
        """Compute stats for all BtResults in self.bt_results to be used for aggregate analysis

        Returns:
            pd.DataFrame: DataFrame where each row represents a backtest from self.bt_results and each column represents a stat for the backtest
        """
        if self.bt_results is None:
            logger.error("No strategy has been run")
            return

        feeds = [r.feed for r in self.bt_results]
        details = spread_details(feeds)

        res = self.bt_results.apply(
            lambda r: pd.Series({"total_return": r.portfolio.total_return(), "end_value": r.portfolio.final_value()})
        )

        res = pd.concat([res, details], axis=1)

        return res

    def plot_result(self, result: Union[int, BtResults]):
        if isinstance(result, int):
            result: BtResults = self.bt_results.iloc[result]

        display(plot_spread_and_zscore(result))

    def inspect_result(self, result: Union[int, BtResults]):
        if isinstance(result, int):
            result: BtResults = self.bt_results.iloc[result]

        display(result.portfolio.plot_cum_returns())
        display(result.portfolio.stats())
        self.plot_result(result)
        display(result.portfolio.trades.records_readable.sort_values("Entry Timestamp").head(10))
        display(result.portfolio.orders.records_readable.sort_values("Timestamp").head(20))
        display(result.feed.underlyings)
