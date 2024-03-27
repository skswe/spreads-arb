"""This module provides an API for running spread backtests from the `vbt_backtest` module.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from ..feeds import Spread
from ..util import redirect_stdout
from . import vbt_backtest, vbt_backtest_chained, vbt_bt_quotes
from .BacktestResult import BacktestResult
from .ChainedBacktestResult import ChainedBacktestResult


class BacktestRunner:
    # Default module
    vbt_module = vbt_backtest
    
    def __init__(
        self,
        initial_cash=150000,
        trade_value=10000,
        z_score_period=30,
        z_score_thresholds=(0, 1),  # (entry_threshold, exit_threshold)
        use_slippage=True,
        use_funding_rate=True,
        profitable_only=True,  # Expected return on total trade size after fees and slippage
        log_dir=None,
        force_logging=False,
    ):
        self.initial_cash = initial_cash
        self.trade_value = trade_value
        self.z_score_period = z_score_period
        self.z_score_thresholds = z_score_thresholds
        self.use_slippage = use_slippage
        self.use_funding_rate = use_funding_rate
        self.profitable_only = profitable_only

        self.logging = force_logging or (log_dir is not None)
        if log_dir is not None:
            print("Logging is enabled. Performance will be impacted.")
            self.log_dir = self.make_log_dir(log_dir)

    @staticmethod
    def make_log_dir(path):
        time_now = datetime.now()
        date_now = time_now.date().strftime("%Y-%m-%d")
        hour_now = time_now.strftime("%H")
        minute_now = time_now.strftime("%M")
        full_log_dir = os.path.join(path, date_now, hour_now, minute_now)
        os.makedirs(full_log_dir, exist_ok=True)
        return full_log_dir

    @staticmethod
    def unique_file_name(path):
        i = 2
        while os.path.exists(path):
            path = f"{path}_{i}"
            i += 1
        return f"{path}.log"

    def create_spreads(self, ohlcvs, fee_info, funding_rates=None, bas=None) -> pd.DataFrame:
        """Create spreads from OHLCV. ohlcvs with 0 rows should be filtered out before calling this function.

        Args:
            ohlcvs: DataFrame with OHLCV data.
            fee_info: DataFrame with margin and fee information.
            funding_rates: DataFrame with funding rate data. Defaults to None.
            bas: DataFrame with bid ask spread and slippage data. Defaults to None.

        Returns:
            DataFrame with spread data
        """

        instrument_data = ohlcvs.copy()

        if funding_rates is not None:
            instrument_data = instrument_data.merge(funding_rates, left_index=True, right_index=True)

        if bas is not None:
            instrument_data = instrument_data.merge(bas, left_index=True, right_index=True)

        instrument_data = instrument_data.join(fee_info).reorder_levels(instrument_data.index.names)

        instrument_data_crossed = (
            instrument_data.reset_index()
            .merge(instrument_data.reset_index(), how="cross", suffixes=("_a", "_b"))
            .pipe(
                lambda df: df[
                    (df.exchange_a < df.exchange_b) & (df.inst_type_a <= df.inst_type_b) & (df.symbol_a == df.symbol_b)
                ]
            )
            .drop(columns="symbol_b")
            .rename(columns={"symbol_a": "symbol"})
            .reset_index(drop=True)
        )

        spreads = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], [], [], [], []], names=["exchange_a", "exchange_b", "inst_type_a", "inst_type_b", "symbol"]
            )
        )

        for idx, row in instrument_data_crossed.iterrows():
            indexer_a = (row.exchange_a, row.inst_type_a, row.symbol)
            indexer_b = (row.exchange_b, row.inst_type_b, row.symbol)
            indexer = (row.exchange_a, row.exchange_b, row.inst_type_a, row.inst_type_b, row.symbol)
            spread = Spread.from_ohlcv(pickle.loads(row.ohlcv_a), pickle.loads(row.ohlcv_b))

            if funding_rates is not None:
                fr_a = pickle.loads(funding_rates.at[indexer_a, "funding_rate"])
                fr_b = pickle.loads(funding_rates.at[indexer_b, "funding_rate"])
                spread.add_funding_rate(fr_a, fr_b)

            if bas is not None:
                bid_ask_a = pickle.loads(bas.at[indexer_a, "bid_ask_spread"])
                bid_ask_b = pickle.loads(bas.at[indexer_b, "bid_ask_spread"])
                spread.add_bid_ask_spread(bid_ask_a, bid_ask_b)
                spreads.at[indexer, (f"avg_ba_spread_{s}" for s in ("a", "b"))] = set(
                    x.bid_ask_spread.mean() for x in (bid_ask_a, bid_ask_b)
                )

            spreads.at[indexer, "spread"] = pickle.dumps(spread)
            spreads.at[indexer, "alias"] = (
                f"{row.exchange_a[:4]}_{row.exchange_b[:4]}_{row.inst_type_a[:4]}_{row.inst_type_b[:4]}_{row.symbol}"
            )
            spreads.at[indexer, "volatility"] = spread.volatility()
            spreads.at[indexer, "earliest_time"] = spread.earliest_time
            spreads.at[indexer, "latest_time"] = spread.latest_time
            spreads.at[indexer, "valid_rows"] = len(spread.valid_rows)
            spreads.at[indexer, "missing_rows"] = len(spread.missing_rows)
            spreads.at[indexer, "gaps"] = spread.gaps

            fee_info_keys = ["init_margin", "maint_margin", "fee_pct", "fee_fixed"]
            for key in ["a", "b"]:
                spreads.at[indexer, f"fee_info_{key}"] = pickle.dumps(
                    {k: getattr(row, f"{k}_{key}") for k in fee_info_keys}
                )

        return spreads

    def _run_single_spread(self, row: pd.Series):
        spread: Spread = pickle.loads(row.spread)
        alias = row.alias

        close_prices = np.array(spread.underlying_col("close"))

        if not self.use_funding_rate:
            zero_funding_rate = pd.DataFrame({"timestamp": spread.time_only, "funding_rate": [0] * len(spread)})
            spread.add_funding_rate(zero_funding_rate, zero_funding_rate)

        if not self.use_slippage:
            zero_bid_ask_spread = pd.DataFrame({"date": spread.time_only, "bid_ask_spread": [0] * len(spread)})
            spread.add_bid_ask_spread(zero_bid_ask_spread, zero_bid_ask_spread)

        funding_rate = np.array(spread.underlying_col("funding_rate"))
        bid_ask_spread = np.array(spread.underlying_col("bid_ask_spread"))

        zscore = np.array(spread.zscore(period=self.z_score_period))
        setattr(spread, "zscore_period", self.z_score_period)
        var = tuple(zip(spread.value_at_risk(percentile=5), spread.value_at_risk(percentile=95)))

        fee_info = {}
        fee_info_a = pickle.loads(row.fee_info_a)
        fee_info_b = pickle.loads(row.fee_info_b)
        for key in ["init_margin", "maint_margin", "fee_pct", "fee_fixed"]:
            fee_info[key] = (fee_info_a[key], fee_info_b[key])

        bt_args = self.vbt_module.BacktestArgs(
            initial_cash=self.initial_cash,
            trade_value=self.trade_value,
            z_score_thresholds=self.z_score_thresholds,
            var=var,
            init_margin=fee_info["init_margin"],
            maint_margin=fee_info["maint_margin"],
            fee_pct=fee_info["fee_pct"],
            fee_fixed=fee_info["fee_fixed"],
            zscore=zscore,
            funding_rate=funding_rate,
            bid_ask_spread=bid_ask_spread,
            logging=self.logging,
            profitable_only=self.profitable_only,
        )
        if hasattr(self, "log_dir"):
            log_file_name = os.path.join(self.log_dir, self.unique_file_name(alias))
            bt_func = redirect_stdout(log_file_name)(self.vbt_module.run)
        else:
            bt_func = self.vbt_module.run
        res = bt_func(close_prices, bt_args)
        res = res.replace(
            wrapper=res.wrapper.replace(
                index=spread.open_time, columns=spread.underlying_col("close").columns.get_level_values(0)
            )
        )
        return BacktestResult(res, spread)

    def run(self, spreads: pd.DataFrame, exchange_subset=[], inst_type_subset=["perpetual"], symbol_subset=[]):
        """Run individual spread backtest for each of the given spreads 

        Args:
            spreads: DataFrame containing spread data. It should be created with the `create_spreads` method.
            exchange_subset: Exchanges to filter the spreads by. Defaults to [].
            inst_type_subset: Instrument Types to filter the spreads by. Defaults to ["perpetual"].
            symbol_subset: Symbols to filter the spreads by. Defaults to [].

        Returns:
            DataFrame containing a `BacktestResult` object for each of the spreads.
        """
        self.vbt_module = vbt_backtest
        index_filter = pd.MultiIndex.from_product(
            [exchange_subset, exchange_subset, inst_type_subset, inst_type_subset, symbol_subset]
        )
        if len(index_filter) > 0:
            spreads = spreads.filter(index_filter, axis=0)

        results = spreads.apply(self._run_single_spread, axis=1)
        return results

    def run_chained(self, spreads: pd.DataFrame):
        """Run the chained spread backtest for the given spreads

        Args:
            spreads: DataFrame containing spread data. It should be created with the `create_spreads` method.

        Returns:
            DataFrame containing a `ChainedBacktestResult` object for each of the spreads.
        """
        self.vbt_module = vbt_backtest_chained
        leg_0_close = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("close").T.iloc[0], axis=1)
        )
        leg_1_close = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("close").T.iloc[1], axis=1)
        )
        close_prices = np.concatenate([leg_0_close, leg_1_close], axis=0).T

        if not self.use_funding_rate:
            funding_rate = np.stack(
                spreads.apply(lambda s: np.zeros((2, len(pickle.loads(s.spread)))), axis=1), axis=1
            ).T
        else:
            funding_rate = np.stack(
                spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("funding_rate").T, axis=1), axis=1
            ).T

        if not self.use_slippage:
            bid_ask_spread = np.stack(
                spreads.apply(lambda s: np.zeros((2, len(pickle.loads(s.spread)))), axis=1), axis=1
            ).T
        else:
            bid_ask_spread = np.stack(
                spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("bid_ask_spread").T, axis=1), axis=1
            ).T

        zscore = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).zscore("close", self.z_score_period), axis=1)
        ).T

        short_var = np.stack(spreads.spread.apply(lambda s: pickle.loads(s).short_var()))
        long_var = np.stack(spreads.spread.apply(lambda s: pickle.loads(s).long_var()))
        fee_info_a = spreads.apply(lambda s: pd.Series(pickle.loads(s.fee_info_a)), axis=1)
        fee_info_b = spreads.apply(lambda s: pd.Series(pickle.loads(s.fee_info_b)), axis=1)

        bt_args = self.vbt_module.BacktestArgs(
            initial_cash=self.initial_cash,
            trade_value=self.trade_value,
            z_score_thresholds=self.z_score_thresholds,
            long_var=long_var,
            short_var=short_var,
            init_margin=np.array([fee_info_a.init_margin, fee_info_b.init_margin]).T,
            maint_margin=np.array([fee_info_a.maint_margin, fee_info_b.maint_margin]).T,
            fee_pct=np.array([fee_info_a.fee_pct, fee_info_b.fee_pct]).T,
            fee_fixed=np.array([fee_info_a.fee_fixed, fee_info_b.fee_fixed]).T,
            zscore=zscore,
            funding_rate=funding_rate,
            bid_ask_spread=bid_ask_spread,
            logging=self.logging,
            profitable_only=self.profitable_only,
        )

        if hasattr(self, "log_dir"):
            log_file_name = os.path.join(self.log_dir, self.unique_file_name(f"{len(spreads)}_chained"))
            bt_func = redirect_stdout(log_file_name)(self.vbt_module.run)
        else:
            bt_func = self.vbt_module.run
        res = bt_func(close_prices, bt_args)

        col_0_names = np.array(
            spreads.apply(
                lambda s: pickle.loads(s.spread).underlying_col("close").columns.get_level_values(0)[0], axis=1
            )
        )
        col_1_names = np.array(
            spreads.apply(
                lambda s: pickle.loads(s.spread).underlying_col("close").columns.get_level_values(0)[1], axis=1
            )
        )
        col_names = np.concatenate([col_0_names, col_1_names], axis=0)
        res = res.replace(
            wrapper=res.wrapper.replace(index=pickle.loads(spreads.iloc[0].spread).time_only, columns=col_names)
        )

        return ChainedBacktestResult(res, spreads)

    def run_quotes(self, spreads: pd.DataFrame):
        """Run the chained quote spread backtest for the given spreads

        Args:
            spreads: DataFrame containing spread data. It should be created with the `create_spreads` method.

        Returns:
            None - check log file for results
        """
        self.vbt_module = vbt_bt_quotes
        leg_0_bid = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("bid_price").T.iloc[0], axis=1)
        )
        leg_1_bid = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("bid_price").T.iloc[1], axis=1)
        )
        bid_prices = np.concatenate([leg_0_bid, leg_1_bid], axis=0).T

        leg_0_ask = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("ask_price").T.iloc[0], axis=1)
        )
        leg_1_ask = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("ask_price").T.iloc[1], axis=1)
        )
        ask_prices = np.concatenate([leg_0_ask, leg_1_ask], axis=0).T

        leg_0_bid_sz = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("bid_amount").T.iloc[0], axis=1)
        )
        leg_1_bid_sz = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("bid_amount").T.iloc[1], axis=1)
        )
        bid_sizes = np.concatenate([leg_0_bid_sz, leg_1_bid_sz], axis=0).T

        leg_0_ask_sz = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("ask_amount").T.iloc[0], axis=1)
        )
        leg_1_ask_sz = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("ask_amount").T.iloc[1], axis=1)
        )
        ask_sizes = np.concatenate([leg_0_ask_sz, leg_1_ask_sz], axis=0).T

        leg_0_mid = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("mid_price").T.iloc[0], axis=1)
        )
        leg_1_mid = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).underlying_col("mid_price").T.iloc[1], axis=1)
        )
        mid_prices = np.concatenate([leg_0_mid, leg_1_mid], axis=0).T

        zscore = np.array(
            spreads.apply(lambda s: pickle.loads(s.spread).zscore("bid_price", self.z_score_period), axis=1)
        ).T

        short_var = np.stack(spreads.spread.apply(lambda s: pickle.loads(s).short_var()))
        long_var = np.stack(spreads.spread.apply(lambda s: pickle.loads(s).long_var()))
        fee_info_a = spreads.apply(lambda s: pd.Series(pickle.loads(s.fee_info_a)), axis=1)
        fee_info_b = spreads.apply(lambda s: pd.Series(pickle.loads(s.fee_info_b)), axis=1)

        bt_args = self.vbt_module.BacktestArgs(
            initial_cash=self.initial_cash,
            trade_value=self.trade_value,
            z_score_thresholds=self.z_score_thresholds,
            long_var=long_var,
            short_var=short_var,
            init_margin=np.array([fee_info_a.init_margin, fee_info_b.init_margin]).T,
            maint_margin=np.array([fee_info_a.maint_margin, fee_info_b.maint_margin]).T,
            fee_pct=np.array([fee_info_a.fee_pct, fee_info_b.fee_pct]).T,
            fee_fixed=np.array([fee_info_a.fee_fixed, fee_info_b.fee_fixed]).T,
            zscore=zscore,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
            logging=self.logging,
            profitable_only=self.profitable_only,
        )

        if hasattr(self, "log_dir"):
            log_file_name = os.path.join(self.log_dir, self.unique_file_name(f"{len(spreads)}_chained"))
            bt_func = redirect_stdout(log_file_name)(self.vbt_module.run)
        else:
            bt_func = self.vbt_module.run
        res = bt_func(mid_prices, bt_args)
