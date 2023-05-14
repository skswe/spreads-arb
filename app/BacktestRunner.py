import os
import pickle
from datetime import datetime

import app
import numpy as np
import pandas as pd
import pyutil

from .analysis_tools import BacktestResult, ChainedBacktestResult
from .feeds import Spread


class BacktestRunner:
    def __init__(
        self,
        initial_cash=150000,
        trade_value=10000,
        vbt_function=app.vbt.from_order_func_wrapper_chained,
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
        self.vbt_function = vbt_function
        self.z_score_period = z_score_period
        self.z_score_thresholds = z_score_thresholds
        self.use_slippage = use_slippage
        self.use_funding_rate = use_funding_rate
        self.profitable_only = profitable_only

        self.logging = force_logging or (log_dir is not None)
        if log_dir is not None:
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

        bt_args = app.vbt.vbt_backtest.BacktestArgs(
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
            bt_func = pyutil.io.redirect_stdout(log_file_name)(self.vbt_function)
        else:
            bt_func = self.vbt_function
        res = bt_func(close_prices, bt_args)
        res = res.replace(
            wrapper=res.wrapper.replace(
                index=spread.open_time, columns=spread.underlying_col("close").columns.get_level_values(0)
            )
        )
        return BacktestResult(res, spread)

    def run(self, spreads: pd.DataFrame, exchange_subset=[], inst_type_subset=["perpetual"], symbol_subset=[]):
        index_filter = pd.MultiIndex.from_product(
            [exchange_subset, exchange_subset, inst_type_subset, inst_type_subset, symbol_subset]
        )
        if len(index_filter) > 0:
            spreads = spreads.filter(index_filter, axis=0)

        results = spreads.apply(self._run_single_spread, axis=1)
        return results

    def run_chained(self, spreads: pd.DataFrame):
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

        bt_args = app.vbt.vbt_backtest_chained.BacktestArgs(
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
            bt_func = pyutil.io.redirect_stdout(log_file_name)(self.vbt_function)
        else:
            bt_func = self.vbt_function
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
