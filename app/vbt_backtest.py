from collections import namedtuple
from typing import Tuple

import numpy as np
import vectorbt as vbt
import vectorbt.portfolio.enums as enums
import vectorbt.portfolio.nb as nb
from numba import njit
from pyutil.dicts import format_dict

LONG = 1
SHORT = -1
NEUTRAL = 0

BacktestArgs = namedtuple(
    "BacktestArgs",
    [
        "initial_cash",
        "trade_value",
        "z_score_thresholds",
        "var",
        "init_margin",
        "maint_margin",
        "fee_pct",
        "fee_fixed",
        "zscore",
        "funding_rate",
        "bid_ask_spread",
        "logging",
    ],
)


@njit
def log(logging, *msg):
    if logging:
        print(*msg)


@njit
def sort_call_seq_for_entry(c, call_seq_out, bt_args):
    current_zscore = bt_args.zscore[c.i]
    potential_short_entry = current_zscore > bt_args.z_score_thresholds[0]
    if potential_short_entry:
        call_seq_out[:] = np.array([1, 0])
    else:
        call_seq_out[:] = np.array([0, 1])


@njit
def sort_call_seq_for_exit(c, call_seq_out):
    call_seq_out[:] = (-c.last_position).argsort()


@njit
def pre_group_func_nb(c, bt_args):
    """Called before processing a group. (Group contains both legs of spread)"""
    directions = np.array([NEUTRAL, NEUTRAL], dtype=np.int_)
    spread_dir = np.array([NEUTRAL])
    call_seq_out = np.empty(c.group_len, dtype=np.int_)
    return (directions, spread_dir, call_seq_out)


@njit
def pre_segment_func_nb(c, directions, spread_dir, call_seq_out, bt_args):
    """Called before segment is processed. Segment refers to a single time step within a group"""
    if (directions == NEUTRAL).all():
        sort_call_seq_for_entry(c, call_seq_out, bt_args)
    else:
        sort_call_seq_for_exit(c, call_seq_out)

    log(
        bt_args.logging,
        "\ntime_idx",
        c.i,
        "|dir:",
        directions,
        "|spread_dir:",
        spread_dir[0],
        "|zscore:",
        bt_args.zscore[c.i],
        "|close:",
        c.close[c.i],
        "|positions:",
        c.last_position,
        "|col_val:",
        c.last_val_price,
        "|cash:",
        c.last_cash,
        "|value",
        c.last_value,
        "|last_return",
        c.last_return,
        "|bid_ask_spread",
        bt_args.bid_ask_spread[c.i],
        "|call_seq_out",
        call_seq_out,
    )
    return (directions, spread_dir, call_seq_out)


@njit
def flex_order_func_nb(c, directions, spread_dir, call_seq_out, bt_args):
    current_zscore = bt_args.zscore[c.i]
    current_spread_direction = spread_dir[0]
    low_zscore = current_zscore < -bt_args.z_score_thresholds[0]
    high_zscore = current_zscore > bt_args.z_score_thresholds[1]
    col = call_seq_out[c.call_idx % c.group_len]
    trade_col = c.close[c.i, :].argmax()  # Column to use for determining trade size (highest leg price)

    fee_pct = bt_args.fee_pct[col]
    fee_fixed = bt_args.fee_fixed[col]
    slippage = bt_args.bid_ask_spread[c.i, col]
    init_margin = bt_args.init_margin[col]
    maint_margin = bt_args.maint_margin[col]

    low_var = bt_args.var[col][0]
    high_var = bt_args.var[col][1]
    SAFETY_BUFFER = 3

    current_price = c.close[c.i, col]
    current_leg_direction = directions[col]

    if current_leg_direction == NEUTRAL:
        log(bt_args.logging, "Not in market", "col=", col)

        #                  Nominal amount          Leverage factor
        # trade_size = -(trade_value / price) * (target_vol / volatility)
        leverage_factor = 1 / (
            (max(abs(bt_args.var[trade_col][0]), abs(bt_args.var[trade_col][1])) * SAFETY_BUFFER) + maint_margin
        )
        trade_size = (bt_args.trade_value / c.close[c.i, trade_col]) * leverage_factor

        if low_zscore:
            log(bt_args.logging, "Going long on spread", "leverage_factor=", leverage_factor)

            return col, nb.order_nb(
                size=trade_size if col == 1 else -trade_size,
                price=c.close[c.i, col],
                fees=fee_pct,
                fixed_fees=fee_fixed,
                raise_reject=True,
                allow_partial=False,
            )

        elif high_zscore:
            log(bt_args.logging, "Going short on spread", "leverage_factor=", leverage_factor)

            return col, nb.order_nb(
                size=trade_size if col == 0 else -trade_size,
                price=c.close[c.i, col],
                fees=fee_pct,
                fixed_fees=fee_fixed,
                raise_reject=True,
                allow_partial=False,
            )

        log(bt_args.logging, "Will not enter")
        return -1, nb.order_nothing_nb()

    else:
        # Break loop after an entry
        if c.call_idx >= c.group_len:
            return -1, nb.order_nothing_nb()

        current_position = c.last_position[col]

        # Check for liquidation.
        liq_price = c.close[c.i, col] * (
            1 + (max(abs(low_var), abs(high_var)) * SAFETY_BUFFER) * -current_leg_direction
        )

        log(
            bt_args.logging,
            "In market",
            "col=",
            col,
            "leg_dir=",
            current_leg_direction,
            "cur_pos=",
            current_position,
            "liq_price=",
            liq_price,
        )

        if (current_leg_direction == LONG and current_price < liq_price) or (
            current_leg_direction == SHORT and current_price > liq_price
        ):
            log(bt_args.logging, "liquidated at price:", current_price)
            return col, nb.close_position_nb()

        if current_spread_direction == LONG and current_zscore >= -bt_args.z_score_thresholds[1]:
            log(bt_args.logging, "Closing a long spread")
            return col, nb.order_nb(
                size=-current_position,
                price=c.close[c.i, col],
                fees=fee_pct,
                fixed_fees=fee_fixed,
                raise_reject=True,
            )

        elif current_spread_direction == SHORT and current_zscore <= bt_args.z_score_thresholds[1]:
            log(bt_args.logging, "Closing a short spread")
            return col, nb.order_nb(
                size=-current_position,
                price=c.close[c.i, col],
                fees=fee_pct,
                fixed_fees=fee_fixed,
                raise_reject=True,
            )

        log(bt_args.logging, "Will not exit")
        return -1, nb.order_nothing_nb()


@njit
def post_order_func_nb(c, directions, spread_dir, call_seq_out, bt_args):
    if c.order_result.status == enums.OrderStatus.Filled:
        if directions[c.col] == NEUTRAL:
            # Entering a position
            directions[c.col] = LONG if c.order_result.side == enums.OrderSide.Buy else SHORT
        else:
            # Closing a position
            directions[c.col] = NEUTRAL
            if c.call_idx == c.group_len - 1:
                # Finishing closing a spread position
                # sort call_seq to favor shorts first on potential entry in same time_idx
                sort_call_seq_for_entry(c, call_seq_out, bt_args)

        log(
            bt_args.logging,
            c.last_oidx,
            "Order filled on column=",
            c.col,
            "(side=",
            "[BUY]" if c.order_result.side == enums.OrderSide.Buy else "[SELL]",
            c.order_result.side,
            "size=",
            c.order_result.size,
            "price=",
            c.order_result.price,
            "value=",
            c.order_result.size * c.order_result.price,
            "fees=",
            c.order_result.fees,
            ")",
        )

        # Update direction state
        if (directions != NEUTRAL).all():
            if directions[0] == LONG:
                spread_dir[0] = SHORT
            else:
                spread_dir[0] = LONG
        elif (directions == NEUTRAL).all():
            spread_dir[0] = NEUTRAL
    else:
        log(bt_args.logging, c.order_result.status)
        log(bt_args.logging, c.order_result.status_info)
    return None


def from_order_func_wrapper(close_prices: np.ndarray, bt_args: BacktestArgs):
    log(bt_args.logging, "Run backtest with the following settings:")
    log(
        bt_args.logging,
        format_dict(
            {
                **bt_args._asdict(),
                "zscore": bt_args.zscore.shape,
                "funding_rate": bt_args.funding_rate.shape,
                "bid_ask_spread": bt_args.bid_ask_spread.shape,
                "zscore": bt_args.zscore.shape,
            },
        ),
    )
    log(bt_args.logging, "=" * 100 + "\n\n")
    return vbt.Portfolio.from_order_func(
        close_prices,
        flex_order_func_nb,
        bt_args,
        group_by=True,
        cash_sharing=True,
        init_cash=bt_args.initial_cash,
        flexible=True,
        max_orders=close_prices.shape[0] * 4,
        pre_group_func_nb=pre_group_func_nb,
        pre_group_args=(bt_args,),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(bt_args,),
        post_order_func_nb=post_order_func_nb,
        post_order_args=(bt_args,),
        freq="1d",
    )
