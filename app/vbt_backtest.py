from collections import namedtuple
from typing import Tuple

import numpy as np
import vectorbt as vbt
import vectorbt.portfolio.enums as enums
import vectorbt.portfolio.nb as nb
from numba import njit

LONG = 1
SHORT = -1
NEUTRAL = 0

BacktestArgs = namedtuple(
    "BacktestArgs",
    [
        "zscore",
        "trade_value",
        "zscore_entry_thresh",
        "zscore_exit_thresh",
        "fee_info",
        "var",
        "logging",
    ],
)


@njit
def log(logging, *msg):
    if logging:
        print(*msg)


@njit
def pre_group_func_nb(c, *args):
    """Called before processing a group. (Group contains both legs of spread)"""
    bt_args = BacktestArgs(*args)
    log(bt_args.logging, "Run backtest with the following settings:")
    log(bt_args.logging, "trade_value:", bt_args.trade_value)
    log(bt_args.logging, "zscore_entry_thresh:", bt_args.zscore_entry_thresh)
    log(bt_args.logging, "zscore_exit_thresh:", bt_args.zscore_exit_thresh)
    log(bt_args.logging, "fee_info:", bt_args.fee_info)
    log(bt_args.logging, "var:", bt_args.var)
    log(bt_args.logging, "logging:", bt_args.logging)
    log(bt_args.logging, "===========================")
    log(bt_args.logging, "===========================\n\n")
    
    directions = np.array([NEUTRAL, NEUTRAL], dtype=np.int_)
    spread_dir = np.array([NEUTRAL])
    call_seq_out = np.empty(c.group_len, dtype=np.int_)
    return (directions, spread_dir, call_seq_out)


@njit
def sort_call_seq_for_entry(c, call_seq_out, bt_args):
    current_zscore = bt_args.zscore[c.i]
    potential_short_entry = current_zscore > bt_args.zscore_entry_thresh
    if potential_short_entry:
        call_seq_out[:] = np.array([1, 0])
    else:
        call_seq_out[:] = np.array([0, 1])
        
@njit
def sort_call_seq_for_exit(c, call_seq_out):
    call_seq_out[:] = (-c.last_position).argsort()

        
@njit
def pre_segment_func_nb(c, directions, spread_dir, call_seq_out, *args):
    """Called before segment is processed. Segment refers to a single time step within a group"""
    bt_args = BacktestArgs(*args)

    if (directions == NEUTRAL).all():
        sort_call_seq_for_entry(c, call_seq_out, bt_args)
    else:
        sort_call_seq_for_exit(c, call_seq_out)
        
    log(
        bt_args.logging, 
        "\ntime_idx", c.i, 
        "|dir:", directions,
        "|spread_dir:", spread_dir[0],
        "|zscore:", bt_args.zscore[c.i],
        "|close:", c.close[c.i],
        "|positions:", c.last_position,
        "|col_val:", c.last_val_price,
        "|cash:", c.last_cash,
        "|value", c.last_value,
        "|last_return", c.last_return,
        "|call_seq_out", call_seq_out,
    )
    return (directions, spread_dir, call_seq_out)


@njit
def flex_order_func_nb(c, directions, spread_dir, call_seq_out, *args):
    bt_args = BacktestArgs(*args)
    current_zscore = bt_args.zscore[c.i]
    current_spread_direction = spread_dir[0]
    low_zscore = current_zscore < -bt_args.zscore_entry_thresh
    high_zscore = current_zscore > bt_args.zscore_entry_thresh
    col = call_seq_out[c.call_idx % c.group_len]
    trade_col = c.close[c.i, :].argmax() # Column to use for determining trade size (highest leg price)

    fee_pct = bt_args.fee_info[col].fee_pct
    fee_fixed = bt_args.fee_info[col].fee_fixed
    slippage = bt_args.fee_info[col].slippage
    init_margin = bt_args.fee_info[col].init_margin
    maint_margin = bt_args.fee_info[col].maint_margin

    low_var = bt_args.var[col][0]
    high_var = bt_args.var[col][1]
    SAFETY_BUFFER = 3

    current_price = c.close[c.i, col]
    current_leg_direction = directions[col]

    if current_leg_direction == NEUTRAL:
        log(bt_args.logging, "Not in market", "col=", col)

        #                  Nominal amount          Leverage factor
        # trade_size = -(trade_value / price) * (target_vol / volatility)
        leverage_factor = 1 / ((max(abs(bt_args.var[trade_col][0]), abs(bt_args.var[trade_col][1])) * SAFETY_BUFFER) + maint_margin)
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
            "col=", col,
            "leg_dir=", current_leg_direction,
            "cur_pos=", current_position,
            "liq_price=", liq_price,
        )


        if (current_leg_direction == LONG and current_price < liq_price) or (
            current_leg_direction == SHORT and current_price > liq_price
        ):
            log(bt_args.logging, "liquidated at price:", current_price)
            return col, nb.close_position_nb()

        if current_spread_direction == LONG and current_zscore >= -bt_args.zscore_exit_thresh:
            log(bt_args.logging, "Closing a long spread")
            return col, nb.order_nb(
                size=-current_position,
                price=c.close[c.i, col],
                fees=fee_pct,
                fixed_fees=fee_fixed,
                raise_reject=True,
            )

        elif current_spread_direction == SHORT and current_zscore <= bt_args.zscore_exit_thresh:
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
def post_order_func_nb(c, directions, spread_dir, call_seq_out, *args):
    bt_args = BacktestArgs(*args)
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


@njit
def post_segment_func_nb(c, directions, spread_dir, call_seq_out):
    return None


def from_order_func_wrapper(close_prices: np.ndarray, func_args: Tuple, initial_cash: int = 150000):
    return vbt.Portfolio.from_order_func(
        close_prices,
        flex_order_func_nb,
        *func_args,
        group_by=True,
        cash_sharing=True,
        init_cash=initial_cash,
        flexible=True,
        pre_group_func_nb=pre_group_func_nb,
        pre_group_args=func_args,
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=func_args,
        post_segment_func_nb=post_segment_func_nb,
        post_order_func_nb=post_order_func_nb,
        post_order_args=func_args,
        freq="1d",
    )
