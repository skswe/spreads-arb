from collections import namedtuple

import numpy as np
import vectorbt as vbt
import vectorbt.portfolio.enums as enums
import vectorbt.portfolio.nb as nb
from numba import njit
from pyutil.dicts import format_dict

NONE = -999
LONG = 1
SHORT = -1
NEUTRAL = 0


@njit
def direction_str_map(direction):
    if direction == LONG:
        return "long"
    elif direction == SHORT:
        return "short"
    elif direction == NEUTRAL:
        return "neutral"
    return ""


ENTRY = 10
EXIT = 11
EXIT_AND_ENTRY = 12
NO_ACTION = 13


@njit
def action_str_map(action):
    if action == ENTRY:
        return "entry"
    elif action == EXIT:
        return "exit"
    elif action == EXIT_AND_ENTRY:
        return "exit_and_entry"
    elif action == NO_ACTION:
        return "no_action"
    return ""


MARGIN_SAFETY_BUFFER = 3
EXPECTED_PROFIT_FACTOR = 1.1

NO_ORDER = nb.order_nothing_nb()

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
        "profitable_only",
    ],
)


@njit
def log(logging, *msg):
    if logging:
        print(*msg)


@njit
def sort_call_seq_for_entry(c, call_seq_out, bt_args):
    current_zscore = bt_args.zscore[c.i]
    potential_short_entry = current_zscore > bt_args.z_score_thresholds[1]
    if potential_short_entry:
        # col_1 goes short, so process first to get short cash
        call_seq_out[:] = np.array([1, 0])
    else:  # potential long entry
        # col_0 goes short, so process first to get short cash
        call_seq_out[:] = np.array([0, 1])


@njit
def sort_call_seq_for_exit(c, call_seq_out):
    # Reverse sort last position to close the long position first
    call_seq_out[:] = (-c.last_position).argsort()


@njit
def determine_spread_direction(current_zscore, outer_band_threshold):
    if abs(current_zscore) > outer_band_threshold:
        if current_zscore > 0:
            return SHORT
        else:
            return LONG
    return NONE


@njit
def get_entry_orders(c, funding_debit, bt_args):
    """Determines the size and direction of both legs of the spread when the zscore is outside the outer band"""
    current_zscore = bt_args.zscore[c.i]

    fee_pct = bt_args.fee_pct
    fee_fixed = bt_args.fee_fixed
    slippage = bt_args.bid_ask_spread[c.i] / c.close[c.i]
    init_margin = bt_args.init_margin
    maint_margin = bt_args.maint_margin
    max_vars = np.array([max(np.abs(np.array(bt_args.var[0]))), max(np.abs(np.array(bt_args.var[1])))])

    leverage_factors = 1 / ((max_vars * MARGIN_SAFETY_BUFFER) + np.array(maint_margin))
    trade_sizes = (bt_args.trade_value / c.close[c.i]) * leverage_factors
    direction = determine_spread_direction(current_zscore, bt_args.z_score_thresholds[1])
    trade_sizes = direction * np.array([-1, 1]) * trade_sizes

    log(bt_args.logging, direction_str_map(direction))

    return (
        nb.order_nb(
            size=trade_sizes[0],
            price=c.close[c.i, 0],
            fees=fee_pct[0],
            fixed_fees=fee_fixed[0] + funding_debit[0],
            slippage=slippage[0],
            raise_reject=False,
            allow_partial=False,
        ),
        nb.order_nb(
            size=trade_sizes[1],
            price=c.close[c.i, 1],
            fees=fee_pct[1],
            fixed_fees=fee_fixed[1] + funding_debit[1],
            slippage=slippage[1],
            raise_reject=False,
            allow_partial=False,
        ),
    )


@njit
def get_exit_orders(c, funding_debit, bt_args):
    fee_pct = bt_args.fee_pct
    fee_fixed = bt_args.fee_fixed
    slippage = bt_args.bid_ask_spread[c.i] / c.close[c.i]

    return (
        nb.order_nb(
            size=-c.last_position[0],
            price=c.close[c.i, 0],
            fees=fee_pct[0],
            fixed_fees=fee_fixed[0] + funding_debit[0],
            slippage=slippage[0],
            raise_reject=False,
        ),
        nb.order_nb(
            size=-c.last_position[1],
            price=c.close[c.i, 1],
            fees=fee_pct[1],
            fixed_fees=fee_fixed[1] + funding_debit[1],
            slippage=slippage[1],
            raise_reject=False,
        ),
    )


@njit
def get_expected_profit(c, bt_args):
    if not bt_args.profitable_only:
        return 1
    
    current_zscore = bt_args.zscore[c.i]
    current_prices = c.close[c.i]
    final_price = sum(c.close[c.i]) / 2

    fee_pct = bt_args.fee_pct
    fee_fixed = bt_args.fee_fixed
    slippage = bt_args.bid_ask_spread[c.i] / c.close[c.i]
    init_margin = bt_args.init_margin
    maint_margin = bt_args.maint_margin
    max_vars = np.array([max(np.abs(np.array(bt_args.var[0]))), max(np.abs(np.array(bt_args.var[1])))])

    leverage_factors = 1 / ((max_vars * MARGIN_SAFETY_BUFFER) + np.array(maint_margin))
    trade_sizes = (bt_args.trade_value / c.close[c.i]) * leverage_factors
    direction = determine_spread_direction(current_zscore, bt_args.z_score_thresholds[1])
    trade_sizes = direction * np.array([-1, 1]) * trade_sizes

    raw_profit = sum(trade_sizes * (final_price - current_prices))
    entry_fees = np.abs(trade_sizes) * current_prices * np.array(fee_pct) + np.array(fee_fixed)
    exit_fees = np.abs(trade_sizes) * final_price * np.array(fee_pct) + np.array(fee_fixed)
    entry_slippage = np.abs(trade_sizes) * current_prices * slippage
    exit_slippage = np.abs(trade_sizes) * final_price * slippage
    final_profit = (
        raw_profit - sum(entry_fees) - sum(entry_slippage) - sum(exit_fees) - sum(exit_slippage)
    ) / EXPECTED_PROFIT_FACTOR

    return final_profit


@njit
def pre_group_func_nb(c):
    """Called before processing a group. (Group contains both legs of spread)"""
    directions = np.array([NEUTRAL, NEUTRAL], dtype=np.int_)
    spread_dir = np.array([NEUTRAL])
    funding_debit = np.array([0, 0])
    call_seq_out = np.arange(c.group_len)
    return (directions, spread_dir, funding_debit, call_seq_out)


@njit
def pre_segment_func_nb(c, directions, spread_dir, funding_debit, call_seq_out, bt_args):
    """Prepare call_seq for either entering or exiting a position. On entry, col_0 goes short, on exit, col_1 goes short.
    Called before segment is processed. Segment refers to a single time step within a group"""

    log(
        bt_args.logging,
        "\ntime_idx",
        c.i,
        "|in_market:",
        spread_dir[0] != NEUTRAL,
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
        "|cash:",
        c.last_cash,
        "|value",
        c.last_value,
        "|last_return",
        c.last_return,
        "|bid_ask_spread",
        bt_args.bid_ask_spread[c.i],
        "|funding_rate",
        bt_args.funding_rate[c.i],
        "|funding_debit",
        funding_debit,
        "|call_seq_out",
        call_seq_out,
    )

    action = NO_ACTION
    liquidated = False
    entry_orders = (NO_ORDER, NO_ORDER)
    exit_orders = (NO_ORDER, NO_ORDER)
    
    if c.last_cash == 0:
        # Ran out of cash
        pass

    # Entry logic
    elif spread_dir[0] == NEUTRAL:
        if abs(bt_args.zscore[c.i]) > bt_args.z_score_thresholds[1] and get_expected_profit(c, bt_args) > 0:
            action = ENTRY
            sort_call_seq_for_entry(c, call_seq_out, bt_args)
            entry_orders = get_entry_orders(c, funding_debit, bt_args)

    # Exit logic
    else:
        # Pay funding when position is held
        update_funding_debit(c, 0, funding_debit, bt_args)
        update_funding_debit(c, 1, funding_debit, bt_args)

        # check for liquidation
        max_vars = np.array(
            [max(np.abs(np.array(bt_args.var[0]))), max(np.abs(np.array(bt_args.var[1])))]
        )  # var = value_at_risk
        liq_prices = c.close[c.i] * (1 + (max_vars * MARGIN_SAFETY_BUFFER) * -directions)
        for i in range(len(liq_prices)):
            if (directions[i] == LONG and c.close[c.i, i] < liq_prices[i]) or (
                directions[i] == SHORT and c.close[c.i, i] > liq_prices[i]
            ):
                liquidated = True
                log(bt_args.logging, "liquidated")
                action = EXIT
                exit_orders = get_exit_orders(c, funding_debit, bt_args)
                break

        if not liquidated and bt_args.zscore[c.i] * bt_args.zscore[c.i - 1] <= 0:  # z score crossed or touched 0
            sort_call_seq_for_exit(c, call_seq_out)
            if abs(bt_args.zscore[c.i]) > (bt_args.z_score_thresholds[1]) and get_expected_profit(c, bt_args) > 0:
                action = EXIT_AND_ENTRY
                entry_orders = get_entry_orders(c, funding_debit, bt_args)
                exit_orders = get_exit_orders(c, funding_debit, bt_args)
            else:
                action = EXIT
                exit_orders = get_exit_orders(c, funding_debit, bt_args)

    log(bt_args.logging, "action:", action_str_map(action), "|call_seq_out", call_seq_out)

    return (
        directions,
        spread_dir,
        funding_debit,
        call_seq_out,
        action,
        entry_orders,
        exit_orders,
    )


@njit
def update_funding_debit(c, col, funding_debit, bt_args):
    """Update outstanding funding rate fees for `col` to debit from cash via fixed_fees on next order"""
    funding_rate_debit = bt_args.funding_rate[c.i, col] * c.last_position[col] * c.close[c.i, col]
    log(bt_args.logging, "funding rate debit (col=", col, "):", funding_rate_debit)
    funding_debit[col] += funding_rate_debit


#                                          1
# leverage_factor =   --------------------------------------------
#                     mmr + max(low_var, high_var) * SAFETY_BUFFER

#                          trade_value * target_vol
# trade_size =   --------------------------------------------
#                             price * volatility

# Nominal amount: trade_value / price
# leverage_factor: target_vol / volatility


@njit
def flex_order_func_nb(
    c, directions, spread_dir, funding_debit, call_seq_out, action, entry_orders, exit_orders, bt_args
):
    col = call_seq_out[c.call_idx % c.group_len]

    if action == EXIT_AND_ENTRY and c.call_idx >= 2 * c.group_len:
        # Break loop for exit and entry
        return -1, nb.order_nothing_nb()
    elif action != EXIT_AND_ENTRY and c.call_idx >= c.group_len:
        # Break loop for all other actions
        return -1, nb.order_nothing_nb()

    if action == ENTRY:
        return col, entry_orders[col]

    elif action == EXIT:
        return col, exit_orders[col]

    elif action == EXIT_AND_ENTRY:
        if c.call_idx < c.group_len:
            return col, exit_orders[col]
        else:
            return col, entry_orders[col]

    return -1, nb.order_nothing_nb()

    # expected_profit = determine_expected_profit(direction, c.close[c.i], trade_size, bt_args.fee_pct, bt_args.bid_ask_spread[c.i] / c.close[c.i], bt_args.ev_threshold)
    # if expected_profit > bt_args.ev_threshold:
    #     log(bt_args.logging, "entering market with expected profit:", expected_profit, "| direction:", direction)


@njit
def post_order_func_nb(
    c, directions, spread_dir, funding_debit, call_seq_out, action, entry_orders, exit_orders, bt_args
):
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
            "slippage=",
            abs(c.order_result.price - c.close[c.i, c.col]) * c.order_result.size,
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

        # Reset funding debit for the filled column
        funding_debit[c.col] = 0
    else:
        log(bt_args.logging, "order rejected")
        log(bt_args.logging, c.order_result.status)
        log(bt_args.logging, c.order_result.status_info)
    return None


def run(close_prices: np.ndarray, bt_args: BacktestArgs):
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
        # max_orders = at most a buy/sell on each leg = 4 orders per tick
        max_orders=close_prices.shape[0] * 4,
        pre_group_func_nb=pre_group_func_nb,
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(bt_args,),
        post_order_func_nb=post_order_func_nb,
        post_order_args=(bt_args,),
        freq="1d",
        update_value=True,
    )
