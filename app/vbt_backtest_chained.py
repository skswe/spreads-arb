from collections import namedtuple

import numpy as np
import vectorbt as vbt
import vectorbt.portfolio.enums as enums
import vectorbt.portfolio.nb as nb
from numba import njit, prange
from pyutil.dicts import format_dict

NONE = -999999
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
def sum_axis_1(arr):
    res = []
    for i in prange(arr.shape[0]):
        res.append(arr[:, i].sum())
    return np.array(res)


@njit
def get_max_var(a, col):
    res = []
    for i in prange(len(a)):
        res.append(max(a[i, col, :]))
    return np.array(res)


@njit
def sort_call_seq_for_entry(c, call_seq_out, best_spread_idx, bt_args):
    log(bt_args.logging, "    sorting call_seq for entry:")
    current_zscore = bt_args.zscore[c.i, best_spread_idx]
    first_half = np.arange(int(c.group_len / 2))
    second_half = np.arange(int(c.group_len / 2), c.group_len)

    potential_short_entry = current_zscore > bt_args.z_score_thresholds[1]
    log(bt_args.logging, "    current_zscore:", current_zscore)
    log(bt_args.logging, "    potential_short_entry:", potential_short_entry)
    if potential_short_entry:
        # col_1 goes short, so process first to get short cash
        call_seq_out[:] = np.concatenate((second_half, first_half))
    else:  # potential long entry
        # col_0 goes short, so process first to get short cash
        call_seq_out[:] = np.concatenate((first_half, second_half))
    log(bt_args.logging, "    resulting call_seq:", call_seq_out)


@njit
def sort_call_seq_for_exit(c, call_seq_out, bt_args):
    log(bt_args.logging, "    sorting call_seq for exit")
    short_column = (c.last_position).argmin()
    log(bt_args.logging, "    short column:", short_column)

    first_half = np.arange(int(c.group_len / 2))
    second_half = np.arange(int(c.group_len / 2), c.group_len)

    if short_column < int(c.group_len / 2):
        # col_0 was short, so col_1 goes short to exit
        call_seq_out[:] = np.concatenate((second_half, first_half))
    else:
        # col_1 was short, so col_0 goes short to exit
        call_seq_out[:] = np.concatenate((first_half, second_half))
    log(bt_args.logging, "    resulting call_seq:", call_seq_out)


@njit
def determine_spread_direction(current_zscore, outer_band_threshold):
    if abs(current_zscore) > outer_band_threshold:
        if current_zscore > 0:
            return SHORT
        else:
            return LONG
    return NONE


@njit
def index_both_legs(array, spread_idx):
    assert int(spread_idx) == spread_idx
    return np.array([array[int(spread_idx)], array[int(spread_idx + len(array) / 2)]])


@njit
def get_entry_orders(c, funding_debit, best_spread_idx, bt_args):
    """Determines the size and direction of both legs of the spread when the zscore is outside the outer band"""
    log(bt_args.logging, "    +++ determining entry orders +++")

    current_zscore = bt_args.zscore[c.i, best_spread_idx]
    current_prices = index_both_legs(c.close[c.i], best_spread_idx)

    fee_pct = bt_args.fee_pct[best_spread_idx]
    fee_fixed = bt_args.fee_fixed[best_spread_idx]
    slippage = bt_args.bid_ask_spread[c.i, best_spread_idx] / current_prices
    init_margin = bt_args.init_margin[best_spread_idx]
    maint_margin = bt_args.maint_margin[best_spread_idx]
    var = bt_args.var[best_spread_idx]
    max_vars = np.array([max(np.abs(var[0])), max(np.abs(var[1]))])

    leverage_factors = 1 / ((max_vars * MARGIN_SAFETY_BUFFER) + maint_margin)
    trade_sizes = (bt_args.trade_value / current_prices) * leverage_factors
    direction = determine_spread_direction(current_zscore, bt_args.z_score_thresholds[1])
    trade_sizes = direction * np.array([-1, 1]) * trade_sizes

    short_leg = np.argmin(trade_sizes)
    long_leg = np.argmax(trade_sizes)

    log(bt_args.logging, "    direction:", direction, direction_str_map(direction))
    log(bt_args.logging, "    trade_sizes:", trade_sizes, "short_leg:", short_leg, "long_leg", long_leg)

    return (
        nb.order_nb(
            size=trade_sizes[short_leg],
            price=current_prices[short_leg],
            fees=fee_pct[short_leg],
            fixed_fees=fee_fixed[short_leg] + funding_debit[short_leg],
            slippage=slippage[short_leg],
            raise_reject=False,
            allow_partial=False,
        ),
        nb.order_nb(
            size=trade_sizes[long_leg],
            price=current_prices[long_leg],
            fees=fee_pct[long_leg],
            fixed_fees=fee_fixed[long_leg] + funding_debit[long_leg],
            slippage=slippage[long_leg],
            raise_reject=False,
            allow_partial=False,
        ),
    )


@njit
def get_exit_orders(c, funding_debit, held_spread_idx, bt_args):
    log(bt_args.logging, "    --- determining exit orders ---")
    fee_pct = bt_args.fee_pct[held_spread_idx[0]]
    fee_fixed = bt_args.fee_fixed[held_spread_idx[0]]
    current_prices = index_both_legs(c.close[c.i], held_spread_idx[0])
    slippage = bt_args.bid_ask_spread[c.i, held_spread_idx[0]] / current_prices
    sizes = index_both_legs(c.last_position, held_spread_idx[0])

    short_leg = np.argmin(-sizes)
    long_leg = np.argmax(-sizes)

    log(bt_args.logging, "    current_positions:", sizes)
    log(bt_args.logging, "    short_leg:", short_leg, "long_leg:", long_leg)

    return (
        nb.order_nb(
            size=-sizes[short_leg],
            price=current_prices[short_leg],
            fees=fee_pct[short_leg],
            fixed_fees=fee_fixed[short_leg] + funding_debit[short_leg],
            slippage=slippage[short_leg],
            raise_reject=False,
        ),
        nb.order_nb(
            size=-sizes[long_leg],
            price=current_prices[long_leg],
            fees=fee_pct[long_leg],
            fixed_fees=fee_fixed[long_leg] + funding_debit[long_leg],
            slippage=slippage[long_leg],
            raise_reject=False,
        ),
    )


@njit
def get_expected_profit_vectorized(c, bt_args):
    if not bt_args.profitable_only:
        return np.array([1] * c.group_len, dtype=np.float_)

    current_zscore = bt_args.zscore[c.i]
    current_prices = np.stack((c.close[c.i, : int(c.group_len / 2)], c.close[c.i, int(c.group_len / 2) :])).T
    final_price = np.expand_dims(np.sum(current_prices, axis=1) / 2, 1)

    fee_pct = bt_args.fee_pct
    fee_fixed = bt_args.fee_fixed
    slippage = bt_args.bid_ask_spread[c.i] / current_prices
    init_margin = bt_args.init_margin
    maint_margin = bt_args.maint_margin
    var = bt_args.var
    max_vars = np.stack((get_max_var(np.abs(var), 0), get_max_var(np.abs(var), 1))).T

    # debug_idx = 184
    # print("current_zscore", current_zscore.shape, current_zscore[debug_idx])
    # print("current_prices", current_prices.shape, current_prices[debug_idx])
    # print("final_price", final_price.shape, final_price[debug_idx])
    # print("fee_pct", fee_pct.shape, fee_pct[debug_idx])
    # print("fee_fixed", fee_fixed.shape, fee_fixed[debug_idx])
    # print("slippage", slippage.shape, slippage[debug_idx])
    # print("maint_margin", maint_margin.shape, maint_margin[debug_idx])
    # print("var", var.shape, var[debug_idx])
    # print("max_vars", max_vars.shape, max_vars[debug_idx])

    leverage_factors = 1 / ((max_vars * MARGIN_SAFETY_BUFFER) + maint_margin)
    trade_sizes = (bt_args.trade_value / current_prices) * leverage_factors
    direction = np.where(np.abs(current_zscore) > bt_args.z_score_thresholds[1], -np.sign(current_zscore), 0)
    direction = np.expand_dims(direction, 1)
    dirs = np.stack((np.array([-1] * len(trade_sizes)), np.array([1] * len(trade_sizes)))).T
    trade_sizes = direction * dirs * trade_sizes

    # print("leverage_factor", leverage_factors.shape, leverage_factors[debug_idx])
    # print("trade_sizes", trade_sizes.shape, trade_sizes[debug_idx])
    # print("direction", direction.shape, direction[debug_idx])
    # print("dirs", dirs.shape, dirs[debug_idx])

    raw_profit = np.sum(trade_sizes * (final_price - current_prices), axis=1)
    entry_fees = np.abs(trade_sizes) * current_prices * fee_pct + fee_fixed
    exit_fees = np.abs(trade_sizes) * final_price * fee_pct + fee_fixed
    total_fees = np.sum(entry_fees, axis=1) + np.sum(exit_fees, axis=1)
    entry_slippage = np.abs(trade_sizes) * current_prices * slippage
    exit_slippage = np.abs(trade_sizes) * final_price * slippage
    total_slippage = np.sum(entry_slippage, axis=1) + np.sum(exit_slippage, axis=1)
    final_profit = (raw_profit - total_fees - total_slippage) / EXPECTED_PROFIT_FACTOR

    # print("raw_profit", raw_profit.shape, raw_profit[debug_idx])
    # print("entry_fees", entry_fees.shape, entry_fees[debug_idx])
    # print("exit_fees", exit_fees.shape, exit_fees[debug_idx])
    # print("total_fees", total_fees.shape, total_fees[debug_idx])
    # print("entry_slippage", entry_slippage.shape, entry_slippage[debug_idx])
    # print("exit_slippage", exit_slippage.shape, exit_slippage[debug_idx])
    # print("total_slippage", total_slippage.shape, total_slippage[debug_idx])
    # print("final_profit", final_profit.shape, final_profit[debug_idx])

    return final_profit


@njit
def pre_group_func_nb(c, bt_args):
    """Called before processing a group. (Group contains both legs of spread)"""
    directions = np.array([NEUTRAL] * c.group_len, dtype=np.int_)
    spread_dir = np.array([NEUTRAL])
    funding_debit = np.array([0] * c.group_len, dtype=np.float_)
    call_seq_out = np.arange(c.group_len)
    held_spread_idx = np.array([NONE])
    log(bt_args.logging, "pre_group_func_nb global variables -- ")
    log(
        bt_args.logging,
        "directions",
        directions.shape,
        "|spread_dir",
        spread_dir,
        "|funding_debit",
        funding_debit.shape,
        "|call_seq_out",
        call_seq_out.shape,
        "|held_spread_idx",
        held_spread_idx,
    )
    return (directions, spread_dir, funding_debit, call_seq_out, held_spread_idx)


@njit
def pre_segment_func_nb(c, directions, spread_dir, funding_debit, call_seq_out, held_spread_idx, bt_args):
    """Prepare call_seq for either entering or exiting a position. On entry, col_0 goes short, on exit, col_1 goes short.
    Called before segment is processed. Segment refers to a single time step within a group"""

    log(bt_args.logging, "-" * 100)
    log(
        bt_args.logging,
        "pre_segment_func_nb\n",
        ">>time_idx",
        c.i,
        "|in_market:",
        spread_dir[0] != NEUTRAL,
        "|spread_dir:",
        spread_dir[0],
        "|cash:",
        c.last_cash,
        "|value",
        c.last_value,
        "|last_return",
        c.last_return,
        "\n|funding_debit",
        funding_debit,
        "|call_seq_out",
        call_seq_out,
    )

    expected_profits = get_expected_profit_vectorized(c, bt_args)
    good_z_score_mask = (np.abs(bt_args.zscore[c.i]) > bt_args.z_score_thresholds[0]).astype(np.int_)
    expected_profits = expected_profits * good_z_score_mask
    ep_indices = np.argsort(expected_profits)

    log(bt_args.logging, "expected_profits", expected_profits.shape, expected_profits[ep_indices][-5:])
    log(bt_args.logging, "expected_profits_indices", ep_indices[-5:])

    if sum(expected_profits > 0) > 0:
        best_spread_idx = np.argmax(expected_profits)
    else:
        best_spread_idx = NONE
    log(bt_args.logging, "BEST SPREAD INDEX=", best_spread_idx)
    log(bt_args.logging, "directions that are not zero", np.nonzero(directions)[0])

    if best_spread_idx != NONE:
        best_spread_zscore = bt_args.zscore[c.i, best_spread_idx]
        best_spread_expected_profit = expected_profits[best_spread_idx]
        best_spread_close = index_both_legs(c.close[c.i], best_spread_idx)
        best_spread_position = index_both_legs(c.last_position, best_spread_idx)
        best_spread_bas = bt_args.bid_ask_spread[c.i, best_spread_idx]
        best_spread_fr = bt_args.funding_rate[c.i, best_spread_idx]
        log(
            bt_args.logging,
            "best_spread_idx",
            best_spread_idx,
            "|best_spread_expected_profit",
            best_spread_expected_profit,
            "|best_spread_zscore",
            best_spread_zscore,
            "|best_spread_close",
            best_spread_close,
            "|best_spread_position",
            best_spread_position,
            "best_spread_bas",
            best_spread_bas,
            "best_spread_fr",
            best_spread_fr,
        )

    if held_spread_idx[0] != NONE:
        held_spread_zscore = bt_args.zscore[c.i, held_spread_idx[0]]
        held_spread_close = index_both_legs(c.close[c.i], held_spread_idx[0])
        held_spread_position = index_both_legs(c.last_position, held_spread_idx[0])
        held_spread_bas = bt_args.bid_ask_spread[c.i, held_spread_idx[0]]
        held_spread_fr = bt_args.funding_rate[c.i, held_spread_idx[0]]
        held_spread_directions = index_both_legs(directions, held_spread_idx[0])
        log(
            bt_args.logging,
            "held_spread_idx",
            held_spread_idx[0],
            "|held_spread_zscore",
            held_spread_zscore,
            "|held_spread_close",
            held_spread_close,
            "|held_spread_position",
            held_spread_position,
            "held_spread_bas",
            held_spread_bas,
            "held_spread_fr",
            held_spread_fr,
            "held_spread_directions",
            held_spread_directions,
        )

    liquidated = False
    action = NO_ACTION
    entry_orders = (NO_ORDER, NO_ORDER)
    exit_orders = (NO_ORDER, NO_ORDER)

    if c.last_cash == 0:
        log(bt_args.logging, "***** Ran out of cash ******")
        pass

    # Entry logic
    elif spread_dir[0] == NEUTRAL and best_spread_idx != NONE:
        action = ENTRY
        sort_call_seq_for_entry(c, call_seq_out, best_spread_idx, bt_args)
        entry_orders = get_entry_orders(c, funding_debit, best_spread_idx, bt_args)

    # Exit logic
    elif held_spread_idx[0] != NONE:
        # Pay funding when position is held
        update_funding_debit(c, funding_debit, bt_args)

        # check for liquidation
        var = bt_args.var[best_spread_idx]
        max_vars = np.array([max(np.abs(var[0])), max(np.abs(var[1]))])
        liq_prices = held_spread_close * (1 + (max_vars * MARGIN_SAFETY_BUFFER) * -held_spread_directions)
        for i in range(len(liq_prices)):
            if (held_spread_directions[i] == LONG and held_spread_close[i] < liq_prices[i]) or (
                held_spread_directions[i] == SHORT and held_spread_close[i] > liq_prices[i]
            ):
                liquidated = True
                log(bt_args.logging, "liquidated")
                action = EXIT
                exit_orders = get_exit_orders(c, funding_debit, held_spread_idx, bt_args)
                break

        if (
            not liquidated
            and bt_args.zscore[c.i, held_spread_idx[0]] * bt_args.zscore[c.i - 1, held_spread_idx[0]] <= 0
        ):  # z score crossed or touched 0
            sort_call_seq_for_exit(c, call_seq_out, bt_args)
            if best_spread_idx != NONE:
                action = EXIT_AND_ENTRY
                entry_orders = get_entry_orders(c, funding_debit, best_spread_idx, bt_args)
                exit_orders = get_exit_orders(c, funding_debit, held_spread_idx, bt_args)
            else:
                action = EXIT
                exit_orders = get_exit_orders(c, funding_debit, held_spread_idx, bt_args)

    log(bt_args.logging, "action:", action_str_map(action))

    # print("exit_orders", exit_orders[0], "\n", exit_orders[1])
    # print("entry_orders", entry_orders[0], "\n", entry_orders[1])

    return (
        directions,
        spread_dir,
        funding_debit,
        call_seq_out,
        held_spread_idx,
        best_spread_idx,
        action,
        entry_orders,
        exit_orders,
    )


@njit
def update_funding_debit(c, funding_debit, bt_args):
    """Update outstanding funding rate fees for `col` to debit from cash via fixed_fees on next order"""
    for i in range(int(c.group_len / 2)):
        funding_rate = bt_args.funding_rate[c.i, i]
        last_position = index_both_legs(c.last_position, i)
        prices = index_both_legs(c.close[c.i], i)
        funding_fee = funding_rate[0] * last_position[0] * prices[0] + funding_rate[1] * last_position[1] * prices[1]
        funding_debit[i] += funding_fee
        if funding_fee > 0:
            log(bt_args.logging, "added funding debit", funding_fee)


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
    c,
    directions,
    spread_dir,
    funding_debit,
    call_seq_out,
    held_spread_idx,
    best_spread_idx,
    action,
    entry_orders,
    exit_orders,
    bt_args,
):

    # Short circuit exit if there is no action
    if action == NO_ACTION:
        return -1, NO_ORDER

    col = call_seq_out[c.call_idx % c.group_len]

    if action == EXIT_AND_ENTRY and c.call_idx >= 2 * c.group_len:
        log(bt_args.logging, "break loop for exit AND entry")
        return -1, NO_ORDER
    elif action != EXIT_AND_ENTRY and c.call_idx >= c.group_len:
        log(bt_args.logging, "break loop for exit OR entry")
        return -1, NO_ORDER

    entry_indices = [best_spread_idx, int(c.group_len / 2) + best_spread_idx]
    exit_indices = [held_spread_idx[0], int(c.group_len / 2) + held_spread_idx[0]]

    if action == ENTRY:
        if col not in entry_indices:
            return col, NO_ORDER
        if (directions != NEUTRAL).sum() == 0:
            log(bt_args.logging, "flex_order_func_nb: // Entering on short column", "col=", col)
            # First entry use short column
            # log(bt_args.logging, "entry short column", col, entry_orders[0])
            return col, entry_orders[0]
        else:
            log(bt_args.logging, "flex_order_func_nb: // Entering on long column", "col=", col)
            # Subsequent entries use long column
            # log(bt_args.logging, "entry long column", col, entry_orders[1])
            return col, entry_orders[1]

    elif action == EXIT:
        if col not in exit_indices:
            return col, NO_ORDER
        if (directions != NEUTRAL).sum() == 2:
            log(bt_args.logging, "flex_order_func_nb: // Exiting on short column", "col=", col)
            # First exit use short column
            # log(bt_args.logging, "exit short column", col, exit_orders[0])
            return col, entry_orders[0]
        else:
            log(bt_args.logging, "flex_order_func_nb: // Exiting on long column", "col=", col)
            # Subsequent exit use long column
            # log(bt_args.logging, "exit long column", col, exit_orders[1])
            return col, entry_orders[1]

    elif action == EXIT_AND_ENTRY:
        if c.call_idx < c.group_len:
            # Exit side
            if col not in exit_indices:
                return col, NO_ORDER

            if (directions != NEUTRAL).sum() == 2:
                log(bt_args.logging, "flex_order_func_nb: // Exiting on short column", "col=", col)
                # First exit use short column
                # log(bt_args.logging, "exit short column", col, exit_orders[0])
                return col, exit_orders[0]
            else:
                log(bt_args.logging, "flex_order_func_nb: // Exiting on long column", "col=", col)
                # log(bt_args.logging, "exit long column", col, exit_orders[1])
                return col, exit_orders[1]
        else:
            # Entry side
            if col not in entry_indices:
                return col, NO_ORDER
            if (directions != NEUTRAL).sum() == 0:
                log(bt_args.logging, "flex_order_func_nb: // Entering on short column", "col=", col)
                # First entry use short column
                # log(bt_args.logging, "entry short column", col, entry_orders[0])
                return col, entry_orders[0]
            else:
                log(bt_args.logging, "flex_order_func_nb: // Entering on long column", "col=", col)
                # log(bt_args.logging, "entry long column", col, entry_orders[1])
                return col, entry_orders[1]

    return -1, NO_ORDER


@njit
def post_order_func_nb(
    c,
    directions,
    spread_dir,
    funding_debit,
    call_seq_out,
    held_spread_idx,
    best_spread_idx,
    action,
    entry_orders,
    exit_orders,
    bt_args,
):
    if c.order_result.status == enums.OrderStatus.Filled:
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
            "slippage=",
            abs(c.order_result.price - c.close[c.i, c.col]) * c.order_result.size,
            ")",
        )
        if directions[c.col] == NEUTRAL:
            # Entering a position
            directions[c.col] = LONG if c.order_result.side == enums.OrderSide.Buy else SHORT

            held_spread_idx[0] = best_spread_idx
            if directions[best_spread_idx] == LONG:
                spread_dir[0] = SHORT
            else:
                spread_dir[0] = LONG

        else:
            # Closing a position
            directions[c.col] = NEUTRAL
            if (directions != NEUTRAL).sum() == 0:
                # Finishing closing a spread position
                # sort call_seq to favor shorts first on potential entry in same time_idx

                if best_spread_idx != NONE:
                    sort_call_seq_for_entry(c, call_seq_out, best_spread_idx, bt_args)

                # reset held idx
                held_spread_idx[0] = NONE

        # Reset funding debit for the filled column
        funding_debit[c.col] = 0

        log(
            bt_args.logging,
            "post_order_func_nb: // directions that are nonzero after order",
            np.nonzero(directions)[0],
            directions[np.nonzero(directions)[0]],
        )
    elif c.order_result.status == enums.OrderStatus.Ignored:
        pass
    else:
        log(bt_args.logging, "order rejected")
        log(bt_args.logging, c.order_result.status)
        log(bt_args.logging, c.order_result.status_info)
    return None


def from_order_func_wrapper_chained(close_prices: np.ndarray, bt_args: BacktestArgs):
    log(bt_args.logging, "Run backtest with the following settings:")
    log(
        bt_args.logging,
        format_dict(
            {
                **bt_args._asdict(),
                "close": close_prices.shape,
                "var": bt_args.var.shape,
                "init_margin": bt_args.init_margin.shape,
                "maint_margin": bt_args.maint_margin.shape,
                "fee_pct": bt_args.fee_pct.shape,
                "fee_fixed": bt_args.fee_fixed.shape,
                "zscore": bt_args.zscore.shape,
                "funding_rate": bt_args.funding_rate.shape,
                "bid_ask_spread": bt_args.bid_ask_spread.shape,
                "zscore": bt_args.zscore.shape,
            },
        ),
    )
    # log(True, bt_args.maint_margin[0])
    # log(bt_args.logging, "=" * 100 + "\n\n")
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
        pre_group_args=(bt_args,),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(bt_args,),
        post_order_func_nb=post_order_func_nb,
        post_order_args=(bt_args,),
        freq="1d",
        update_value=True,
    )
