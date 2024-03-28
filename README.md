# Crypto Perpetual Futures Spread Arbitrage Strategy

This module contains a series of experiments that simulate an algorithmic
trading strategy designed to capitalize on spread differentials between two
underlying assets on different exchanges.

## Introduction

The strategy primarily revolves around "spreads," which represent the difference
in price between two identical underlying assets on two distinct exchanges.
These spreads are treated as individual financial instruments, allowing for long
or short positions to be taken based on their respective pricing differentials.

## Strategy Overview

The trading strategy implemented in this module operates based on the z-score of
the spread. Specifically, it involves:

- **Going Long:** Opening a long position on the spread when its z-score falls
  below a certain threshold.
- **Going Short:** Opening a short position on the spread when its z-score
  exceeds a certain threshold.
- **Position Closure:** Closing the long or short position when the z-score
  touches or crosses zero.

## Experimentation

### Individual Spread Backtesting

The initial experiments involved testing the strategy on individual spreads
using daily OHLCV data. Leveraging the `vectorbt` module for efficient
computation, these experiments provided insights into the strategy's performance
on specific spreads. The base backtest is provided in
`spread_arb.vbt_backtest.vbt_backtest`. It can be run using the `backtest.ipynb`
notebook.

### Spread Chaining

Building upon successful results from individual spread testing, the next set of
experiments focused on simulating the strategy across all available spreads in
the market. This approach, termed spread chaining, involves selecting the spread
with the highest nonzero z-score at each time step, effectively chaining
together optimal trades across the market. This experiment is implemented in
`spread_arb.vbt_backtest.vbt_backtest_chained`. It can be run using the
`backtest_chained.ipynb` notebook.

### Order Book Quote Data Testing

To simulate real market conditions with granular order filling, experiments were
conducted using order book quote data instead of OHLCV bars. While the initial
results from this approach were promising, subsequent testing revealed poorer
performance. The code for this backtest can be found in
`spread_arb.vbt_backtest.vbt_bt_quotes`. It can be run using the
`backtest_quotes.ipynb` notebook.

### Granular Order Book Quote Analysis

In a bid to confirm the absence of alpha in the strategy, further experiments
were conducted using the most granular order book quote data available,
utilizing pandas for analysis. Unfortunately, these experiments also yielded
unfavorable results, indicating insufficient market volatility to capture alpha.
This experiment is implemented in `spread_arb.pandas_backtest.backtest`. It can
be run using the `granular_backtest.ipynb` notebook.

## Conclusion

The series of experiments detailed in this module represents over a year of
meticulous testing and analysis aimed at developing and refining a crypto
arbitrage strategy. While certain approaches showed promise initially, the
overall conclusion suggests that the strategy may not be viable due to the lack
of sufficient market volatility to exploit spread differentials effectively.
These findings underscore the importance of rigorous testing and adaptability in
algorithmic trading strategies.

For further details on the experiments and code implementation, please refer to
the respective modules and files within this repository.
