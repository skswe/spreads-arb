#!/bin/bash

options=(
    "--identifier spreads-arb-v2"
    "--z-score-period 500"
    "--data-start 2022-02-01"
    "--data-end 2023-05-04"

    # "--refresh-ohlcv-data"
    # "--refresh-bas-data"
    # "--refresh-fr-data"
    # "--refresh-fee-data"

    "--dummy-slippage 0.02"
    # "--force-default-slippgae"
    # "--use-fixed-fr 0.01"
    
    "--load-spreads cached_spreads.pkl"
    # "--save-spreads cached_spreads.pkl"
)

command=$(printf "%s " "${options[@]}")

# echo $command

python3 run_backtest_chained.py $command
