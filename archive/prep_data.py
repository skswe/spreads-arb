import argparse
import os
import pickle

import cryptomart as cm
import pandas as pd
from app.data_prep.dummy import dummy_bid_ask_spreads
from app.data_prep.real import all_bid_ask_spreads, all_funding_rates, all_ohlcv, get_fee_info
from app.feeds import Spread


if __name__ == "__main__":
    cm_client = cm.Client(quiet=True)

    parser = argparse.ArgumentParser(description="Prep backtest data")

    parser.add_argument("--identifier", type=str, default="spread_arb_v2")
    parser.add_argument("--z-score-period", type=int, default=30)
    parser.add_argument("--data-start", type=str, default="2022-10-10")
    parser.add_argument("--data-end", type=str, default="2023-05-04")

    parser.add_argument("--refresh-ohlcv-data", action="store_true", default=False)
    parser.add_argument("--refresh-bas-data", action="store_true", default=False)
    parser.add_argument("--refresh-fr-data", action="store_true", default=False)
    parser.add_argument("--refresh-fee-data", action="store_true", default=False)

    parser.add_argument("--dummy-slippage", type=float, default=None)
    parser.add_argument("--force-default-slippage", action="store_true", default=False)
    parser.add_argument("--use-fixed-fr", type=float, default=None)

    parser.add_argument("--load-spreads", type=str, default=None)
    parser.add_argument("--save-spreads", type=str, default=None)

    args = parser.parse_args()

    # Load data

    if args.load_spreads is not None:
        spreads = pd.read_pickle(args.load_spreads)

    else:
        ohlcvs = all_ohlcv(
            args.data_start, args.data_end, refresh=args.refresh_ohlcv_data, identifiers=[args.identifier]
        )

        ohlcvs = ohlcvs[ohlcvs["missing_rows"] == 0]

        if args.use_fixed_fr is not None:
            pass
        else:
            funding_rates = all_funding_rates(
                args.data_start, args.data_end, refresh=args.refresh_fr_data, identifiers=[args.identifier]
            )

        if args.dummy_slippage is not None:
            ba_spreads = dummy_bid_ask_spreads(ohlcvs, args.dummy_slippage, args.force_default_slippage)
        else:
            ba_spreads = all_bid_ask_spreads(
                args.data_start, args.data_end, refresh=args.refresh_bas_data, identifiers=[args.identifier]
            )

        fee_info = get_fee_info(refresh=args.refresh_fee_data, identifiers=[args.identifier])

        spreads = create_spreads(ohlcvs, fee_info, funding_rates=funding_rates, bas=ba_spreads)

        if args.save_spreads is not None:
            spreads.to_pickle(args.save_spreads)
