import argparse
import os
import pickle

import cryptomart as cm
import pandas as pd
from app.data_prep.dummy import dummy_bid_ask_spreads
from app.data_prep.real import all_bid_ask_spreads, all_funding_rates, all_ohlcv, get_fee_info
from app.feeds import Spread


def create_spreads(ohlcvs, fee_info, funding_rates=None, bas=None, bas_type="sum") -> pd.DataFrame:
    assert bas_type in ["sum", "mean"]
    
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
            if bas_type == "mean":
                bid_ask_a = pickle.loads(bas.at[indexer_a, "bid_ask_spread"])
                bid_ask_b = pickle.loads(bas.at[indexer_b, "bid_ask_spread"])
                spread.add_bid_ask_spread(bid_ask_a, bid_ask_b)
                spreads.at[indexer, (f"avg_ba_spread_{s}" for s in ("a", "b"))] = set(
                    x.bid_ask_spread.mean() for x in (bid_ask_a, bid_ask_b)
                )
            elif bas_type == "sum":
                bid_ask_a = pickle.loads(bas.at[indexer_a, "slippage"])
                bid_ask_b = pickle.loads(bas.at[indexer_b, "slippage"])
                spread.add_bid_ask_spread(bid_ask_a, bid_ask_b, type="sum")

        spreads.at[indexer, "spread"] = pickle.dumps(spread)
        spreads.at[
            indexer, "alias"
        ] = f"{row.exchange_a[:4]}_{row.exchange_b[:4]}_{row.inst_type_a[:4]}_{row.inst_type_b[:4]}_{row.symbol}"
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
