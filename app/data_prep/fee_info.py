import os

import cryptomart as cm
import numpy as np
import pandas as pd
import pyutil
import requests


@pyutil.cache.cached("/tmp/cache/fee_margin_data", refresh=False)
def get_fee_info(**cache_kwargs) -> pd.DataFrame:
    cm_client = cm.Client(quiet=True)

    # BINANCE
    fee_info_binance = cm_client.binance.instrument_info("perpetual")
    fee_info_binance = fee_info_binance.set_index("cryptomart_symbol").rename_axis(index="symbol")
    fee_info_binance = (
        fee_info_binance[["maintMarginPercent", "requiredMarginPercent"]]
        .rename(columns={"maintMarginPercent": "maint_margin", "requiredMarginPercent": "init_margin"})
        .astype(float)
        .apply(lambda x: x / 100)
    )
    fee_info_binance = fee_info_binance.assign(fee_pct=0.0004, fee_fixed=0)

    # BITMEX
    fee_info_bitmex = cm_client.bitmex.instrument_info("perpetual")
    fee_info_bitmex = fee_info_bitmex.set_index("cryptomart_symbol").rename_axis(index="symbol")
    fee_info_bitmex = fee_info_bitmex[["maintMargin", "initMargin", "takerFee"]].rename(
        columns={"maintMargin": "maint_margin", "initMargin": "init_margin", "takerFee": "fee_pct"}
    )
    fee_info_bitmex = fee_info_bitmex.assign(fee_fixed=0)
    fee_info_bitmex

    # BYBIT
    @pyutil.cache.cached("/tmp/cache/fee_margin_data", refresh=False, path_seperators=["exchange"])
    def bybit_get_single_margin(exchange_symbol, exchange="bybit"):
        url = os.path.join(cm_client.bybit.base_url, "public", "linear", "risk-limit")
        params = {"symbol": exchange_symbol}
        request = requests.Request("GET", url=url, params=params)
        response = cm_client.bybit.dispatcher.send_request(request)
        response = cm.interfaces.api.APIInterface.extract_response_data(
            response, ["result"], ["ret_code"], 0, ["ret_msg"], raw=True
        )
        risk_tiers = pd.DataFrame(response)
        lowest_risk_tier = risk_tiers[risk_tiers.limit == risk_tiers.limit.min()]
        return (
            lowest_risk_tier[["maintain_margin", "starting_margin"]]
            .rename(columns={"maintain_margin": "maint_margin", "starting_margin": "init_margin"})
            .iloc[0]
        )

    fee_info_bybit = cm_client.bybit.instrument_info("perpetual")
    fee_info_bybit = fee_info_bybit.set_index("cryptomart_symbol").rename_axis(index="symbol")

    fee_info_bybit = pd.concat([fee_info_bybit, fee_info_bybit.exchange_symbol.apply(bybit_get_single_margin)], axis=1)
    fee_info_bybit = fee_info_bybit[["maint_margin", "init_margin", "taker_fee"]].rename(
        columns={"taker_fee": "fee_pct"}
    )
    fee_info_bybit = fee_info_bybit.assign(fee_fixed=0)
    fee_info_bybit

    # GATEIO
    # gateio does not provide init margin rate. Set to NaN and fill with the average of the other exchanges
    fee_info_gateio = cm_client.gateio.instrument_info("perpetual")
    fee_info_gateio = fee_info_gateio.set_index("cryptomart_symbol").rename_axis(index="symbol")
    fee_info_gateio = fee_info_gateio[["taker_fee_rate", "maintenance_rate"]].rename(
        columns={"taker_fee_rate": "fee_pct", "maintenance_rate": "maint_margin"}
    )
    fee_info_gateio = fee_info_gateio.assign(init_margin=np.nan)
    fee_info_gateio = fee_info_gateio.assign(fee_fixed=0)
    fee_info_gateio = fee_info_gateio[["maint_margin", "init_margin", "fee_pct", "fee_fixed"]]
    fee_info_gateio

    # KUCOIN
    fee_info_kucoin = cm_client.kucoin.instrument_info("perpetual")
    fee_info_kucoin = fee_info_kucoin.set_index("cryptomart_symbol").rename_axis(index="symbol")
    fee_info_kucoin = fee_info_kucoin[["maintainMargin", "initialMargin", "takerFeeRate", "makerFixFee"]].rename(
        columns={
            "maintainMargin": "maint_margin",
            "initialMargin": "init_margin",
            "takerFeeRate": "fee_pct",
            "makerFixFee": "fee_fixed",
        }
    )
    fee_info_kucoin

    # OKEX
    # Exchange-wide taker fee: https://www.okx.com/fees
    @pyutil.cache.cached("/tmp/cache/fee_margin_data", refresh=False, path_seperators=["exchange"])
    def okex_get_single_margin(exchange_symbol, exchange="okex"):
        url = os.path.join(cm_client.okex.base_url, "api", "v5", "public", "position-tiers")
        params = {"instType": "SWAP", "tdMode": "isolated", "uly": exchange_symbol}
        request = requests.Request("GET", url=url, params=params)
        response = cm_client.okex.dispatcher.send_request(request)
        response = cm.interfaces.api.APIInterface.extract_response_data(
            response, ["data"], ["code"], "0", ["msg"], raw=True
        )
        risk_tiers = pd.DataFrame(response)
        return (
            risk_tiers.loc[risk_tiers.tier == "2", ["mmr", "imr"]]
            .rename(columns={"mmr": "maint_margin", "imr": "init_margin"})
            .iloc[0]
        )

    fee_info_okex = cm_client.okex.instrument_info("perpetual")
    fee_info_okex = fee_info_okex.set_index("cryptomart_symbol").rename_axis(index="symbol")
    fee_info_okex = pd.concat([fee_info_okex, fee_info_okex.uly.apply(okex_get_single_margin)], axis=1)
    fee_info_okex = fee_info_okex[["maint_margin", "init_margin"]]
    fee_info_okex = fee_info_okex.assign(fee_pct=0.0005, fee_fixed=0)
    fee_info_okex

    # Join with instrument_data
    DEFAULT_INIT_MARGIN = 0
    DEFAULT_MAINT_MARGIN = 0.15
    all_fee_info = pd.concat(
        [
            fee_info_binance,
            fee_info_bitmex,
            fee_info_bybit,
            fee_info_gateio,
            fee_info_kucoin,
            fee_info_okex,
        ],
        keys=["binance", "bitmex", "bybit", "gateio", "kucoin", "okex"],
        names=["exchange"],
    )

    all_fee_info = (
        all_fee_info.astype(float)
        .groupby("symbol")
        .apply(lambda c: c.fillna(c.mean()))
        .fillna({"init_margin": DEFAULT_INIT_MARGIN, "maint_margin": DEFAULT_MAINT_MARGIN})
    )

    return all_fee_info
