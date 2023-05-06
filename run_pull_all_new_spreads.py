import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cryptomart as cm
import pandas as pd
from cryptomart.errors import NotSupportedError
from cryptomart.exchanges.base import ExchangeAPIBase
from pyutil.dicts import format_dict
from pyutil.misc import make_timestamped_dir, unique_file_name

from app.globals import BLACKLISTED_SYMBOLS

print(__name__)
main_logger = logging.getLogger(f"cryptomart")
LOG_DIR = make_timestamped_dir(f"logs/pull_spreads")
main_log_file = unique_file_name(os.path.join(LOG_DIR, "main.log"))
main_logger.addHandler(logging.FileHandler(main_log_file))

data_control = {
    cm.Exchange.BINANCE: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
    cm.Exchange.BITMEX: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": True},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
    cm.Exchange.BYBIT: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
    cm.Exchange.GATEIO: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
    cm.Exchange.KUCOIN: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
    cm.Exchange.OKEX: {
        "enabled": True,
        cm.enums.Interface.OHLCV: {
            "enabled": True,
            cm.InstrumentType.PERPETUAL: {
                "enabled": True,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.SPOT: {
                "enabled": False,
                "cache_kwargs": {"disabled": False, "refresh": False},
            },
            cm.InstrumentType.QUARTERLY: {"enabled": False},
            cm.InstrumentType.MONTHLY: {"enabled": False},
        },
        cm.enums.Interface.FUNDING_RATE: {
            "enabled": False,
            "cache_kwargs": {"disabled": False, "refresh": False},
        },
    },
}


def pull_one_exchange(exchange_name, exchange_inst: ExchangeAPIBase):
    exchange_logger = logging.getLogger(f"cryptomart.{exchange_name}")
    log_file = unique_file_name(os.path.join(LOG_DIR, f"{exchange_name}.log"))
    exchange_logger.addHandler(logging.FileHandler(log_file))

    if not data_control[exchange_name]["enabled"]:
        exchange_logger.info(f"Skipping exchange: {exchange_name}")
        return

    all_ohlcv = pd.DataFrame()
    all_funding = pd.DataFrame()
    try:
        if data_control[exchange_name]["ohlcv"]["enabled"]:
            exchange_logger.info(f"Pulling data for {exchange_name}")
            exchange_logger.info(f"Pulling ohlcv for {exchange_name}")
            for inst_type in cm.InstrumentType:
                if not data_control[exchange_name]["ohlcv"][inst_type]["enabled"]:
                    exchange_logger.info(f"Skipping {inst_type} OHLCV for {exchange_name}")
                    continue

                try:
                    instruments = exchange_inst.instrument_info(inst_type, map_column="exchange_symbol")
                except NotSupportedError:
                    continue

                for cryptomart_symbol, instrument_id in instruments.items():
                    if cryptomart_symbol in BLACKLISTED_SYMBOLS:
                        continue
                    try:
                        ohlcv = exchange_inst.ohlcv(
                            cryptomart_symbol,
                            inst_type,
                            starttime=(2022, 10, 10),
                            endtime=(2023, 5, 4),
                            cache_kwargs=data_control[exchange_name]["ohlcv"][inst_type]["cache_kwargs"],
                        )
                        stats = pd.Series(
                            {
                                "exchange": exchange_name,
                                "inst_type": inst_type,
                                "symbol": cryptomart_symbol,
                                "earliest_time": ohlcv.earliest_time,
                                "latest_time": ohlcv.latest_time,
                                "missing_rows": len(ohlcv.missing_rows),
                                "valid_rows": len(ohlcv.valid_rows),
                                "gaps": ohlcv.gaps,
                            }
                        )
                        all_ohlcv = pd.concat([all_ohlcv, stats.to_frame().T], ignore_index=True)
                    except NotSupportedError:
                        exchange_logger.info(f"OHLCV Not supported: {exchange_name} | {inst_type}.{cryptomart_symbol}")
                        pass
        else:
            exchange_logger.info(f"Skipping ohlcv for {exchange_name}")

        if data_control[exchange_name]["funding_rate"]["enabled"]:
            exchange_logger.info(f"Pulling funding rate for {exchange_name}")

            perp_instruments = exchange_inst.instrument_info(cm.InstrumentType.PERPETUAL, map_column="exchange_symbol")

            for cryptomart_symbol, instrument_id in perp_instruments.items():
                try:
                    funding_rate = exchange_inst.funding_rate(
                        cryptomart_symbol,
                        starttime=(2019, 1, 1),
                        endtime=(2022, 6, 19),
                        cache_kwargs=data_control[exchange_name]["funding_rate"]["cache_kwargs"],
                    )

                    stats = pd.Series(
                        {
                            "exchange": exchange_name,
                            "symbol": cryptomart_symbol,
                            "earliest_time": funding_rate.earliest_time,
                            "latest_time": funding_rate.latest_time,
                            "missing_rows": len(funding_rate.missing_rows),
                            "valid_rows": len(funding_rate.valid_rows),
                            "gaps": funding_rate.gaps,
                        }
                    )
                    all_funding = pd.concat([all_funding, stats.to_frame().T], ignore_index=True)
                except NotSupportedError:
                    exchange_logger.info(
                        f"Funding Rate Not supported: {exchange_name} | {inst_type}.{cryptomart_symbol}"
                    )
                    pass
        else:
            exchange_logger.info(f"Skipping funding rate for {exchange_name}")

        exchange_logger.info(f"Done pulling data for {exchange_name}")
        return {
            "ohlcv": all_ohlcv,
            "funding": all_funding,
        }

    except Exception as e:
        exchange_logger.error(f"Exception occured while running for {exchange_name}: \n{e}\n", exc_info=True)


if __name__ == "__main__":

    log_file = "logs/pull_all_data_2.log"
    mode = "parallel"

    client = cm.Client(
        log_level="DEBUG",
        quiet=True,
        # Set refresh to true to pull up to date symbols. Else cached instrument_info will be used
        cache_kwargs={"refresh": True},
    )

    main_logger.info(f"Cache control: \n{format_dict(data_control)}\n")

    if mode == "serial":
        for exchange_name, exchange_inst in client._exchange_instance_map.items():
            res = pull_one_exchange(exchange_name, exchange_inst)
            if res is not None:
                if res["funding"] is not None:
                    res["funding"].to_csv(f"data/funding_{exchange_name}_stats.csv", index=False)
                if res["ohlcv"] is not None:
                    res["ohlcv"].to_csv(f"data/ohlcv_{exchange_name}_stats.csv", index=False)

                main_logger.info(f"Result: \n{res}\n")

    elif mode == "parallel":
        with ThreadPoolExecutor(max_workers=len(client._exchange_instance_map)) as executor:
            res = executor.map(
                pull_one_exchange,
                client._exchange_instance_map.keys(),
                client._exchange_instance_map.values(),
            )

        main_logger.info("Done pulling data")
        main_logger.info(f"Result: \n{res}\n")
