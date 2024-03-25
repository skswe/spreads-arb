import asyncio
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta

import cryptomart as cm
import nest_asyncio
import numpy as np
import pandas as pd

logger = logging.getLogger("cryptomart")
logger.setLevel(logging.DEBUG)
logging.getLogger("tardis_dev.datasets.download").setLevel(logging.DEBUG)

# filehandler = logging.FileHandler("tardis_download.log", mode="w")
# logger.addHandler(filehandler)
# logging.getLogger("tardis_dev.datasets.download").addHandler(filehandler)

try:
    from tardis_dev import datasets, get_exchange_details

    nest_asyncio.apply()

    class TardisData:
        """Class for managing Tardis data

        Attributes:
            api_key: API key for Tardis
            base_url: Base URL for Tardis API
            data_root_path: Root path for data storage
            tardis_data_root_path: Root path for Tardis data storage
            loop: Event loop for async operations
            tardis_exchanges: Mapping of Tardis exchanges to Cryptomart exchanges
            cryptomart_exchanges: Mapping of Cryptomart exchanges to Tardis exchanges
            exchange_info: Exchange information for Tardis exchanges
            instrument_info: Instrument information for Tardis exchanges
            all_symbols: DataFrame of all symbols for Tardis exchanges
            all_symbols_with_spread: DataFrame of symbols with spread for Tardis exchanges
        """

        exchange_map = pd.DataFrame(
            [
                {
                    "binance": "binance-futures",
                    "bitmex": "bitmex",
                    "bybit": "bybit",
                    "gateio": "gate-io-futures",
                    "okex": "okex-swap",
                }
            ]
        ).melt(var_name="cryptomart_exchange", value_name="id")

        def __init__(self):
            """Initialize TardisData object"""

            self.cm_client = cm.Client(quiet=True)

            self.api_key = os.getenv("TARDIS_API_KEY")
            self.base_url = "https://api.tardis.dev/v1"
            self.data_root_path = os.path.join(os.getenv("ACTIVE_DEV_PATH"), "spreads-arb", "data")
            self.tardis_data_root_path = os.path.join(self.data_root_path, "tardis")
            self.loop = asyncio.get_event_loop()
            self.tardis_exchanges = self.exchange_map.set_index("cryptomart_exchange").iloc[:, 0].to_dict()
            self.cryptomart_exchanges = self.exchange_map.set_index("id").iloc[:, 0].to_dict()

            self.exchange_info = self.load_exchange_info()
            self.instrument_info = self.load_instrument_info()

            self.all_symbols = pd.concat(
                [
                    self.get_symbols(exchange, with_spread=False).assign(exchange=exchange)
                    for exchange in self.tardis_exchanges
                ]
            )
            self.all_symbols_with_spread = (
                self.all_symbols.join(self.all_symbols, lsuffix="_x", rsuffix="_y", how="cross")
                .pipe(
                    lambda df: df[(df.exchange_x < df.exchange_y) & (df.cryptomart_symbol_x == df.cryptomart_symbol_y)]
                )[["id_x", "cryptomart_symbol_x"]]
                .rename(columns={"id_x": "id", "cryptomart_symbol_x": "cryptomart_symbol"})[["cryptomart_symbol"]]
                .drop_duplicates()
            )
            logging.getLogger("tardis_dev.datasets.download").addHandler(logging.StreamHandler(sys.stdout))

        def load_exchange_info(self):
            exchange_info_futures = [get_exchange_details(exchange) for exchange in self.tardis_exchanges.values()]
            exchange_info = self.loop.run_until_complete(asyncio.gather(*exchange_info_futures))
            return {self.cryptomart_exchanges[info["id"]]: info for info in exchange_info}

        def load_instrument_info(self):
            return {
                exchange: self.cm_client.instrument_info(exchange, "perpetual", cache_kwargs={"refresh": False})
                for exchange in self.cryptomart_exchanges.values()
            }

        @staticmethod
        def get_data_filename(exchange, data_type, date, symbol, format):
            return f"{exchange}/{data_type}/{symbol}/{date.strftime('%Y-%m-%d')}.{format}.gz"

        def get_symbol_id(self, exchange, cryptomart_symbol, ignore_errors=False):
            if not isinstance(cryptomart_symbol, list):
                cryptomart_symbol = [cryptomart_symbol]
                ret_fn = lambda x: x[0]
            else:
                ret_fn = lambda x: list(x)

            cryptomart_symbols = pd.Series(cryptomart_symbol, name="cryptomart_symbol")

            symbols = self.all_symbols[self.all_symbols.exchange == exchange].merge(cryptomart_symbols, how="right")
            if not ignore_errors and symbols.id.isna().any():
                raise ValueError(
                    f"{list(symbols[symbols.id.isna()].cryptomart_symbol)} not found on exchange {exchange}"
                )

            return ret_fn(symbols.id)

        def get_symbols(
            self,
            exchange,
            data_types=["quotes", "derivative_ticker"],
            from_date="2023-04-10",
            to_date="2023-05-10",
            with_spread=True,
        ):
            all_symbols = pd.DataFrame(self.exchange_info[exchange]["datasets"]["symbols"]).pipe(
                lambda df: df[
                    (df.type == "perpetual")
                    & (df.availableSince <= from_date)
                    & (df.availableTo >= to_date)
                    & (df.dataTypes.apply(lambda l: np.isin(data_types, l).all()))
                ]
            )[["id"]]
            cryptomart_symbols = self.instrument_info[exchange][["cryptomart_symbol", "exchange_symbol"]].rename(
                columns={"exchange_symbol": "id"}
            )
            if not with_spread:
                return all_symbols.merge(cryptomart_symbols)
            else:
                return all_symbols.merge(cryptomart_symbols).merge(self.all_symbols_with_spread)

        def download(
            self,
            exchange,
            data_types=["trades"],
            from_date="2023-01-12",
            to_date="2023-05-04",
            symbols=None,
            compress=False,
            **kwargs,
        ):
            if symbols is None:
                symbols = self.get_symbols(exchange, data_types, from_date, to_date, with_spread=True)
            else:
                all_symbols = self.get_symbols(exchange, data_types, from_date, to_date, with_spread=False)
                symbols = all_symbols.merge(pd.Series(symbols, name="cryptomart_symbol"), how="right")
                unavailable_symbols = symbols[symbols.id.isna()].cryptomart_symbol.unique()
                if len(unavailable_symbols) > 0:
                    print("Warning: some symbols are not available for download", unavailable_symbols)
                symbols = symbols.dropna()

            tardis_exchange = self.tardis_exchanges[exchange]

            future = datasets.download(
                exchange=tardis_exchange,
                data_types=data_types,
                from_date=from_date,
                to_date=to_date,
                symbols=list(symbols.id),
                api_key=self.api_key,
                download_dir=self.tardis_data_root_path,
                get_filename=self.get_data_filename,
                **kwargs,
            )

            self.loop.run_until_complete(future)

            if compress:
                for data_type in data_types:
                    if data_type not in ["derivative_ticker", "quotes"]:
                        print("No compression implemented for data type", data_type)
                        continue
                    for symbol in symbols.cryptomart_symbol:
                        if data_type == "derivative_ticker":
                            self.load_ticker(exchange, symbol, from_date, to_date, compress=True)
                        elif data_type == "quotes":
                            self.load_quotes(exchange, symbol, from_date, to_date, compress=True)

        def data_iterator(self, exchange, symbol, data_type, from_date="2023-01-12", to_date="2023-05-04"):
            start_time = datetime.fromisoformat(from_date)
            end_time = datetime.fromisoformat(to_date)
            day_timedelta = timedelta(days=1)
            tardis_symbol = self.get_symbol_id(exchange, symbol)
            tardis_exchange = self.tardis_exchanges[exchange]

            for day in range((end_time - start_time).days):
                date = start_time + day * day_timedelta
                yield os.path.join(
                    self.tardis_data_root_path,
                    self.get_data_filename(tardis_exchange, data_type, date, tardis_symbol, "csv"),
                )

        def load_trades(self, exchange, symbol, from_date="2023-01-12", to_date="2023-05-04"):
            dfs = []
            for filename in self.data_iterator(exchange, symbol, "trades", from_date, to_date):
                dfs.append(pd.read_csv(filename))
            return pd.concat(dfs, ignore_index=True)

        def load_ticker(self, exchange, symbol, from_date="2023-01-12", to_date="2023-05-04", compress=False):
            dfs = []
            for filename in self.data_iterator(exchange, symbol, "derivative_ticker", from_date, to_date):
                dfs.append(pd.read_csv(filename, usecols=["timestamp", "last_price"]))
            df = pd.concat(dfs, ignore_index=True)
            df["timestamp"] = df.timestamp.apply(lambda x: datetime.utcfromtimestamp(x / 1e6))
            df = df.dropna()

            if compress:
                outpath = os.path.join(self.data_root_path, "tick_prices", exchange, f"{symbol}.parquet")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                df.to_parquet(outpath)
                shutil.rmtree(os.path.dirname(filename))

            return df

        def load_quotes(self, exchange, symbol, from_date="2023-04-10", to_date="2023-05-10", compress=False):
            dfs = []
            for filename in self.data_iterator(exchange, symbol, "quotes", from_date, to_date):
                try:
                    dfs.append(
                        pd.read_csv(
                            filename, usecols=["timestamp", "ask_price", "ask_amount", "bid_price", "bid_amount"]
                        )
                    )
                except FileNotFoundError:
                    print("skipping", exchange, symbol)
                    return
            df = pd.concat(dfs, ignore_index=True)
            df["timestamp"] = df.timestamp.apply(lambda x: datetime.utcfromtimestamp(x / 1e6))
            df = df.dropna()
            if compress:
                print("Saving quotes for", exchange, symbol, from_date, to_date)
                outpath = os.path.join(self.data_root_path, "tick_quotes", exchange, f"{symbol}.parquet")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                df.to_parquet(outpath)
                shutil.rmtree(os.path.dirname(filename))

            return df

except ImportError:
    logger.warning("Tardis not installed. Some functionality will be unavailable.")

    TardisData = None
