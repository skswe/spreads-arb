import asyncio
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import cached_property

import app
import cryptomart as cm
import pandas as pd
import requests
from tardis_dev import datasets, get_exchange_details

cm_client = cm.Client()

filehandler = logging.FileHandler("tardis_download.log", mode="w")
logger = logging.getLogger("cryptomart")
logger.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
logging.getLogger("pyutil.cache").setLevel(logging.WARNING)
logging.getLogger("pyutil.cache").addHandler(filehandler)
logging.getLogger("tardis_dev.datasets.download").setLevel(logging.DEBUG)
logging.getLogger("tardis_dev.datasets.download").addHandler(filehandler)


class TardisData:
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
        self.api_key = os.getenv("TARDIS_API_KEY")
        self.base_url = "https://api.tardis.dev/v1"
        self.data_root_path = os.path.join(os.getenv("ACTIVE_DEV_PATH"), "spreads-arb", "data", "tardis")
        self.tardis_exchanges = self.exchange_map.set_index("cryptomart_exchange").iloc[:, 0].to_dict()
        self.cryptomart_exchanges = self.exchange_map.set_index("id").iloc[:, 0].to_dict()
        self.exchange_info = self.load_exchange_info()

    @staticmethod
    def get_data_filename(exchange, data_type, date, symbol, format):
        return f"{exchange}/{data_type}/{symbol}/{date.strftime('%Y-%m-%d')}.{format}.gz"

    @cached_property
    def exchanges(self):
        exchanges = pd.DataFrame(requests.get(f"{self.base_url}/exchanges").json())
        exchanges = exchanges[exchanges.delisted.isna()]
        exchanges = exchanges[exchanges.enabled]
        exchanges = exchanges.merge(self.exchange_map)
        return exchanges

    def load_exchange_info(self):
        return {self.cryptomart_exchanges[exchange]: get_exchange_details(exchange) for exchange in self.exchanges.id}

    def exchange_datasets(self, exchange):
        symbols = cm_client.instrument_info(exchange, "perpetual")[["cryptomart_symbol", "exchange_symbol"]].rename(
            columns={"exchange_symbol": "id"}
        )
        return pd.DataFrame(self.exchange_info[exchange]["datasets"]["symbols"]).merge(symbols, how="inner")

    def download(self, exchange, data_types=["trades"], from_date="2023-01-12", to_date="2023-05-04", symbols=None):
        tardis_symbols = self.exchange_datasets(exchange).pipe(lambda df: df[df.availableSince <= from_date]).id
        if symbols is None:
            symbols = tardis_symbols
        else:
            instrument_info = cm_client.instrument_info(exchange, "perpetual", map_column="exchange_symbol")
            symbols = [instrument_info[symbol] for symbol in symbols]
            symbols = [symbol for symbol in symbols if symbol in tardis_symbols]

        tardis_exchange = self.tardis_exchanges[exchange]

        datasets.download(
            exchange=tardis_exchange,
            data_types=data_types,
            from_date=from_date,
            to_date=to_date,
            symbols=symbols,
            api_key=self.api_key,
            download_dir=self.data_root_path,
            get_filename=self.get_data_filename,
        )

    def data_iterator(self, exchange, symbol, data_type, from_date="2023-01-12", to_date="2023-05-04"):
        start_time = datetime.fromisoformat(from_date)
        end_time = datetime.fromisoformat(to_date)
        day_timedelta = timedelta(days=1)
        tardis_symbol = self.exchange_datasets(exchange).set_index("cryptomart_symbol").loc[symbol, "id"]
        tardis_exchange = self.tardis_exchanges[exchange]

        for day in range((end_time - start_time).days):
            date = start_time + day * day_timedelta
            yield os.path.join(
                self.data_root_path, self.get_data_filename(tardis_exchange, data_type, date, tardis_symbol, "csv")
            )

    def load_trades(self, exchange, symbol, from_date="2023-01-12", to_date="2023-05-04"):
        dfs = []
        for filename in self.data_iterator(exchange, symbol, "trades", from_date, to_date):
            dfs.append(pd.read_csv(filename))
        return pd.concat(dfs, ignore_index=True)

    def load_bas(self, exchange, symbol, from_date="2023-01-12", to_date="2023-05-04"):
        dfs = []
        for filename in self.data_iterator(exchange, symbol, "quotes", from_date, to_date):

            def largest_spread(g):
                g["bas"] = round(g["ask_price"] - g["bid_price"], 15)
                return g.groupby("bas", as_index=True)[["bid_amount", "ask_amount"]].sum().iloc[:3]

            df = pd.read_csv(filename, usecols=["timestamp", "ask_price", "ask_amount", "bid_price", "bid_amount"])
            df["timestamp"] = df.timestamp.apply(lambda x: datetime.utcfromtimestamp(x / 1e6))
            df = df.resample(timedelta(hours=1), on="timestamp").apply(largest_spread).reset_index()
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


def download_and_process_bas(exchange, symbols, from_date="2023-02-20", to_date="2023-05-04"):
    tardis_exchange = td.tardis_exchanges[exchange]
    tardis_symbols = td.exchange_datasets(exchange).set_index("cryptomart_symbol").loc[symbols, "id"]

    td.download(exchange, ["quotes"], from_date, to_date, symbols)

    for symbol, tardis_symbol in zip(symbols, tardis_symbols):
        tardis_download_dir = os.path.join(td.data_root_path, tardis_exchange, "quotes", tardis_symbol)
        logger.info(f"Processing {exchange} {symbol}")
        bas = td.load_bas(exchange, symbol, from_date, to_date)

        outpath = os.path.join(
            os.getenv("ACTIVE_DEV_PATH"), "spreads-arb", "data", "bid_ask_spreads_1h", exchange, f"{symbol}.pkl"
        )
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        bas.to_pickle(outpath)

        os.remove(tardis_download_dir)
        logger.info(f"Successfully loaded {exchange} {symbol}")


# if __name__ == "__main__":
#     event_loop = asyncio.get_event_loop()
#     td = TardisData()

#     ohlcvs = app.data_prep.all_ohlcv(
#         "2022-02-01", "2023-05-04", "interval_1h", refresh=False, identifiers=["spreads-arb-v2"]
#     )
#     ohlcvs = ohlcvs[ohlcvs.missing_rows <= 0]
#     ohlcvs = ohlcvs.reset_index()[["exchange", "symbol"]]

#     for exchange in ohlcvs.exchange.unique():
#         symbols = ohlcvs[ohlcvs.exchange == exchange].symbol.unique()
#         download_and_process_bas(exchange, symbols)


td = TardisData()

try:
    td.download("binance", ["quote"], "2022-02-20", "2023-05-04", ["BTC", "ETH"])
except Exception as e:
    print(e)
