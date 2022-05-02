import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

from numpy import isin
from pandas import DataFrame

# Since GOOGLE_FUNCTION_SOURCE is modified, the directory of this file will be in sys.path rather than the root workspace directory
# Insert root directory of build location (/workspace) to sys.path
sys.path.append(os.getcwd())
import cryptomart
from pyutil.profiling import timed

N_WORKERS = 10
ORDER_BOOK_DEPTH = 20
ORDER_BOOK_SHAPE = (ORDER_BOOK_DEPTH * 2, len(cryptomart.enums.OrderBookSchema))
SKIP_SYMBOLS = []
SKIP_EXCHANGES = set()

# GCP variables
GCP_PROJECT = os.getenv("GCP_PROJECT", "crypto-arb-341504")
BQ_DATASET_NAME = os.getenv("BQ_DATASET_NAME", "{exchange}_order_book")
DISABLE_PUSH_TO_BQ = os.getenv("DISABLE_PUSH_TO_BQ")


def order_book_ping_test(request):
    dm = cryptomart.Client(debug=True, exchange_init_kwargs={"cache_path": "cache"})
    logger = logging.getLogger("cryptomart")
    errors = []

    def get_order_book(id, exchange, symbol, instType):
        try:
            logger.info(f"Getting order book: #{id:<4} {exchange:<15} {symbol:<15} {instType}")

            order_book = dm._exchange_instance_map[exchange].order_book(
                symbol, instType, depth=ORDER_BOOK_DEPTH, log_level="INFO"
            )
            if isinstance(order_book, DataFrame) and len(order_book.columns) == ORDER_BOOK_SHAPE[1]:
                logger.info(
                    f"Order book received: #{id:<4} {exchange:<15} {symbol:<15} {instType} ... Pushing to database"
                )

                # Push to database
                exchange_name = exchange.lower()
                table_name = f"{BQ_DATASET_NAME.format(exchange=exchange_name)}.{symbol}_{instType}"
                if not DISABLE_PUSH_TO_BQ:
                    order_book.to_gbq(table_name, project_id=GCP_PROJECT, if_exists="append")
                    logger.info(f"Success: #{id:<4} {exchange:<15} {symbol:<15} {instType}")
            else:
                raise Exception("Invalid orderbook received")

        except Exception as e:
            tb = traceback.format_exc()
            errors.append((exchange, symbol, instType, tb))

    def get_all_instruments(exchanges=set(cryptomart.Exchange._values()) - SKIP_EXCHANGES):
        exchange_symbols = dict()
        symbols = []
        for exchange in exchanges:
            exchange_inst = dm._exchange_instance_map[exchange]
            perpetual_instruments = exchange_inst.active_instruments[
                exchange_inst.active_instruments.instType == cryptomart.InstrumentType.PERPETUAL
            ]
            filtered_symbols = perpetual_instruments[~isin(perpetual_instruments.symbol, SKIP_SYMBOLS)]
            exchange_symbols[exchange] = list(
                zip(filtered_symbols.symbol.to_list(), filtered_symbols.instType.to_list())
            )

        exchange_pointer = 0
        while len(exchange_symbols.keys()) > 0:
            exchange = list(exchange_symbols.keys())[exchange_pointer % len(exchange_symbols.keys())]
            try:
                symbol = exchange_symbols[exchange].pop()
                symbols.append((exchange_pointer, exchange, *symbol))
            except IndexError:
                exchange_symbols.pop(exchange)
            exchange_pointer += 1

        logger.info(f"Total symbols to query: {len(symbols)}")
        return symbols

    @timed()
    def run_main_script():
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            executor.map(lambda args: get_order_book(*args), get_all_instruments())

    run_main_script()

    logger.info("Order book completed")
    logger.info("Exceptions thrown by worker threads: ")
    for i, error in enumerate(errors):
        msg = "-" * 20 + f" {i}/{len(errors)}" + "\n" + " ".join(error)
        msg = msg.replace("\n", " ")
        logger.warning(msg)

    if len(errors) == 0:
        return "Success"
    else:
        return "Some threads raised exceptions. See cloud logs for more details."
