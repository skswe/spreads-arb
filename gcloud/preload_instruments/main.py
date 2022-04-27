from concurrent.futures import ThreadPoolExecutor
import traceback
import cryptomart

N_WORKERS = 10

client = cryptomart.Client()
errors = []

def get_all_instruments(exchanges=set(cryptomart.Exchange._values())):
    exchange_symbols = dict()
    symbols = []
    for exchange in exchanges:
        exchange_inst = client._exchange_instance_map[exchange]
        perpetual_instruments = exchange_inst.active_instruments[
            exchange_inst.active_instruments.instType == cryptomart.InstrumentType.PERPETUAL
        ]
        exchange_symbols[exchange] = list(
            zip(perpetual_instruments.symbol.to_list(), perpetual_instruments.instType.to_list())
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

    return symbols

def load_order_book_quantity_multiplier(id, exchange, symbol, instType):
    try:
        client._exchange_instance_map[exchange]._order_book_quantity_multiplier(instType, symbol)
    except Exception as e:
        tb = traceback.format_exc()
        errors.append((exchange, symbol, instType, tb))

def run_main_script():
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        executor.map(lambda args: load_order_book_quantity_multiplier(*args), get_all_instruments())
        
run_main_script()