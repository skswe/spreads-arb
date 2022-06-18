from cryptomart.enums import Symbol
from datetime import date

# These symbols represent exchange indices and not unique tradable assets
BLACKLISTED_SYMBOLS = [Symbol.DEFI, Symbol.PRIV, Symbol.ALT, Symbol.EXCH, Symbol.SHIT, Symbol.MID]
STUDY_START_DATE = date(2019, 1, 1)
