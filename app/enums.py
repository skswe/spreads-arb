from cryptomart import enums as cm_enums

from .globals import BLACKLISTED_SYMBOLS, STUDY_INST_TYPES


class OHLCVColumn(cm_enums.OHLCVColumn):
    pass


class SpreadColumn(OHLCVColumn):
    pass


class Exchange(cm_enums.Exchange):
    pass


class Symbol(cm_enums.Symbol):
    pass


class InstrumentType(cm_enums.InstrumentType):
    pass


class Interval(cm_enums.Interval):
    pass


for symbol in BLACKLISTED_SYMBOLS:
    delattr(Symbol, symbol)

_blacklist_inst_types = set(InstrumentType) - set(STUDY_INST_TYPES)
for inst_type in _blacklist_inst_types:
    delattr(InstrumentType, inst_type.upper())
