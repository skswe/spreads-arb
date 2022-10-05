import logging

from cryptomart.enums import InstrumentType

LOGGING_FORMATTER = logging.Formatter(
    "{asctime}.{msecs:03.0f} {levelname:<8} {name:<15}{funcName:>15}:{lineno:<4} {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BLACKLISTED_SYMBOLS = [
    # These symbols represent exchange indices and not unique tradable assets
    "DEFI",
    "PRIV",
    "ALT",
    "EXCH",
    "SHIT",
    "MID",
    # These symbols have been exploited
    "LUNA",
]
STUDY_START_DATE = (2019, 1, 1)
STUDY_END_DATE = (2022, 6, 19)
STUDY_TIME_RANGE = (STUDY_START_DATE, STUDY_END_DATE)
STUDY_INST_TYPES = [InstrumentType.PERPETUAL]
