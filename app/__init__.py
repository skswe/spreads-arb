import logging

from .core import Client

root_logger = logging.getLogger("app")
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())
root_logger.handlers[0].setFormatter(
    logging.Formatter("{levelname:<8} {name:>30}{funcName:>35}:{lineno:<4} {message}", style="{")
)


def set_log_level(level):
    logging.getLogger("cryptomart").setLevel(level)
    logging.getLogger("pyutil").setLevel(level)
    logging.getLogger("app").setLevel(level)
