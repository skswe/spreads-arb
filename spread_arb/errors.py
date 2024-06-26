"""This module contains the error classes for the backtest.
"""

from cryptomart.errors import APIError as ExchangeAPIError
from cryptomart.errors import MissingDataError, NotSupportedError
from google.api_core.exceptions import GoogleAPIError as BigQueryError
