import logging

logger = logging.getLogger(__name__)
try:
    import vectorbt

    from . import vbt_backtest, vbt_backtest_chained, vbt_bt_chained_acc_slip, vbt_bt_quotes
    from .BacktestResult import BacktestResult
    from .BacktestRunner import BacktestRunner
    from .ChainedBacktestResult import ChainedBacktestResult
except ImportError:
    logger.warning("vectorbt module not found. Backtest functionality disabled.")
