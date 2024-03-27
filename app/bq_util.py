"""DEPRECATED - This module contains utility functions for working with Google Cloud BigQuery.
"""

import logging
import os

import dotenv
import pandas as pd
from cryptomart import Exchange, InstrumentType
from .util import cached

logger = logging.getLogger(__name__)

try:
    from google.api_core.exceptions import GoogleAPIError
    from google.cloud.bigquery import Client, Dataset, QueryJob, Table

    dotenv.load_dotenv()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["BQ_CRED"]

    # Requires $GOOGLE_APPLICATION_CREDENTIALS to be set
    client = Client()

    def to_df(result: QueryJob) -> pd.DataFrame:
        """Convert a bigquery QueryJob result to a pandas DataFrame"""
        return pd.DataFrame([dict(row) for row in result])

    def get_dataset(exchange: Exchange) -> Dataset:
        """Get reference to order book dataset for a given exchange. Throws GoogleAPIError if dataset does not exist"""
        assert exchange in Exchange
        return client.get_dataset(f"{exchange}_order_book")

    def get_table(exchange: Exchange, symbol: str, inst_type: InstrumentType, legacy=False) -> Table:
        """Get reference to a order book table

        Args:
            exchange (Exchange): exchange
            symbol (str): symbol
            inst_type (InstrumentType): instrument type
            legacy (bool, optional): If True, use legacy table name (uppercase inst_type). Defaults to False.
        """
        assert inst_type in InstrumentType
        if legacy:
            inst_type = inst_type.upper()
        dataset = get_dataset(exchange)
        return client.get_table(f"{dataset.dataset_id}.{symbol}_{inst_type}")

    def run_query(exchange: Exchange, symbol: str, inst_type: InstrumentType, qry: str) -> pd.DataFrame:
        """Run query against a single order book table.

        Args:
            exchange (Exchange): exchange
            symbol (str): symbol
            inst_type (InstrumentType): inst_type
            qry (str): SQL query to run against the table. Use {table_id} in the query string to reference the table.
        """
        table = get_table(exchange, symbol, inst_type)
        table_id = f"{table.dataset_id}.{table.table_id}"
        result = client.query(qry.format(table_id=table_id))
        return pd.DataFrame(dict(row) for row in result)

    def migrate_table(exchange: Exchange, symbol: str, inst_type: InstrumentType):
        """Migrate legacy table structure to new table structure

        Args:
            exchange (Exchange): exchange
            symbol (str): symbol
            inst_type (InstrumentType): instrument type
        """
        target_table = get_table(exchange, symbol, inst_type)
        source_table = get_table(exchange, symbol, inst_type, legacy=True)

        target_table_id = f"{target_table.dataset_id}.{target_table.table_id}"
        source_table_id = f"{source_table.dataset_id}.{source_table.table_id}"

        qry = f"""
            MERGE {target_table_id} target
            USING {source_table_id} source
            ON target.timestamp=source.timestamp AND target.side=source.side
            WHEN NOT MATCHED 
            THEN INSERT 
                (price, quantity, side, timestamp)
            VALUES (source.price, source.quantity, source.side, source.timestamp);

            DROP TABLE {source_table_id}
        """
        return client.query(qry)

    @cached(
        os.path.join(os.getenv("SA_CACHE_PATH", "/tmp/cache"), "bid_ask_spread"),
        path_seperators=["exchange", "inst_type"],
    )
    def get_bid_ask_spread(exchange: Exchange, symbol: str, inst_type: InstrumentType, **cache_kwargs) -> pd.DataFrame:
        """Get daily bid-ask spread averages from order book table

        Args:
            exchange (Exchange): exchange
            symbol (str): symbol
            inst_type (InstrumentType): instrument type
        """
        qry = """
            -- Bid/ask spread by timestamp (8 hour intervals) 
            WITH spreads AS (
                SELECT
                    -- 4. Take spread as difference between average ask and bid
                    *,
                    avg_ask - avg_bid as spread
                FROM
                    (
                        SELECT
                            -- 2. Take weighted average over timestamp
                            DISTINCT timestamp,
                            side,
                            SUM(price * weight) OVER (PARTITION BY timestamp, side) as w_avg_price
                        FROM
                            (
                                SELECT
                                    -- 1. Use quantity / SUM(quantity) as price-weight
                                    timestamp,
                                    side,
                                    quantity,
                                    price,
                                    quantity / SUM(quantity) OVER (PARTITION BY timestamp, side) AS weight
                                FROM
                                    {table_id}
                            )
                    ) PIVOT (
                        -- 3. Pivot side column into avg_ask and avg_bid
                        MAX(w_avg_price) for side in ("a" avg_ask, "b" avg_bid)
                    )
            )
            SELECT
                -- 5. Aggregate spread by date
                DATE(timestamp) as date,
                AVG(spread) as bid_ask_spread
            FROM
                spreads
            GROUP BY
                date
            ORDER BY
                date
        """
        return run_query(exchange, symbol, inst_type, qry).apply(
            lambda c: pd.to_datetime(c) if c.name == "date" else c
        )

    @cached(
        os.path.join(os.getenv("SA_CACHE_PATH", "/tmp/cache"), "order_book_stats"),
        path_seperators=["exchange", "inst_type"],
    )
    def get_order_book_stats(
        exchange: Exchange, symbol: str, inst_type: InstrumentType, **cache_kwargs
    ) -> pd.DataFrame:
        """Get gaps and first_date from order book table

        Args:
            exchange (Exchange): exchange
            symbol (str): symbol
            inst_type (InstrumentType): instrument type
        """
        qry = """
            WITH dates AS
                (
                    SELECT DISTINCT DATE(timestamp) as date FROM {table_id}
                ),

            lags AS
                (
                    SELECT DATE_DIFF(date, LAG(date, 1) OVER (ORDER BY date), DAY) as diff FROM dates
                ),

            gaps AS
                (
                    SELECT MAX(gaps) AS gaps FROM
                    ((SELECT count(diff) as gaps FROM lags WHERE diff > 1 GROUP BY diff) UNION ALL (SELECT 0 gaps))
                ),

            first_date AS
                (
                    SELECT MIN(date) as first_date FROM dates
                )

            SELECT DISTINCT
                gaps.gaps,
                first_date.first_date
            FROM
                gaps CROSS JOIN first_date
        """
        return run_query(exchange, symbol, inst_type, qry)

except ImportError:
    logger.warning("Google Cloud BigQuery not installed. bq_util will not work.")
