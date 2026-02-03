"""
AlloyDB implementation of usage metrics storage.
Uses PostgreSQL / AlloyDB for persistent metrics storage with SQL queries.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool, PoolError

from src.services.storage_interface import StorageInterface
from config import AppConfig

logger = logging.getLogger(__name__)


class AlloyDBMetricsStorage(StorageInterface):
    """
    AlloyDB-backed metrics storage.
    """

    TABLE_NAME = "usage_metrics"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.connection_pool: Optional[SimpleConnectionPool] = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize AlloyDB connection pool."""
        try:
            if not all([
                self.config.ALLOYDB_HOST,
                self.config.ALLOYDB_DB,
                self.config.ALLOYDB_USER,
                self.config.ALLOYDB_PASS,
            ]):
                logger.error(
                    "AlloyDB configuration incomplete - "
                    f"host={bool(self.config.ALLOYDB_HOST)}, "
                    f"db={bool(self.config.ALLOYDB_DB)}, "
                    f"user={bool(self.config.ALLOYDB_USER)}, "
                    f"pass={bool(self.config.ALLOYDB_PASS)}"
                )
                return

            sslmode = (
                getattr(self.config, "ALLOYDB_SSLMODE", None)
                or os.getenv("ALLOYDB_SSLMODE")
                or "require"
            )

            self.connection_params = {
                "host": self.config.ALLOYDB_HOST,
                "port": int(self.config.ALLOYDB_PORT or 5432),
                "database": self.config.ALLOYDB_DB,
                "user": self.config.ALLOYDB_USER,
                "password": self.config.ALLOYDB_PASS,
                "sslmode": sslmode,
            }

            attempts = 3
            backoff = 2
            last_exc = None

            for attempt in range(1, attempts + 1):
                try:
                    self.connection_pool = SimpleConnectionPool(
                        minconn=1,
                        maxconn=10,
                        **self.connection_params,
                    )
                    logger.info("✓ AlloyDB connection pool created")
                    last_exc = None
                    break
                except (PoolError, Exception) as e:
                    last_exc = e
                    logger.warning(
                        f"Attempt {attempt}/{attempts} failed creating pool: {e}"
                    )
                    time.sleep(backoff ** attempt)

            if last_exc:
                raise last_exc

            self.initialize_schema()

        except Exception as e:
            logger.error("Failed to initialize AlloyDB", exc_info=True)
            self.connection_pool = None

    def is_available(self) -> bool:
        """Check DB availability."""
        if not self.connection_pool:
            return False
        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return True
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize_schema(self) -> bool:
        """Create table and indexes if not present."""
        if not self.connection_pool:
            return False

        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMPTZ NOT NULL,
                            tribe_name VARCHAR(100) NOT NULL,
                            squad_name VARCHAR(100),
                            database_name VARCHAR(100),
                            user_id VARCHAR(100),
                            request_type VARCHAR(50) NOT NULL,
                            model_name VARCHAR(100) NOT NULL,
                            input_tokens INTEGER DEFAULT 0,
                            output_tokens INTEGER DEFAULT 0,
                            total_tokens INTEGER DEFAULT 0,
                            duration_ms INTEGER,
                            success BOOLEAN DEFAULT TRUE,
                            error_message TEXT,
                            session_id VARCHAR(100),
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_timestamp "
                        f"ON {self.TABLE_NAME}(timestamp);"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_tribe "
                        f"ON {self.TABLE_NAME}(tribe_name);"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_user "
                        f"ON {self.TABLE_NAME}(user_id);"
                    )

                conn.commit()

                # Partial unique index for session dedupe
                try:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                            ux_{self.TABLE_NAME}_session_request_tribe
                            ON {self.TABLE_NAME} (session_id, request_type, tribe_name)
                            WHERE session_id IS NOT NULL AND session_id <> '';
                        """)
                except Exception:
                    conn.autocommit = False
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS
                            ux_{self.TABLE_NAME}_session_request_tribe
                            ON {self.TABLE_NAME} (session_id, request_type, tribe_name)
                            WHERE session_id IS NOT NULL AND session_id <> '';
                        """)
                        conn.commit()
                finally:
                    conn.autocommit = False

                return True
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Schema initialization failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Insert APIs
    # ------------------------------------------------------------------

    def store_metric(self, metric: Dict[str, Any]) -> bool:
        return self.insert_metrics_batch([metric])

    def insert_metrics_batch(self, metrics: List[Dict[str, Any]]) -> bool:
        if not metrics or not self.is_available():
            return False

        try:
            conn = self.connection_pool.getconn()
            try:
                values = []
                for m in metrics:
                    values.append((
                        m["timestamp"],
                        m.get("tribe_name", "").lower(),
                        m.get("squad_name"),
                        m.get("database_name"),
                        m.get("user_id"),
                        m["request_type"],
                        m["model_name"],
                        int(m.get("input_tokens", 0)),
                        int(m.get("output_tokens", 0)),
                        int(m.get("total_tokens", 0)),
                        int(m.get("duration_ms", 0)) if m.get("duration_ms") else None,
                        m.get("success", True),
                        m.get("error_message"),
                        m.get("session_id"),
                    ))

                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {self.TABLE_NAME}
                        (timestamp, tribe_name, squad_name, database_name, user_id,
                         request_type, model_name, input_tokens, output_tokens,
                         total_tokens, duration_ms, success, error_message, session_id)
                        VALUES %s
                        ON CONFLICT (session_id, request_type, tribe_name) DO NOTHING
                        """,
                        values,
                    )
                conn.commit()
                return True
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Insert metrics failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Query APIs
    # ------------------------------------------------------------------

    def query_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []

        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = f"""
                        SELECT *
                        FROM {self.TABLE_NAME}
                        WHERE timestamp >= %s AND timestamp <= %s
                    """
                    params = [start_date, end_date]

                    if tribe_name and tribe_name != "All Tribes":
                        query += " AND tribe_name = %s"
                        params.append(tribe_name.lower())

                    query += " ORDER BY timestamp DESC"
                    cur.execute(query, params)
                    return cur.fetchall()
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Query metrics failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_summary_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not self.is_available():
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "avg_duration_ms": 0,
                "unique_users": 0,
                "success_count": 0,
                "error_count": 0,
            }

        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = f"""
                        SELECT
                            COUNT(*) AS total_requests,
                            SUM(input_tokens) AS total_input_tokens,
                            SUM(output_tokens) AS total_output_tokens,
                            SUM(total_tokens) AS total_tokens,
                            AVG(duration_ms) AS avg_duration_ms,
                            COUNT(DISTINCT user_id) AS unique_users,
                            SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
                            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS error_count
                        FROM {self.TABLE_NAME}
                        WHERE timestamp >= %s AND timestamp <= %s
                    """
                    params = [start_date, end_date]

                    if tribe_name and tribe_name != "All Tribes":
                        query += " AND tribe_name = %s"
                        params.append(tribe_name.lower())

                    cur.execute(query, params)
                    r = cur.fetchone()

                    return {
                        "total_requests": r["total_requests"] or 0,
                        "total_input_tokens": r["total_input_tokens"] or 0,
                        "total_output_tokens": r["total_output_tokens"] or 0,
                        "total_tokens": r["total_tokens"] or 0,
                        "avg_duration_ms": float(r["avg_duration_ms"] or 0),
                        "unique_users": r["unique_users"] or 0,
                        "success_count": r["success_count"] or 0,
                        "error_count": r["error_count"] or 0,
                    }
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Failed to get summary stats", exc_info=True)
            return {}

    def get_daily_aggregates(
        self,
        start_date: datetime,
        end_date: datetime,
        tribe_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not self.is_available():
            return []

        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = f"""
                        SELECT
                            DATE(timestamp) AS date,
                            COUNT(*) AS requests,
                            SUM(input_tokens) AS input_tokens,
                            SUM(output_tokens) AS output_tokens,
                            SUM(total_tokens) AS total_tokens,
                            COUNT(DISTINCT user_id) AS unique_users
                        FROM {self.TABLE_NAME}
                        WHERE timestamp >= %s AND timestamp <= %s
                    """
                    params = [start_date, end_date]

                    if tribe_name and tribe_name != "All Tribes":
                        query += " AND tribe_name = %s"
                        params.append(tribe_name.lower())

                    query += " GROUP BY DATE(timestamp) ORDER BY date"
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    return [{
                        "date": r["date"].strftime("%Y-%m-%d"),
                        "requests": r["requests"],
                        "input_tokens": r["input_tokens"] or 0,
                        "output_tokens": r["output_tokens"] or 0,
                        "total_tokens": r["total_tokens"] or 0,
                        "unique_users": r["unique_users"] or 0,
                    } for r in rows]
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Failed to get daily aggregates", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_total_count(self) -> int:
        """Return total number of stored metrics."""
        if not self.is_available():
            return 0

        try:
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")
                    return cur.fetchone()[0]
            finally:
                self.connection_pool.putconn(conn)
        except Exception:
            logger.error("Failed to get total count", exc_info=True)
            return 0

    def close(self):
        """Close all DB connections."""
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                logger.info("✓ AlloyDB connection pool closed")
            except Exception:
                logger.error("Error closing connection pool", exc_info=True)
