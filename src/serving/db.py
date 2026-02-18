"""Database persistence for anomaly detection events.

Writes detection results to the PostgreSQL anomaly_events table.
Graceful: if DB is unavailable, logs a warning but doesn't fail the request.
"""

from __future__ import annotations

import logging
from datetime import datetime

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

# Module-level connection pool (initialized lazily)
_pool: psycopg2.pool.SimpleConnectionPool | None = None


def _get_pool(config: dict) -> psycopg2.pool.SimpleConnectionPool | None:
    """Get or create the connection pool. Returns None if DB is unavailable."""
    global _pool
    if _pool is not None:
        return _pool

    try:
        db_config = config.get("database", {})

        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=db_config.get("host", "localhost"),
            port=int(db_config.get("port", 5432)),
            dbname=db_config.get("name", "anomaly_db"),
            user=db_config.get("user", "anomaly_user"),
            password=db_config.get("password", "changeme_in_prod"),
        )
        logger.info("Database connection pool created.")
        return _pool
    except Exception as e:
        logger.warning("Could not create DB connection pool: %s", e)
        return None


def store_anomaly_event(
    config: dict,
    sensor_id: str,
    detected_at: datetime,
    anomaly_score: float,
    is_anomaly: bool,
    confidence: float,
    model_version: str,
) -> None:
    """Insert a single anomaly event into the database.

    Fails gracefully — logs a warning if the DB is unavailable.
    """
    pool = _get_pool(config)
    if pool is None:
        return

    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO anomaly_events
                    (sensor_id, detected_at, anomaly_score, is_anomaly, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (sensor_id, detected_at, anomaly_score, is_anomaly, confidence, model_version),
            )
        conn.commit()
    except Exception as e:
        logger.warning("Failed to store anomaly event: %s", e)
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            pool.putconn(conn)


def store_anomaly_events_batch(
    config: dict,
    events: list[dict],
) -> None:
    """Insert multiple anomaly events in a single transaction.

    Each dict must have keys: sensor_id, detected_at, anomaly_score,
    is_anomaly, confidence, model_version.

    Fails gracefully — logs a warning if the DB is unavailable.
    """
    if not events:
        return

    pool = _get_pool(config)
    if pool is None:
        return

    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            values = [
                (
                    e["sensor_id"],
                    e["detected_at"],
                    e["anomaly_score"],
                    e["is_anomaly"],
                    e["confidence"],
                    e["model_version"],
                )
                for e in events
            ]
            cur.executemany(
                """
                INSERT INTO anomaly_events
                    (sensor_id, detected_at, anomaly_score, is_anomaly, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                values,
            )
        conn.commit()
    except Exception as e:
        logger.warning("Failed to store batch anomaly events: %s", e)
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            pool.putconn(conn)


def close_pool() -> None:
    """Close the connection pool (call during app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed.")
