"""Run database migrations against PostgreSQL.

Usage:
    python migrations/run.py
    python migrations/run.py --host localhost --port 5432 --db anomaly_db
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import psycopg2

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent


def get_connection(
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> psycopg2.extensions.connection:
    """Create a PostgreSQL connection using args or env vars."""
    conn = psycopg2.connect(
        host=host or os.environ.get("POSTGRES_HOST", "localhost"),
        port=port or int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=dbname or os.environ.get("POSTGRES_DB", "anomaly_db"),
        user=user or os.environ.get("POSTGRES_USER", "anomaly_user"),
        password=password or os.environ.get("POSTGRES_PASSWORD", "changeme_in_prod"),
    )
    conn.autocommit = False
    return conn


def get_applied_versions(conn: psycopg2.extensions.connection) -> set[int]:
    """Get already-applied migration versions, or empty set if table doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations")
            return {row[0] for row in cur.fetchall()}
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return set()


def run_migrations(
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Execute all pending SQL migrations in order."""
    conn = get_connection(host, port, dbname, user, password)

    try:
        applied = get_applied_versions(conn)
        logger.info("Already applied migrations: %s", applied or "none")

        # Find all .sql migration files, sorted by version number
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            # Extract version from filename: 001_create_... → 1
            version = int(migration_file.stem.split("_")[0])

            if version in applied:
                logger.info("Skipping migration %03d (already applied)", version)
                continue

            logger.info("Applying migration %03d: %s", version, migration_file.name)
            sql = migration_file.read_text()

            with conn.cursor() as cur:
                cur.execute(sql)

            conn.commit()
            logger.info("Migration %03d applied successfully", version)

        logger.info("All migrations complete.")

    except Exception:
        conn.rollback()
        logger.exception("Migration failed, rolled back")
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    run_migrations(
        host=args.host,
        port=args.port,
        dbname=args.db,
        user=args.user,
        password=args.password,
    )


if __name__ == "__main__":
    main()
