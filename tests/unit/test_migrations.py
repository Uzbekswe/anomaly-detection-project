"""Unit tests for migrations/run.py.

All database interactions are mocked — no real PostgreSQL needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from migrations.run import get_applied_versions, get_connection, run_migrations

# ──────────────────────────────────────────────
# TestGetConnection
# ──────────────────────────────────────────────


class TestGetConnection:
    @patch("migrations.run.psycopg2.connect")
    def test_uses_env_defaults(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with patch.dict("os.environ", {}, clear=True):
            get_connection()

        mock_connect.assert_called_once_with(
            host="localhost",
            port=5432,
            dbname="anomaly_db",
            user="anomaly_user",
            password="changeme_in_prod",
        )
        assert mock_conn.autocommit is False

    @patch("migrations.run.psycopg2.connect")
    def test_accepts_explicit_args(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        get_connection(
            host="myhost",
            port=5433,
            dbname="mydb",
            user="myuser",
            password="mypass",
        )

        mock_connect.assert_called_once_with(
            host="myhost",
            port=5433,
            dbname="mydb",
            user="myuser",
            password="mypass",
        )


# ──────────────────────────────────────────────
# TestGetAppliedVersions
# ──────────────────────────────────────────────


class TestGetAppliedVersions:
    def test_returns_set_of_versions(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [(1,), (2,), (3,)]

        result = get_applied_versions(mock_conn)
        assert result == {1, 2, 3}

    def test_returns_empty_set_when_table_missing(self) -> None:
        import psycopg2.errors

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.side_effect = psycopg2.errors.UndefinedTable("table does not exist")

        result = get_applied_versions(mock_conn)
        assert result == set()
        mock_conn.rollback.assert_called_once()


# ──────────────────────────────────────────────
# TestRunMigrations
# ──────────────────────────────────────────────


class TestRunMigrations:
    @patch("migrations.run.get_connection")
    @patch("migrations.run.get_applied_versions")
    @patch("migrations.run.MIGRATIONS_DIR")
    def test_applies_pending_migrations(
        self,
        mock_migrations_dir: MagicMock,
        mock_get_applied: MagicMock,
        mock_get_conn: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        # Create fake migration files
        mig1 = tmp_path / "001_create_table.sql"
        mig1.write_text("CREATE TABLE test;")
        mig2 = tmp_path / "002_add_column.sql"
        mig2.write_text("ALTER TABLE test ADD col INT;")

        mock_migrations_dir.glob.return_value = sorted([mig1, mig2])

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_get_applied.return_value = set()  # No migrations applied yet

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        run_migrations()

        # Both migrations should be executed
        assert mock_cursor.execute.call_count == 2
        assert mock_conn.commit.call_count == 2

    @patch("migrations.run.get_connection")
    @patch("migrations.run.get_applied_versions")
    @patch("migrations.run.MIGRATIONS_DIR")
    def test_skips_already_applied(
        self,
        mock_migrations_dir: MagicMock,
        mock_get_applied: MagicMock,
        mock_get_conn: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        mig1 = tmp_path / "001_create_table.sql"
        mig1.write_text("CREATE TABLE test;")
        mig2 = tmp_path / "002_add_column.sql"
        mig2.write_text("ALTER TABLE test ADD col INT;")

        mock_migrations_dir.glob.return_value = sorted([mig1, mig2])

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_get_applied.return_value = {1}  # Migration 1 already applied

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        run_migrations()

        # Only migration 2 should be executed
        assert mock_cursor.execute.call_count == 1
        mock_cursor.execute.assert_called_once_with("ALTER TABLE test ADD col INT;")
        assert mock_conn.commit.call_count == 1
