import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from maple_vision.knowledge import initialize, migrate, open_readonly


def test_migration_preserves_existing_v1_records() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO source_records VALUES (?, ?, ?, ?, ?)",
            ("fixture", "fixture://sample", "test-v1", "abc", "PROVISIONAL"),
        )
        connection.commit()
        before = connection.execute("SELECT * FROM source_records").fetchall()
        migrate(connection)
        assert connection.execute("SELECT * FROM source_records").fetchall() == before
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)


@pytest.mark.parametrize("version", [0, 1])
def test_incompatible_schema_is_rejected_without_changes(version: int) -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE source_records (id TEXT)")
        connection.execute("INSERT INTO source_records VALUES ('keep-me')")
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
        before = connection.execute("SELECT sql FROM sqlite_schema").fetchall()
        with pytest.raises(ValueError, match="Incompatible"):
            migrate(connection)
        assert connection.execute("SELECT sql FROM sqlite_schema").fetchall() == before
        assert connection.execute("SELECT * FROM source_records").fetchall() == [("keep-me",)]
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)


def test_version_one_missing_table_is_rejected() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA user_version = 1")
        with pytest.raises(ValueError, match="Incompatible"):
            migrate(connection)
        assert not connection.execute("SELECT name FROM sqlite_schema").fetchall()


@pytest.mark.parametrize("version", [-1, 2])
def test_unsupported_version_leaves_database_unchanged(version: int) -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
        with pytest.raises(ValueError, match="Unsupported"):
            migrate(connection)
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)
        assert not connection.execute("SELECT name FROM sqlite_schema").fetchall()


def test_migration_rollback_includes_created_table_and_version() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:

        def deny_version_write(
            action: int,
            arg1: str | None,
            arg2: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_PRAGMA and arg1 == "user_version" and arg2 is not None:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(deny_version_write)
        with pytest.raises(sqlite3.DatabaseError):
            migrate(connection)
        connection.set_authorizer(None)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert not connection.execute("SELECT name FROM sqlite_schema").fetchall()
        assert not connection.in_transaction


def test_migration_does_not_commit_callers_transaction() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE caller_data (value TEXT)")
        connection.execute("INSERT INTO caller_data VALUES ('uncommitted')")
        migrate(connection)
        assert connection.in_transaction
        connection.rollback()
        assert connection.execute("SELECT * FROM caller_data").fetchall() == []
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE name = 'source_records'"
            ).fetchall()
            == []
        )


def test_failed_migration_does_not_rollback_callers_work() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE source_records (id TEXT)")
        connection.execute("INSERT INTO source_records VALUES ('uncommitted')")
        with pytest.raises(ValueError, match="Incompatible"):
            migrate(connection)
        assert connection.in_transaction
        assert connection.execute("SELECT * FROM source_records").fetchall() == [("uncommitted",)]
        connection.rollback()
        assert connection.execute("SELECT * FROM source_records").fetchall() == []


@pytest.mark.parametrize("status", ["VERIFIED", "PROVISIONAL", "UNKNOWN", "CONFLICTED"])
def test_known_verification_statuses_are_accepted(status: str) -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        migrate(connection)
        connection.execute(
            "INSERT INTO source_records VALUES (?, ?, ?, ?, ?)",
            ("record", "fixture://sample", "v1", "abc", status),
        )
        assert connection.execute("SELECT verification_status FROM source_records").fetchone() == (
            status,
        )


@pytest.mark.parametrize("status", ["SUCCESS", "verified", "", None])
def test_invalid_verification_status_is_rejected(status: str | None) -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        migrate(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO source_records VALUES (?, ?, ?, ?, ?)",
                ("record", "fixture://sample", "v1", "abc", status),
            )


def test_initialize_and_readonly_support_encoded_paths(tmp_path: Path) -> None:
    path = tmp_path / "knowledge # 中文.sqlite"
    initialize(path)
    initialize(path)
    with closing(open_readonly(path)) as connection:
        assert connection.execute("SELECT count(*) FROM source_records").fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute(
                "INSERT INTO source_records VALUES ('a', 'source', 'v1', 'abc', 'UNKNOWN')"
            )


def test_readonly_does_not_create_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        open_readonly(path)
    assert not path.exists()


def test_readonly_does_not_migrate_an_empty_database(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite"
    with closing(sqlite3.connect(path)):
        pass
    with pytest.raises(ValueError, match="initialize"):
        open_readonly(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert not connection.execute("SELECT name FROM sqlite_schema").fetchall()


def test_missing_status_constraint_is_not_adopted() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            """CREATE TABLE source_records (
                id TEXT PRIMARY KEY,
                source_uri TEXT NOT NULL,
                game_version TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                verification_status TEXT NOT NULL
            )"""
        )
        with pytest.raises(ValueError, match="Incompatible"):
            migrate(connection)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
