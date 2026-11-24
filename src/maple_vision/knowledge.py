"""P0 SQLite bootstrap, with transaction ownership left to the caller."""

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = 1

_SOURCE_RECORDS_SQL = """CREATE TABLE source_records (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    game_version TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    verification_status TEXT NOT NULL CHECK (
        verification_status IN ('VERIFIED', 'PROVISIONAL', 'UNKNOWN', 'CONFLICTED')
    )
)"""


def _normalized_sql(sql: str) -> str:
    """Ignore formatting and SQL keyword case, but preserve string literals.

    This is intentionally a strict v1 shape check, not a general SQL parser.
    A changed schema needs an explicit migration, rather than silent adoption.
    """
    tokens = re.findall(r"'(?:''|[^'])*'|[^\s]", sql)
    normalized = "".join(token if token.startswith("'") else token.lower() for token in tokens)
    return normalized.replace("ifnotexists", "").rstrip(";")


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA main.user_version").fetchone()
    if row is None:
        raise ValueError("Cannot read SQLite schema version")
    return int(row[0])


def _validate_v1(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT type, sql FROM main.sqlite_schema WHERE name = 'source_records'"
    ).fetchone()
    if (
        row is None
        or row[0] != "table"
        or not isinstance(row[1], str)
        or _normalized_sql(row[1]) != _normalized_sql(_SOURCE_RECORDS_SQL)
    ):
        raise ValueError("Incompatible source_records schema; expected P0 schema v1")


def migrate(connection: sqlite3.Connection) -> None:
    """Create or validate schema v1, atomically, without owning caller transactions.

    An outermost savepoint commits this migration only. Inside an existing
    transaction it is released without committing the caller's work; the caller
    may still roll everything back. Both DDL and user_version are rolled back on
    failure. Existing v1 data and constraints remain unchanged.
    """
    savepoint = "maple_migration_" + uuid4().hex
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        version = _schema_version(connection)
        if version not in (0, SCHEMA_VERSION):
            raise ValueError(f"Unsupported schema version: {version}")
        if version == 0:
            existing = connection.execute(
                "SELECT 1 FROM main.sqlite_schema WHERE name = 'source_records'"
            ).fetchone()
            if existing is None:
                connection.execute(_SOURCE_RECORDS_SQL)
            _validate_v1(connection)
            connection.execute(f"PRAGMA main.user_version = {SCHEMA_VERSION}")
        else:
            _validate_v1(connection)
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
    except BaseException:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def initialize(path: str | Path) -> None:
    """Initialize or validate a database file; its parent directory must exist."""
    with closing(sqlite3.connect(path)) as connection:
        migrate(connection)


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing v1 file for reads only; the caller must close it.

    URI encoding handles spaces, non-ASCII names, '?' and '#'. This does not
    assume the file is immutable, so SQLite still handles its journal normally.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        if _schema_version(connection) != SCHEMA_VERSION:
            raise ValueError("Read-only access requires schema v1; initialize it separately")
        _validate_v1(connection)
    except BaseException:
        connection.close()
        raise
    return connection
