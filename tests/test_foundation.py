import sqlite3

import pytest

from maple_vision.contracts import Frame, Status
from maple_vision.knowledge import migrate
from maple_vision.pipeline import NoOpPerception, run_once


class Source:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame


def test_noop_preserves_identity_and_unknown():
    result = run_once(Source(Frame(7, 100, b"")), NoOpPerception())
    assert result is not None
    assert (result.sequence, result.timestamp_ms, result.status) == (
        7, 100, Status.UNKNOWN
    )


def test_empty_source():
    assert run_once(Source(None), NoOpPerception()) is None


def test_migration_idempotent():
    with sqlite3.connect(":memory:") as conn:
        migrate(conn)
        migrate(conn)
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM source_records").fetchone() == (0,)


def test_future_schema_rejected():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA user_version = 2")
        with pytest.raises(ValueError):
            migrate(conn)
