import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    with connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > 1:
            raise ValueError("Unsupported schema version")
        if version == 0:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS source_records (
                    id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    game_version TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    verification_status TEXT NOT NULL CHECK (
                        verification_status IN (
                            'VERIFIED', 'PROVISIONAL', 'UNKNOWN', 'CONFLICTED'
                        )
                    )
                )"""
            )
            connection.execute("PRAGMA user_version = 1")
