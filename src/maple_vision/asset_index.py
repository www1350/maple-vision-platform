"""A bounded SQLite discovery index over verified, versioned registry assets.

The registry remains the content authority. Every returned entry is checked
against its metadata record and decoded source bytes; the index never upgrades
an importer's permission assertion into proof of ownership.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .asset_registry import MAX_ASSET_BYTES, AssetEntry
from .asset_registry import _decode as _decode_image
from .catalog import canonical_json, strict_json, validate_provenance

INDEX_SCHEMA_VERSION = 1
INDEX_APPLICATION_ID = 0x4D564149
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_INDEX_ENTRIES = 4096
MAX_INDEX_BATCH = 256
MAX_METADATA_BYTES = 64 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TABLE_SQL = """CREATE TABLE asset_entries (
    asset_id TEXT PRIMARY KEY NOT NULL CHECK (length(asset_id) = 64),
    entity_id TEXT NOT NULL,
    game_version TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json))
)"""
_INDEX_SQL = "CREATE INDEX asset_entity_version ON asset_entries(entity_id, game_version, asset_id)"
_COLUMNS = "asset_id, entity_id, game_version, sha256, metadata_json"


@dataclass(frozen=True)
class IndexResult:
    imported: int
    unchanged: int
    total: int


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError("asset index ID must be a lowercase SHA256")
    return value


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 4096
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"invalid asset index {field}")
    return value


def _sql(value: str) -> str:
    return " ".join(value.split()).casefold()


def _schema(connection: sqlite3.Connection) -> None:
    if (
        connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete"
        or connection.execute("PRAGMA application_id").fetchone()[0] != INDEX_APPLICATION_ID
        or connection.execute("PRAGMA user_version").fetchone()[0] != INDEX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported asset index identity or schema version")
    rows = connection.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name NOT GLOB 'sqlite_*' ORDER BY name LIMIT 3"
    ).fetchall()
    expected = sorted(
        [
            ("table", "asset_entries", _sql(_TABLE_SQL)),
            ("index", "asset_entity_version", _sql(_INDEX_SQL)),
        ],
        key=lambda item: item[1],
    )
    actual = [(kind, name, _sql(sql or "")) for kind, name, sql in rows]
    if actual != expected:
        raise ValueError("asset index SQLite schema differs from its declared contract")
    count = connection.execute("SELECT count(*) FROM asset_entries").fetchone()[0]
    if count > MAX_INDEX_ENTRIES:
        raise ValueError("asset index exceeds its entry budget")
    oversized = connection.execute(
        "SELECT 1 FROM asset_entries WHERE "
        "typeof(asset_id) != 'text' OR length(asset_id) != 64 OR "
        "typeof(entity_id) != 'text' OR length(entity_id) > 4096 OR "
        "typeof(game_version) != 'text' OR length(game_version) > 4096 OR "
        "typeof(sha256) != 'text' OR length(sha256) != 64 OR "
        "typeof(metadata_json) != 'text' OR length(CAST(metadata_json AS BLOB)) > ? LIMIT 1",
        (MAX_METADATA_BYTES,),
    ).fetchone()
    if oversized is not None:
        raise ValueError("asset index row exceeds field type or byte budget")
    if connection.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
        raise ValueError("asset index SQLite integrity check failed")


@contextmanager
def _connection(path: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("asset index must be a regular file")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("asset index must be closed without SQLite sidecar files")
    exists = path.exists()
    if exists and path.stat().st_size > MAX_INDEX_BYTES:
        raise ValueError("asset index exceeds its byte budget")
    if exists:
        # SQLite's read/write version bytes are 2 in WAL mode. Check the small
        # fixed header before SQLite can recreate absent WAL/SHM sidecars.
        with path.open("rb") as stream:
            header = stream.read(20)
        if header[:16] != b"SQLite format 3\x00" or header[18:20] != b"\x01\x01":
            raise ValueError("asset index requires a non-WAL SQLite format")
    if not exists and not write:
        raise ValueError("asset index does not exist")
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection | None = None
    staging: Path | None = None
    try:
        database_path = path
        if write and not exists:
            staging = path.parent / f".asset-index-{uuid4().hex}.sqlite"
            staging.touch(exist_ok=False)
            database_path = staging
        connection = (
            sqlite3.connect(database_path)
            if write
            else sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        if write:
            connection.execute("BEGIN IMMEDIATE")
            if not exists:
                connection.execute(f"PRAGMA application_id = {INDEX_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
                connection.execute(_TABLE_SQL)
                connection.execute(_INDEX_SQL)
        _schema(connection)
        yield connection
        if write:
            page_count = connection.execute("PRAGMA page_count").fetchone()[0]
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            if page_count * page_size > MAX_INDEX_BYTES:
                raise ValueError("asset index exceeds its byte budget")
            connection.commit()
            if staging is not None:
                connection.close()
                connection = None
                try:
                    os.link(staging, path)
                except FileExistsError as exc:
                    raise ValueError(
                        "asset index was created concurrently; retry the batch"
                    ) from exc
    except sqlite3.Error as exc:
        raise ValueError("asset index database cannot be verified") from exc
    finally:
        if connection is not None:
            connection.close()
        if staging is not None:
            staging.unlink(missing_ok=True)


def _bounded(path: Path, budget: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("asset index source must be a regular, nonsymlink file")
    with path.open("rb") as stream:
        data = stream.read(budget + 1)
    if not data or len(data) > budget:
        raise ValueError("asset index source exceeds its byte budget or is empty")
    return data


def _verified(registry: Path, asset_id: str) -> AssetEntry:
    """Use the registry's identity and decoder, with bounded reads before hashes."""
    _identifier(asset_id)
    try:
        for directory in (registry, registry / "records", registry / "objects"):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("invalid asset index source registry directory")
        raw = _bounded(registry / "records" / f"{asset_id}.json", MAX_METADATA_BYTES)
        metadata = strict_json(raw.decode("utf-8"))
        if not isinstance(metadata, dict) or set(metadata) != set(AssetEntry.__dataclass_fields__):
            raise ValueError("invalid asset index source metadata fields")
        entry = AssetEntry(**metadata)
        _identifier(entry.sha256)
        _text(entry.entity_id, "entity_id")
        _text(entry.game_version, "game_version")
        validate_provenance(entry.source_uri, entry.permission, entry.license)
        payload = {key: value for key, value in metadata.items() if key != "asset_id"}
        if (
            entry.asset_id != asset_id
            or hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest() != asset_id
        ):
            raise ValueError("asset index source metadata SHA256 mismatch")
        formats = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
        if (
            not isinstance(entry.media_type, str)
            or entry.media_type not in formats
            or entry.relative_path != f"objects/{entry.sha256}.{formats[entry.media_type]}"
            or any(type(value) is not int or value <= 0 for value in (entry.width, entry.height))
        ):
            raise ValueError("invalid asset index source path or image metadata")
        data = _bounded(registry / entry.relative_path, MAX_ASSET_BYTES)
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise ValueError("asset index source content SHA256 mismatch")
        width, height, extension, media_type = _decode_image(data)
        if (width, height, extension, media_type) != (
            entry.width,
            entry.height,
            formats[entry.media_type],
            entry.media_type,
        ):
            raise ValueError("asset index source decoded metadata mismatch")
        return entry
    except (OSError, UnicodeError) as exc:
        raise ValueError("asset index source registry is missing or unreadable") from exc


def _record(entry: AssetEntry) -> tuple[str, str, str, str, str]:
    metadata = canonical_json(asdict(entry))
    if len(metadata.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("asset index metadata exceeds its byte budget")
    return entry.asset_id, entry.entity_id, entry.game_version, entry.sha256, metadata


def _decode(row: tuple[Any, ...], registry: Path) -> AssetEntry:
    if len(row) != 5 or any(not isinstance(value, str) for value in row):
        raise ValueError("invalid asset index row types")
    asset_id = _identifier(row[0])
    if len(row[4].encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("asset index metadata exceeds its byte budget")
    metadata = strict_json(row[4])
    if not isinstance(metadata, dict) or set(metadata) != set(AssetEntry.__dataclass_fields__):
        raise ValueError("invalid asset index metadata fields")
    entry = _verified(registry, asset_id)
    if row != _record(entry):
        raise ValueError("asset index row disagrees with verified registry content")
    return entry


def index_assets(database: Path, registry: Path, asset_ids: tuple[str, ...]) -> IndexResult:
    """Atomically index a bounded, explicit batch of verified registry IDs.

    Repeating an identical ID is idempotent across calls. Duplicates inside one
    batch are rejected, and a failed source verification creates no database.
    Existing rows are immutable and contradictory entries are not overwritten.
    """
    if (
        not isinstance(asset_ids, tuple)
        or not 1 <= len(asset_ids) <= MAX_INDEX_BATCH
        or any(not isinstance(value, str) or _HASH.fullmatch(value) is None for value in asset_ids)
        or len(set(asset_ids)) != len(asset_ids)
    ):
        raise ValueError("asset index batch must contain 1..256 distinct SHA256 IDs")
    entries = tuple(_verified(Path(registry), asset_id) for asset_id in asset_ids)
    records = tuple(_record(entry) for entry in entries)
    imported = unchanged = 0
    with _connection(Path(database), write=True) as connection:
        total = connection.execute("SELECT count(*) FROM asset_entries").fetchone()[0]
        for record in records:
            existing = connection.execute(
                f"SELECT {_COLUMNS} FROM asset_entries WHERE asset_id = ?", (record[0],)
            ).fetchone()
            if existing is not None:
                if existing != record:
                    raise ValueError("immutable asset index row mismatch")
                unchanged += 1
                continue
            if total + imported >= MAX_INDEX_ENTRIES:
                raise ValueError("asset index exceeds its entry budget")
            connection.execute("INSERT INTO asset_entries VALUES (?, ?, ?, ?, ?)", record)
            imported += 1
    return IndexResult(imported, unchanged, total + imported)


def query_assets(
    database: Path,
    registry: Path,
    *,
    entity_id: str,
    game_version: str,
    after_asset_id: str | None = None,
    limit: int = 256,
) -> tuple[AssetEntry, ...]:
    """Return a deterministic page for an exact entity/version pair.

    Pagination is by asset ID, with no guessed version fallback. Every returned
    row is reverified against registry metadata and bytes at query time.
    """
    _text(entity_id, "entity_id")
    _text(game_version, "game_version")
    if type(limit) is not int or not 1 <= limit <= MAX_INDEX_BATCH:
        raise ValueError("asset index query limit must be 1..256")
    after = "" if after_asset_id is None else _identifier(after_asset_id)
    with _connection(Path(database)) as connection:
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM asset_entries "
            "WHERE entity_id = ? AND game_version = ? AND asset_id > ? "
            "ORDER BY asset_id LIMIT ?",
            (entity_id, game_version, after, limit),
        ).fetchall()
        return tuple(_decode(row, Path(registry)) for row in rows)


def verify_asset_index(database: Path, registry: Path) -> tuple[AssetEntry, ...]:
    """Verify the dedicated schema and every indexed asset, in stable ID order."""
    with _connection(Path(database)) as connection:
        rows = connection.execute(f"SELECT {_COLUMNS} FROM asset_entries ORDER BY asset_id")
        return tuple(_decode(row, Path(registry)) for row in rows)
