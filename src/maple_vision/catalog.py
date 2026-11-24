"""Versioned, provenance-preserving offline knowledge catalog.

The P0 ``knowledge`` module remains a v1 compatibility API. This module explicitly
upgrades its schema to v2 and supplies the import/query/snapshot workflow. No
operation contacts a source website or changes a record's asserted provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .knowledge import _normalized_sql, migrate

CATALOG_SCHEMA_VERSION = 2
IMPORT_SCHEMA_VERSION = 1
MAX_IMPORT_BYTES = 16 * 1024 * 1024
KINDS = frozenset({"classes", "skills", "monsters", "maps", "items", "npcs", "quests", "mechanics"})
STATUSES = frozenset({"VERIFIED", "PROVISIONAL", "UNKNOWN", "CONFLICTED"})
PERMISSIONS = frozenset({"self_authored", "public_license", "explicit_permission"})
SCOPES = frozenset({"synthetic_fixture", "game_reference"})
_HEX = re.compile(r"^[0-9a-f]{64}$")

_SOURCE_V2 = """CREATE TABLE source_records (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    source_uri TEXT NOT NULL,
    game_version TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    verification_status TEXT NOT NULL CHECK (
        verification_status IN ('VERIFIED', 'PROVISIONAL', 'UNKNOWN', 'CONFLICTED')
    )
)"""
_RECORDS_SQL = """CREATE TABLE catalog_records (
    record_id TEXT PRIMARY KEY NOT NULL CHECK (length(record_id) = 64),
    kind TEXT NOT NULL CHECK (kind IN
        ('classes','skills','monsters','maps','items','npcs','quests','mechanics')),
    entity_id TEXT NOT NULL CHECK (length(trim(entity_id)) > 0),
    game_version TEXT NOT NULL CHECK (length(trim(game_version)) > 0),
    source_uri TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (permission IN
        ('self_authored','public_license','explicit_permission')),
    license TEXT NOT NULL CHECK (length(trim(license)) > 0),
    scope TEXT NOT NULL CHECK (scope IN ('synthetic_fixture','game_reference')),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    payload_sha256 TEXT NOT NULL CHECK (length(payload_sha256) = 64),
    declared_status TEXT NOT NULL CHECK (declared_status IN
        ('VERIFIED','PROVISIONAL','UNKNOWN','CONFLICTED')),
    verification_status TEXT NOT NULL CHECK (verification_status IN
        ('VERIFIED','PROVISIONAL','UNKNOWN','CONFLICTED'))
)"""
_INDEX_SQL = "CREATE INDEX catalog_entity ON catalog_records(kind, entity_id, game_version)"
_COLUMNS = (
    "record_id, kind, entity_id, game_version, source_uri, collected_at, permission, license, "
    "scope, payload_json, payload_sha256, declared_status, verification_status"
)
_IMPORT_KEYS = frozenset(
    {
        "kind",
        "entity_id",
        "game_version",
        "source_uri",
        "collected_at",
        "permission",
        "license",
        "verification_status",
        "scope",
        "payload",
    }
)


def canonical_json(value: object) -> str:
    """Canonical JSON for this schema (UTF-8, sorted keys, no non-finite numbers)."""
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("value is not finite JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=nonfinite)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{field} must be a nonempty string of at most 4096 characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains a control character")
    return value


def validate_provenance(source_uri: str, permission: str, license: str) -> None:
    _text(source_uri, "source_uri")
    parsed = urlparse(source_uri)
    if parsed.scheme not in {"https", "http", "urn", "file"}:
        raise ValueError("source_uri must be an absolute http(s), urn, or file URI")
    if parsed.scheme in {"http", "https"} and (
        not parsed.hostname
        or any(char.isspace() for char in parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("source_uri needs a host and must not include credentials")
    if parsed.scheme in {"urn", "file"} and not parsed.path:
        raise ValueError("source_uri must identify a resource")
    _text(permission, "permission")
    if permission not in PERMISSIONS:
        raise ValueError(f"permission must be one of {sorted(PERMISSIONS)}")
    _text(license, "license")


@dataclass(frozen=True)
class CatalogRecord:
    record_id: str
    kind: str
    entity_id: str
    game_version: str
    source_uri: str
    collected_at: str
    permission: str
    license: str
    scope: str
    payload_json: str
    payload_sha256: str
    declared_status: str
    verification_status: str

    @property
    def payload(self) -> dict[str, Any]:
        value = strict_json(self.payload_json)
        if not isinstance(value, dict):
            raise ValueError("catalog payload is not an object")
        return value


@dataclass(frozen=True)
class ImportResult:
    imported: int
    unchanged: int
    conflicted_entities: int
    records_total: int


def _parse_record(value: object) -> CatalogRecord:
    if not isinstance(value, dict) or set(value) != _IMPORT_KEYS:
        raise ValueError(f"record must have exactly these fields: {sorted(_IMPORT_KEYS)}")
    record: dict[str, Any] = value
    for field in _IMPORT_KEYS - {"payload"}:
        _text(record[field], field)
    if record["kind"] not in KINDS:
        raise ValueError(f"unsupported knowledge kind: {record['kind']}")
    if record["verification_status"] not in STATUSES:
        raise ValueError("invalid verification_status")
    if record["scope"] not in SCOPES:
        raise ValueError("scope must identify synthetic_fixture or game_reference")
    validate_provenance(record["source_uri"], record["permission"], record["license"])
    try:
        collected = datetime.fromisoformat(record["collected_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("collected_at must be an ISO 8601 timestamp") from exc
    if collected.tzinfo is None or collected.utcoffset() is None:
        raise ValueError("collected_at must include a time zone")
    if not isinstance(record["payload"], dict):
        raise ValueError("payload must be a JSON object")
    payload_json = canonical_json(record["payload"])
    record_id = _sha(canonical_json(record).encode("utf-8"))
    return CatalogRecord(
        record_id,
        record["kind"],
        record["entity_id"],
        record["game_version"],
        record["source_uri"],
        record["collected_at"],
        record["permission"],
        record["license"],
        record["scope"],
        payload_json,
        _sha(payload_json.encode("utf-8")),
        record["verification_status"],
        record["verification_status"],
    )


def _decode_record(row: tuple[Any, ...]) -> CatalogRecord:
    record = CatalogRecord(*row)
    parsed = _parse_record(
        {
            "kind": record.kind,
            "entity_id": record.entity_id,
            "game_version": record.game_version,
            "source_uri": record.source_uri,
            "collected_at": record.collected_at,
            "permission": record.permission,
            "license": record.license,
            "scope": record.scope,
            "verification_status": record.declared_status,
            "payload": record.payload,
        }
    )
    if (
        parsed.record_id != record.record_id
        or parsed.payload_sha256 != record.payload_sha256
        or parsed.payload_json != record.payload_json
    ):
        raise ValueError("catalog record content or hash has been modified")
    if record.verification_status not in STATUSES:
        raise ValueError("invalid effective verification status")
    if record.verification_status not in {record.declared_status, "CONFLICTED"}:
        raise ValueError("effective status cannot upgrade declared status")
    return record


@contextmanager
def _savepoint(connection: sqlite3.Connection) -> Iterator[None]:
    name = "catalog_" + uuid4().hex
    connection.execute(f"SAVEPOINT {name}")
    try:
        yield
        connection.execute(f"RELEASE SAVEPOINT {name}")
    except BaseException:
        connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
        connection.execute(f"RELEASE SAVEPOINT {name}")
        raise


def _validate_schema(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA user_version").fetchone()[0] != CATALOG_SCHEMA_VERSION:
        raise ValueError("catalog requires schema v2; run initialize_catalog explicitly")
    for name, kind, sql in (
        ("source_records", "table", _SOURCE_V2),
        ("catalog_records", "table", _RECORDS_SQL),
        ("catalog_entity", "index", _INDEX_SQL),
    ):
        row = connection.execute(
            "SELECT type, sql FROM sqlite_schema WHERE name = ?", (name,)
        ).fetchone()
        if row is None or row[0] != kind or _normalized_sql(row[1]) != _normalized_sql(sql):
            raise ValueError(f"incompatible catalog schema: {name}")


def _reject_legacy_dependencies(connection: sqlite3.Connection) -> None:
    """A table rebuild must not silently retarget or delete extension schema."""
    objects = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema WHERE sql IS NOT NULL"
    ).fetchall()
    for kind, name, table, sql in objects:
        if kind == "table" and name.lower() != "source_records":
            quoted = '"' + name.replace('"', '""') + '"'
            for foreign_key in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
                if foreign_key[2].lower() == "source_records":
                    raise ValueError("v1 migration refuses dependent foreign-key schema")
        attached = kind in {"trigger", "index"} and table.lower() == "source_records"
        references = kind in {"view", "trigger"} and re.search(
            r"\bsource_records\b", sql, flags=re.IGNORECASE
        )
        if attached or references:
            raise ValueError("v1 migration refuses dependent extension schema")


def migrate_catalog(connection: sqlite3.Connection) -> None:
    """Explicit v0/v1→v2 migration, preserving caller-owned transactions."""
    with _savepoint(connection):
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == CATALOG_SCHEMA_VERSION:
            _validate_schema(connection)
            return
        if version not in (0, 1):
            raise ValueError(f"unsupported catalog schema version: {version}")
        migrate(connection)
        _reject_legacy_dependencies(connection)
        invalid = connection.execute(
            "SELECT COUNT(*) FROM source_records WHERE id IS NULL OR length(trim(id)) = 0"
        ).fetchone()[0]
        if invalid:
            raise ValueError("v1 source records contain null/empty IDs; migration refused")
        connection.execute("ALTER TABLE source_records RENAME TO source_records_v1")
        connection.execute(_SOURCE_V2)
        connection.execute("INSERT INTO source_records SELECT * FROM source_records_v1")
        connection.execute("DROP TABLE source_records_v1")
        connection.execute(_RECORDS_SQL)
        connection.execute(_INDEX_SQL)
        connection.execute(f"PRAGMA user_version = {CATALOG_SCHEMA_VERSION}")
        _validate_schema(connection)


def initialize_catalog(path: str | Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        migrate_catalog(connection)


def open_catalog_readonly(path: str | Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        _validate_schema(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def load_import(source: str | Path) -> tuple[CatalogRecord, ...]:
    path = Path(source)
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError("knowledge import exceeds 16 MiB")
    value = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "records"}:
        raise ValueError("import needs exactly schema_version and records")
    if type(value["schema_version"]) is not int or value["schema_version"] != IMPORT_SCHEMA_VERSION:
        raise ValueError("unsupported knowledge import schema")
    records = value["records"]
    if not isinstance(records, list) or len(records) > 50_000:
        raise ValueError("records must be a list of at most 50000 entries")
    return tuple(_parse_record(record) for record in records)


def import_catalog(path: str | Path, source: str | Path) -> ImportResult:
    """Validate an entire import before one transaction; never overwrite evidence."""
    records = load_import(source)
    # Require initialization: a typo must not create an empty database during import.
    uri = Path(path).resolve().as_uri() + "?mode=rw"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        _validate_schema(connection)
        imported = 0
        affected: set[tuple[str, str, str]] = set()
        with _savepoint(connection):
            for record in records:
                existing = connection.execute(
                    f"SELECT {_COLUMNS} FROM catalog_records WHERE record_id = ?",
                    (record.record_id,),
                ).fetchone()
                if existing is not None:
                    stored = _decode_record(existing)
                    if replace(stored, verification_status=stored.declared_status) != record:
                        raise ValueError("record hash collision or corrupt existing record")
                else:
                    connection.execute(
                        f"INSERT INTO catalog_records ({_COLUMNS}) VALUES ({','.join(['?'] * 13)})",
                        tuple(asdict(record).values()),
                    )
                    imported += 1
                affected.add((record.kind, record.entity_id, record.game_version))
            conflicts = 0
            for key in sorted(affected):
                hashes = connection.execute(
                    "SELECT COUNT(DISTINCT payload_sha256) FROM catalog_records "
                    "WHERE kind = ? AND entity_id = ? AND game_version = ?",
                    key,
                ).fetchone()[0]
                if hashes > 1:
                    conflicts += 1
                    connection.execute(
                        "UPDATE catalog_records SET verification_status = 'CONFLICTED' "
                        "WHERE kind = ? AND entity_id = ? AND game_version = ?",
                        key,
                    )
            total = connection.execute("SELECT COUNT(*) FROM catalog_records").fetchone()[0]
        return ImportResult(imported, len(records) - imported, conflicts, total)


def query_catalog(
    path: str | Path,
    kind: str,
    entity_id: str,
    game_version: str,
    *,
    allow_unverified: bool = False,
) -> tuple[CatalogRecord, ...]:
    """Default queries expose only verified, non-conflicting records.

    ``allow_unverified`` is for inspection; callers must still respect each
    record's status and scope. It does not promote unknown data to knowledge.
    """
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"unsupported knowledge kind: {kind}")
    if type(allow_unverified) is not bool:
        raise ValueError("allow_unverified must be an explicit boolean")
    _text(entity_id, "entity_id")
    _text(game_version, "game_version")
    with closing(open_catalog_readonly(path)) as connection:
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM catalog_records "
            "WHERE kind = ? AND entity_id = ? AND game_version = ? ORDER BY record_id",
            (kind, entity_id, game_version),
        ).fetchall()
        records = tuple(_decode_record(row) for row in rows)
        if len({record.payload_sha256 for record in records}) > 1 and any(
            record.verification_status != "CONFLICTED" for record in records
        ):
            raise ValueError("catalog conflict flags are inconsistent")
        return (
            records
            if allow_unverified
            else tuple(record for record in records if record.verification_status == "VERIFIED")
        )


def _summary(connection: sqlite3.Connection) -> dict[str, object]:
    records = tuple(
        _decode_record(row)
        for row in connection.execute(f"SELECT {_COLUMNS} FROM catalog_records ORDER BY record_id")
    )
    keys: dict[tuple[str, str, str], list[CatalogRecord]] = {}
    for record in records:
        keys.setdefault((record.kind, record.entity_id, record.game_version), []).append(record)
    for values in keys.values():
        if len({record.payload_sha256 for record in values}) > 1 and any(
            record.verification_status != "CONFLICTED" for record in values
        ):
            raise ValueError("catalog conflict flags are inconsistent")
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "records": len(records),
        "entities": len(keys),
        "legacy_source_records": connection.execute(
            "SELECT COUNT(*) FROM source_records"
        ).fetchone()[0],
        "status_counts": {
            status: sum(record.verification_status == status for record in records)
            for status in sorted(STATUSES)
        },
        "scopes": sorted({record.scope for record in records}),
        "game_versions": sorted({record.game_version for record in records}),
    }


def catalog_summary(path: str | Path) -> dict[str, object]:
    with closing(open_catalog_readonly(path)) as connection:
        return _summary(connection)


def create_snapshot(path: str | Path, destination_dir: str | Path) -> dict[str, object]:
    """Back up a consistent SQLite snapshot and a hash manifest to a new directory."""
    destination = Path(destination_dir)
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=".catalog-", dir=destination.parent))
    try:
        snapshot = staging / "knowledge.sqlite"
        with closing(open_catalog_readonly(path)) as source:
            with closing(sqlite3.connect(snapshot)) as target:
                source.backup(target)
                # The backup inherits the source's journal mode. Publish a standalone
                # database: unmanifested WAL pages must never affect a verified snapshot.
                mode = target.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
                if mode.lower() != "delete":
                    raise ValueError("could not create a standalone snapshot database")
                _validate_schema(target)
                summary = _summary(target)
        _reject_snapshot_sidecars(snapshot)
        manifest: dict[str, object] = {
            "manifest_version": 1,
            "database": "knowledge.sqlite",
            "sha256": _sha(snapshot.read_bytes()),
            "summary": summary,
        }
        (staging / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        # A directory reservation makes publication non-overwriting on all supported OSes.
        destination.mkdir()
        try:
            # Publish the manifest last: its presence is the completion marker.
            for name in ("knowledge.sqlite", "manifest.json"):
                os.replace(staging / name, destination / name)
        except BaseException:
            shutil.rmtree(destination)
            raise
        return manifest
    finally:
        shutil.rmtree(staging)


def _reject_snapshot_sidecars(snapshot: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = snapshot.with_name(snapshot.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("snapshot has unmanifested SQLite journal sidecars")


def verify_snapshot(destination_dir: str | Path) -> dict[str, object]:
    destination = Path(destination_dir)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 64 * 1024:
        raise ValueError("invalid snapshot manifest file")
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"manifest_version", "database", "sha256", "summary"}
        or type(manifest["manifest_version"]) is not int
        or manifest["manifest_version"] != 1
        or manifest["database"] != "knowledge.sqlite"
        or not isinstance(manifest["sha256"], str)
        or not _HEX.fullmatch(manifest["sha256"])
    ):
        raise ValueError("invalid snapshot manifest")
    snapshot = destination / "knowledge.sqlite"
    if snapshot.is_symlink():
        raise ValueError("snapshot database must not be a symlink")
    _reject_snapshot_sidecars(snapshot)
    database_bytes = snapshot.read_bytes()
    if _sha(database_bytes) != manifest["sha256"]:
        raise ValueError("snapshot SHA256 mismatch")
    # Inspect exactly the bytes that passed the digest. Reopening the pathname
    # could observe concurrent writes or journal pages which were never hashed.
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.deserialize(database_bytes)
        connection.execute("PRAGMA query_only = ON")
        _validate_schema(connection)
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("snapshot SQLite integrity check failed")
        if canonical_json(_summary(connection)) != canonical_json(manifest["summary"]):
            raise ValueError("snapshot summary mismatch")
    _reject_snapshot_sidecars(snapshot)
    return dict(manifest)
