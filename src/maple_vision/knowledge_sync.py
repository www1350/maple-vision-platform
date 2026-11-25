"""Auditable incremental plans for existing, authorized local catalog exports.

A plan is self-contained: source bytes and the baseline logical catalog are
captured in JSON. It contains no executable SQL or filesystem destinations.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .catalog import (
    _COLUMNS,
    CATALOG_SCHEMA_VERSION,
    IMPORT_SCHEMA_VERSION,
    MAX_IMPORT_BYTES,
    STATUSES,
    CatalogRecord,
    ImportResult,
    _decode_record,
    _import_records,
    _parse_record,
    _reject_snapshot_sidecars,
    _validate_schema,
    canonical_json,
    strict_json,
    validate_provenance,
)
from .knowledge import _normalized_sql

PLAN_SCHEMA_VERSION = 1
ADAPTER_ID = "local_catalog_json_v1"
MAX_CATALOG_BYTES = 128 * 1024 * 1024
MAX_PLAN_BYTES = 256 * 1024 * 1024
MAX_CATALOG_RECORDS = 100_000
_RECORD_FIELDS = tuple(CatalogRecord.__dataclass_fields__)
_PLAN_KEYS = {
    "schema_version",
    "kind",
    "adapter",
    "source",
    "baseline",
    "baseline_fingerprint",
    "expected_fingerprint",
    "changes",
    "expected_import",
    "plan_id",
}
_STATE_KEYS = {"catalog_schema_version", "schema_sha256", "records", "legacy_source_records"}
_SOURCE_KEYS = {
    "name",
    "sha256",
    "size_bytes",
    "export_utf8",
    "provenance",
    "game_versions",
    "scopes",
    "record_count",
    "unique_record_count",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object) -> str:
    return _sha(canonical_json(value).encode("utf-8"))


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _source_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or len(value) > 4096
        or any(ord(char) < 32 for char in value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("source name must be a plain file name, never a path")
    return value


def _read_regular(path: Path, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing regular file, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"{label} exceeds its size limit or is not a regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds its size limit")
    return data


def _parse_export(data: bytes) -> tuple[CatalogRecord, ...]:
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError("knowledge export exceeds 16 MiB")
    value = strict_json(data.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "records"}:
        raise ValueError("export needs exactly schema_version and records")
    if type(value["schema_version"]) is not int or value["schema_version"] != IMPORT_SCHEMA_VERSION:
        raise ValueError("unsupported catalog export schema")
    records = value["records"]
    if not isinstance(records, list) or len(records) > 50_000:
        raise ValueError("export records must be a list of at most 50000 entries")
    return tuple(_parse_record(record) for record in records)


@dataclass(frozen=True)
class RemoteReceipt:
    endpoint: str
    request_url_sha256: str
    permission: str
    license: str
    fetched_at: str

    def __post_init__(self) -> None:
        validate_provenance(self.endpoint, self.permission, self.license)
        parts = urlsplit(self.endpoint)
        if parts.scheme != "https" or parts.query or parts.fragment:
            raise ValueError("source receipt needs an HTTPS endpoint without a query or fragment")
        if not _valid_hash(self.request_url_sha256):
            raise ValueError("invalid request URL hash")
        if not isinstance(self.fetched_at, str) or len(self.fetched_at) > 48:
            raise ValueError("invalid retrieval time")
        when = datetime.fromisoformat(self.fetched_at)
        if when.utcoffset() != UTC.utcoffset(None):
            raise ValueError("retrieval time must be UTC")


@dataclass(frozen=True)
class SourceSnapshot:
    """The exact bytes read once, plus their parsed immutable records."""

    name: str
    content: bytes
    records: tuple[CatalogRecord, ...]
    retrieval: RemoteReceipt | None = None


class SourceAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def read(self) -> SourceSnapshot: ...


@dataclass(frozen=True)
class LocalExportSource:
    path: Path

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    def read(self) -> SourceSnapshot:
        path = Path(self.path)
        name = _source_name(path.name)
        content = _read_regular(path, MAX_IMPORT_BYTES, "knowledge export")
        return SourceSnapshot(name, content, _parse_export(content))


@dataclass(frozen=True)
class HttpsExportSource:
    """Fetch one operator-selected, permitted catalog export, only during planning."""

    url: str
    permission: str
    license: str
    expected_sha256: str | None = None

    @property
    def adapter_id(self) -> str:
        return "https_catalog_json_v1"

    def read(self) -> SourceSnapshot:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise ValueError("install the sync extra to fetch HTTPS exports") from exc
        if not isinstance(self.url, str) or len(self.url) > 4096:
            raise ValueError("invalid export URL")
        validate_provenance(self.url, self.permission, self.license)
        parts = urlsplit(self.url)
        if parts.scheme != "https" or parts.fragment or any(ord(c) < 32 for c in self.url):
            raise ValueError("an absolute HTTPS export URL is required")
        if self.expected_sha256 is not None and not _valid_hash(self.expected_sha256):
            raise ValueError("expected export digest must be a lowercase SHA-256")
        endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        name = _source_name(Path(parts.path).name or "export.json")
        data = bytearray()
        started = time.monotonic()
        try:
            with httpx.stream(
                "GET",
                self.url,
                follow_redirects=False,
                timeout=15.0,
                headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"export endpoint returned HTTP {response.status_code}")
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ValueError("export endpoint must serve the original uncompressed JSON")
                size = response.headers.get("content-length")
                if size is not None and (
                    not size.isdecimal() or len(size) > 10 or int(size) > MAX_IMPORT_BYTES
                ):
                    raise ValueError("declared export exceeds the size limit")
                for block in response.iter_bytes():
                    if time.monotonic() - started > 60:
                        raise ValueError("export exceeded its download time budget")
                    if len(data) + len(block) > MAX_IMPORT_BYTES:
                        raise ValueError("export exceeds the size limit")
                    data.extend(block)
                if size is not None and int(size) != len(data):
                    raise ValueError("export length differs from Content-Length")
        except ImportError as exc:
            raise ValueError("install the sync extra for this proxy configuration") from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"HTTPS export failed ({type(exc).__name__})") from None
        content = bytes(data)
        if self.expected_sha256 is not None and _sha(content) != self.expected_sha256:
            raise ValueError("export content differs from pinned SHA-256")
        receipt = RemoteReceipt(
            endpoint,
            _sha(self.url.encode()),
            self.permission,
            self.license,
            datetime.now(UTC).isoformat(),
        )
        return SourceSnapshot(name, content, _parse_export(content), receipt)


@dataclass(frozen=True)
class _LegacyRecord:
    id: str
    source_uri: str
    game_version: str
    payload_hash: str
    verification_status: str


@dataclass(frozen=True)
class _CatalogState:
    schema_sha256: str
    records: tuple[CatalogRecord, ...]
    legacy_source_records: tuple[_LegacyRecord, ...]

    def document(self) -> dict[str, object]:
        return {
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "schema_sha256": self.schema_sha256,
            "records": [asdict(record) for record in self.records],
            "legacy_source_records": [asdict(record) for record in self.legacy_source_records],
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.document())


def _entity(record: CatalogRecord) -> tuple[str, str, str]:
    return record.kind, record.entity_id, record.game_version


def _check_records(records: tuple[CatalogRecord, ...]) -> None:
    ids = [record.record_id for record in records]
    if ids != sorted(set(ids)):
        raise ValueError("baseline records must have unique IDs in canonical order")
    groups: dict[tuple[str, str, str], list[CatalogRecord]] = {}
    for record in records:
        groups.setdefault(_entity(record), []).append(record)
    for group in groups.values():
        if len({record.payload_sha256 for record in group}) > 1 and any(
            record.verification_status != "CONFLICTED" for record in group
        ):
            raise ValueError("baseline conflict flags are inconsistent")


def _legacy_record(value: object) -> _LegacyRecord:
    if not isinstance(value, dict) or set(value) != set(_LegacyRecord.__dataclass_fields__):
        raise ValueError("invalid legacy source record")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError("legacy source fields must be strings")
    if not value["id"].strip() or value["verification_status"] not in STATUSES:
        raise ValueError("invalid legacy record identity or status")
    return _LegacyRecord(**value)


def _state_from_document(value: object) -> _CatalogState:
    if not isinstance(value, dict) or set(value) != _STATE_KEYS:
        raise ValueError("invalid baseline catalog fields")
    if (
        type(value["catalog_schema_version"]) is not int
        or value["catalog_schema_version"] != CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("baseline requires catalog schema v2")
    if not _valid_hash(value["schema_sha256"]):
        raise ValueError("invalid catalog schema fingerprint")
    rows, legacy = value["records"], value["legacy_source_records"]
    if (
        not isinstance(rows, list)
        or not isinstance(legacy, list)
        or len(rows) + len(legacy) > MAX_CATALOG_RECORDS
    ):
        raise ValueError("baseline record count exceeds its limit")
    records: list[CatalogRecord] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_RECORD_FIELDS):
            raise ValueError("invalid baseline catalog record")
        records.append(_decode_record(tuple(row[field] for field in _RECORD_FIELDS)))
    parsed_legacy = tuple(_legacy_record(row) for row in legacy)
    legacy_ids = [record.id for record in parsed_legacy]
    if legacy_ids != sorted(set(legacy_ids)):
        raise ValueError("legacy records must have unique IDs in canonical order")
    result = _CatalogState(value["schema_sha256"], tuple(records), parsed_legacy)
    _check_records(result.records)
    return result


def _catalog_state(connection: sqlite3.Connection) -> _CatalogState:
    _validate_schema(connection)
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    if page_count * page_size > MAX_CATALOG_BYTES:
        raise ValueError("catalog logical size exceeds the sync byte budget")
    if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "delete":
        raise ValueError("knowledge sync requires a standalone DELETE-journal catalog")
    records: list[CatalogRecord] = []
    for row in connection.execute(f"SELECT {_COLUMNS} FROM catalog_records ORDER BY record_id"):
        if len(records) >= MAX_CATALOG_RECORDS:
            raise ValueError("catalog record count exceeds the sync limit")
        records.append(_decode_record(row))
    legacy: list[_LegacyRecord] = []
    fields = tuple(_LegacyRecord.__dataclass_fields__)
    for row in connection.execute(
        "SELECT id, source_uri, game_version, payload_hash, verification_status "
        "FROM source_records ORDER BY id"
    ):
        if len(records) + len(legacy) >= MAX_CATALOG_RECORDS:
            raise ValueError("catalog record count exceeds the sync limit")
        legacy.append(_legacy_record(dict(zip(fields, row, strict=True))))
    schema = [
        [kind, name, table, _normalized_sql(sql) if sql is not None else None]
        for kind, name, table, sql in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name"
        )
    ]
    state = _CatalogState(_digest(schema), tuple(records), tuple(legacy))
    _check_records(state.records)
    return state


def _database_path(database: str | Path) -> Path:
    path = Path(database).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CATALOG_BYTES:
        raise ValueError("sync database must be a regular local file of at most 128 MiB")
    _reject_snapshot_sidecars(path)
    return path


@contextmanager
def _catalog_transaction(database: str | Path, *, write: bool) -> Iterator[sqlite3.Connection]:
    path = _database_path(database)
    identity = (path.stat().st_dev, path.stat().st_ino)
    uri = path.resolve().as_uri() + ("?mode=rw" if write else "?mode=ro")
    with closing(sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)) as connection:
        if not write:
            connection.execute("PRAGMA query_only = ON")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            _reject_snapshot_sidecars(path)
            yield connection
            if path.is_symlink() or (path.stat().st_dev, path.stat().st_ino) != identity:
                raise ValueError("catalog file identity changed during sync")
            connection.commit() if write else connection.rollback()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise


def _source_document(source: SourceSnapshot) -> dict[str, object]:
    provenance = {
        (record.source_uri, record.game_version, record.permission, record.license, record.scope)
        for record in source.records
    }
    fields = ("source_uri", "game_version", "permission", "license", "scope")
    return {
        "name": source.name,
        "sha256": _sha(source.content),
        "size_bytes": len(source.content),
        "export_utf8": source.content.decode("utf-8"),
        "provenance": [dict(zip(fields, entry, strict=True)) for entry in sorted(provenance)],
        "game_versions": sorted({record.game_version for record in source.records}),
        "scopes": sorted({record.scope for record in source.records}),
        "record_count": len(source.records),
        "unique_record_count": len({r.record_id for r in source.records}),
    }


def _expected_state(
    baseline: _CatalogState, incoming: tuple[CatalogRecord, ...]
) -> tuple[_CatalogState, ImportResult]:
    merged = {record.record_id: record for record in baseline.records}
    imported = 0
    affected: set[tuple[str, str, str]] = set()
    for record in incoming:
        stored = merged.get(record.record_id)
        if stored is not None:
            if replace(stored, verification_status=stored.declared_status) != record:
                raise ValueError("record hash collision or mismatched baseline evidence")
        else:
            merged[record.record_id] = record
            imported += 1
        affected.add(_entity(record))
    if len(merged) + len(baseline.legacy_source_records) > MAX_CATALOG_RECORDS:
        raise ValueError("expected catalog exceeds the sync record limit")
    payloads: dict[tuple[str, str, str], set[str]] = {}
    for record in merged.values():
        payloads.setdefault(_entity(record), set()).add(record.payload_sha256)
    conflicts = {key for key in affected if len(payloads[key]) > 1}
    records = tuple(
        replace(record, verification_status="CONFLICTED")
        if _entity(record) in conflicts
        else record
        for record in sorted(merged.values(), key=lambda record: record.record_id)
    )
    expected = _CatalogState(baseline.schema_sha256, records, baseline.legacy_source_records)
    _check_records(expected.records)
    return expected, ImportResult(imported, len(incoming) - imported, len(conflicts), len(records))


def _changes(
    baseline: _CatalogState, source: SourceSnapshot, expected: _CatalogState
) -> dict[str, object]:
    before = {record.record_id: record for record in baseline.records}
    incoming = {record.record_id: record for record in source.records}
    baseline_entities = {_entity(record) for record in baseline.records}
    added, unchanged, changed = [], [], []
    for identifier, record in sorted(incoming.items()):
        if identifier in before:
            unchanged.append(identifier)
        elif _entity(record) in baseline_entities:
            changed.append(identifier)
        else:
            added.append(identifier)
    status_changes: list[dict[str, object]] = []
    for record in expected.records:
        old = before.get(record.record_id, incoming.get(record.record_id))
        assert old is not None
        if old.verification_status != record.verification_status:
            status_changes.append(
                {
                    "record_id": record.record_id,
                    "before": old.verification_status,
                    "after": record.verification_status,
                    "existing": record.record_id in before,
                }
            )
    return {
        "added": added,
        "unchanged": unchanged,
        "changed": changed,
        "missing": sorted(set(before) - set(incoming)),
        "status_changes": status_changes,
    }


def _plan_document(
    source: SourceSnapshot, baseline: _CatalogState
) -> tuple[dict[str, object], _CatalogState]:
    expected, result = _expected_state(baseline, source.records)
    document: dict[str, object] = {
        "schema_version": 2 if source.retrieval is not None else PLAN_SCHEMA_VERSION,
        "kind": "knowledge_sync_plan",
        "adapter": "https_catalog_json_v1" if source.retrieval is not None else ADAPTER_ID,
        **({"retrieval": asdict(source.retrieval)} if source.retrieval is not None else {}),
        "source": _source_document(source),
        "baseline": baseline.document(),
        "baseline_fingerprint": baseline.fingerprint,
        "expected_fingerprint": expected.fingerprint,
        "changes": _changes(baseline, source, expected),
        "expected_import": asdict(result),
    }
    document["plan_id"] = _digest(document)
    return document, expected


def _publish_plan(path: Path, data: bytes) -> None:
    if len(data) > MAX_PLAN_BYTES:
        raise ValueError("sync plan exceeds 256 MiB")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".knowledge-plan-", dir=path.parent)
    stage = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _plan_summary(document: dict[str, Any]) -> dict[str, object]:
    return {
        "schema_version": document["schema_version"],
        "plan_id": document["plan_id"],
        "adapter": document["adapter"],
        "source_sha256": document["source"]["sha256"],
        "baseline_fingerprint": document["baseline_fingerprint"],
        "expected_fingerprint": document["expected_fingerprint"],
        "delta_counts": {key: len(value) for key, value in document["changes"].items()},
        "expected_import": document["expected_import"],
        "missing_policy": "retain_history",
    }


def plan_sync(
    database: str | Path, source_json: str | Path, output_plan: str | Path
) -> dict[str, object]:
    """Capture a stable local export and a consistent full logical catalog baseline."""
    return plan_from_source(database, LocalExportSource(Path(source_json)), output_plan)


def plan_from_source(
    database: str | Path, adapter: SourceAdapter, output_plan: str | Path
) -> dict[str, object]:
    path = Path(output_plan).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"plan already exists: {path}")
    source = adapter.read()
    with _catalog_transaction(database, write=False) as connection:
        baseline = _catalog_state(connection)
        document, _ = _plan_document(source, baseline)
    _publish_plan(path, (canonical_json(document) + "\n").encode("utf-8"))
    return {**_plan_summary(document), "status": "planned", "plan": str(path)}


def _read_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], SourceSnapshot, _CatalogState, _CatalogState]:
    content = _read_regular(plan_path, MAX_PLAN_BYTES, "sync plan")
    document = strict_json(content.decode("utf-8"))
    if not isinstance(document, dict) or set(document) != (
        _PLAN_KEYS | {"retrieval"} if document.get("schema_version") == 2 else _PLAN_KEYS
    ):
        raise ValueError("sync plan has unknown or missing fields")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] not in (PLAN_SCHEMA_VERSION, 2)
        or document["kind"] != "knowledge_sync_plan"
        or document["adapter"]
        != ("https_catalog_json_v1" if document["schema_version"] == 2 else ADAPTER_ID)
    ):
        raise ValueError("unsupported sync plan format or source adapter")
    body = {key: value for key, value in document.items() if key != "plan_id"}
    if document["plan_id"] != _digest(body):
        raise ValueError("sync plan integrity hash mismatch")
    metadata = document["source"]
    if not isinstance(metadata, dict) or set(metadata) != _SOURCE_KEYS:
        raise ValueError("invalid captured source fields; embedded paths are not accepted")
    name = _source_name(metadata["name"])
    if not isinstance(metadata["export_utf8"], str):
        raise ValueError("captured export must be UTF-8 text")
    data = metadata["export_utf8"].encode("utf-8")
    if (
        metadata["sha256"] != _sha(data)
        or type(metadata["size_bytes"]) is not int
        or metadata["size_bytes"] != len(data)
    ):
        raise ValueError("captured source bytes/hash/size disagree")
    retrieval = None
    if document["schema_version"] == 2:
        item = document["retrieval"]
        if not isinstance(item, dict) or set(item) != set(RemoteReceipt.__dataclass_fields__):
            raise ValueError("invalid captured retrieval receipt")
        retrieval = RemoteReceipt(**item)
    source = SourceSnapshot(name, data, _parse_export(data), retrieval)
    baseline = _state_from_document(document["baseline"])
    rebuilt, expected = _plan_document(source, baseline)
    # Recompute all deltas, provenance, fingerprints and counts; a recomputed outer
    # checksum cannot make a forged delta or after-state trustworthy.
    if canonical_json(document) != canonical_json(rebuilt):
        raise ValueError("sync plan metadata or transition does not match its captured evidence")
    return document, source, baseline, expected


def apply_sync(database: str | Path, plan_path: str | Path) -> dict[str, object]:
    """Check and apply in one write transaction, retaining all historical evidence."""
    document, source, baseline, expected = _read_plan(Path(plan_path))
    with _catalog_transaction(database, write=True) as connection:
        current = _catalog_state(connection)
        if current.fingerprint == expected.fingerprint:
            return {
                **_plan_summary(document),
                "status": "no_changes"
                if baseline.fingerprint == expected.fingerprint
                else "already_applied",
                "idempotent": True,
                "imported": 0,
                "records_total": len(current.records),
                "catalog_fingerprint": current.fingerprint,
            }
        if current.fingerprint != baseline.fingerprint:
            raise ValueError(
                "stale sync plan: current catalog is neither its baseline nor expected after-state"
            )
        result = _import_records(connection, source.records)
        actual = _catalog_state(connection)
        if actual.fingerprint != expected.fingerprint or canonical_json(
            asdict(result)
        ) != canonical_json(document["expected_import"]):
            raise ValueError(
                "sync postcondition failed; the complete transaction will be rolled back"
            )
    return {
        **_plan_summary(document),
        "status": "applied",
        "idempotent": False,
        "imported": result.imported,
        "records_total": result.records_total,
        "catalog_fingerprint": actual.fingerprint,
    }
