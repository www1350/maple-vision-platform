"""Typed, independently reproducible projections of immutable v2 catalog snapshots.

Only explicitly versioned and complete domain payloads with VERIFIED evidence
are projected. A source declaration is not a certificate of permission or game
accuracy. This module neither synchronizes websites nor loads runtime rules.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from .catalog import (
    _COLUMNS,
    KINDS,
    SCOPES,
    STATUSES,
    CatalogRecord,
    _decode_record,
    _summary,
    _validate_schema,
    canonical_json,
    strict_json,
    validate_provenance,
)

DOMAIN_SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_DOMAIN_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_SOURCE_RECORDS = 100_000
MAX_DOMAIN_ROWS = 250_000
MAX_COLLECTION = 1000
MAX_QUERY_ROWS = 1000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMON = "scope TEXT NOT NULL, game_version TEXT NOT NULL, entity_id TEXT NOT NULL"
_KEY = "scope,game_version,entity_id"
_SOURCE_COLUMNS = tuple(_COLUMNS.split(", "))
_REQUIRED: dict[str, set[str]] = {
    "classes": {"name", "tier", "advancements"},
    "skills": {"name", "class_id", "max_level", "levels"},
    "monsters": {"name", "level", "max_hp", "max_mp", "experience", "drops"},
    "maps": {"name", "width", "height", "connections", "platforms", "portals"},
    "items": {"name", "category", "stack_limit", "equipment", "consumable"},
    "npcs": {"name", "map_id", "x", "y"},
    "quests": {"name", "npc_id", "min_level", "requirements"},
    "mechanics": {"name", "rule_type", "unit", "numeric_value", "text_value"},
}


def _fk(columns: str, target: str, target_columns: str = _KEY) -> str:
    return (
        f"FOREIGN KEY({columns}) REFERENCES {target}({target_columns}) "
        "DEFERRABLE INITIALLY DEFERRED"
    )


def _domain_table(columns: str, *, extra_key: str = "", references: tuple[str, ...] = ()) -> str:
    primary = _KEY + ("," + extra_key if extra_key else "")
    return ", ".join((_COMMON, columns, f"PRIMARY KEY({primary})", *references))


# These are deliberately separate typed relations, not renamed payload JSON.
_TABLES: dict[str, str] = {
    "game_versions": (
        "scope TEXT NOT NULL, game_version TEXT NOT NULL, PRIMARY KEY(scope,game_version)"
    ),
    "knowledge_snapshots": (
        "snapshot_id TEXT PRIMARY KEY NOT NULL, source_manifest_sha256 TEXT NOT NULL, "
        "source_database_sha256 TEXT NOT NULL, source_schema INTEGER NOT NULL"
    ),
    "source_records": (
        "record_id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL, entity_id TEXT NOT NULL, "
        "game_version TEXT NOT NULL, source_uri TEXT NOT NULL, collected_at TEXT NOT NULL, "
        "permission TEXT NOT NULL, license TEXT NOT NULL, scope TEXT NOT NULL, "
        "payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, declared_status TEXT NOT NULL, "
        "verification_status TEXT NOT NULL, confidence REAL"
    ),
    "entity_sources": _domain_table(
        "kind TEXT NOT NULL, record_id TEXT NOT NULL",
        extra_key="kind,record_id",
        references=(_fk("record_id", "source_records", "record_id"),),
    ),
    "projection_issues": _domain_table(
        "kind TEXT NOT NULL, record_id TEXT NOT NULL, reason TEXT NOT NULL, "
        "details_json TEXT NOT NULL",
        extra_key="kind,record_id",
        references=(_fk("record_id", "source_records", "record_id"),),
    ),
    "classes": _domain_table("name TEXT NOT NULL, tier INTEGER NOT NULL"),
    "advancements": _domain_table(
        "to_class_id TEXT NOT NULL, required_level INTEGER NOT NULL",
        extra_key="to_class_id",
        references=(
            _fk(_KEY, "classes"),
            _fk("scope,game_version,to_class_id", "classes"),
        ),
    ),
    "skills": _domain_table(
        "name TEXT NOT NULL, class_id TEXT NOT NULL, max_level INTEGER NOT NULL",
        references=(_fk("scope,game_version,class_id", "classes"),),
    ),
    "skill_levels": _domain_table(
        "level INTEGER NOT NULL, hp_cost INTEGER, mp_cost INTEGER, ammo_item_id TEXT, "
        "ammo_cost INTEGER, cooldown_ms INTEGER, cast_ms INTEGER",
        extra_key="level",
        references=(_fk(_KEY, "skills"), _fk("scope,game_version,ammo_item_id", "items")),
    ),
    "skill_effects": _domain_table(
        "level INTEGER NOT NULL, effect_id TEXT NOT NULL, effect_kind TEXT NOT NULL, "
        "target TEXT NOT NULL, magnitude REAL, duration_ms INTEGER, stat TEXT",
        extra_key="level,effect_id",
        references=(_fk(_KEY + ",level", "skill_levels", _KEY + ",level"),),
    ),
    "monsters": _domain_table(
        "name TEXT NOT NULL, level INTEGER NOT NULL, max_hp INTEGER, max_mp INTEGER, "
        "experience INTEGER"
    ),
    "monster_drops": _domain_table(
        "item_id TEXT NOT NULL, probability REAL, quantity_min INTEGER NOT NULL, "
        "quantity_max INTEGER NOT NULL",
        extra_key="item_id",
        references=(_fk(_KEY, "monsters"), _fk("scope,game_version,item_id", "items")),
    ),
    "maps": _domain_table("name TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL"),
    "map_connections": _domain_table(
        "connection_id TEXT NOT NULL, destination_map_id TEXT NOT NULL, travel_kind TEXT NOT NULL",
        extra_key="connection_id",
        references=(_fk(_KEY, "maps"), _fk("scope,game_version,destination_map_id", "maps")),
    ),
    "map_platforms": _domain_table(
        "platform_id TEXT NOT NULL, x1 REAL NOT NULL, x2 REAL NOT NULL, y REAL NOT NULL",
        extra_key="platform_id",
        references=(_fk(_KEY, "maps"),),
    ),
    "map_portals": _domain_table(
        "portal_id TEXT NOT NULL, destination_map_id TEXT NOT NULL, destination_portal_id TEXT, "
        "x REAL NOT NULL, y REAL NOT NULL",
        extra_key="portal_id",
        references=(
            _fk(_KEY, "maps"),
            _fk("scope,game_version,destination_map_id", "maps"),
            _fk(
                "scope,game_version,destination_map_id,destination_portal_id",
                "map_portals",
                _KEY + ",portal_id",
            ),
        ),
    ),
    "items": _domain_table(
        "name TEXT NOT NULL, category TEXT NOT NULL, stack_limit INTEGER NOT NULL"
    ),
    "equipment": _domain_table(
        "slot TEXT NOT NULL, required_level INTEGER NOT NULL, required_class_id TEXT, "
        "attack INTEGER, defense INTEGER",
        references=(_fk(_KEY, "items"), _fk("scope,game_version,required_class_id", "classes")),
    ),
    "consumables": _domain_table(
        "hp_restore INTEGER, mp_restore INTEGER, duration_ms INTEGER",
        references=(_fk(_KEY, "items"),),
    ),
    "npcs": _domain_table(
        "name TEXT NOT NULL, map_id TEXT NOT NULL, x REAL NOT NULL, y REAL NOT NULL",
        references=(_fk("scope,game_version,map_id", "maps"),),
    ),
    "quests": _domain_table(
        "name TEXT NOT NULL, npc_id TEXT, min_level INTEGER NOT NULL",
        references=(_fk("scope,game_version,npc_id", "npcs"),),
    ),
    "quest_requirements": _domain_table(
        "requirement_id TEXT NOT NULL, requirement_kind TEXT NOT NULL, item_id TEXT, "
        "monster_id TEXT, prerequisite_quest_id TEXT, class_id TEXT, quantity INTEGER NOT NULL",
        extra_key="requirement_id",
        references=(
            _fk(_KEY, "quests"),
            _fk("scope,game_version,item_id", "items"),
            _fk("scope,game_version,monster_id", "monsters"),
            _fk("scope,game_version,prerequisite_quest_id", "quests"),
            _fk("scope,game_version,class_id", "classes"),
        ),
    ),
    "mechanics_rules": _domain_table(
        "name TEXT NOT NULL, rule_type TEXT NOT NULL, unit TEXT NOT NULL, "
        "numeric_value REAL, text_value TEXT"
    ),
    "asset_index": _domain_table(
        "kind TEXT NOT NULL, asset_id TEXT NOT NULL, role TEXT NOT NULL",
        extra_key="kind,asset_id,role",
    ),
    "validation_evidence": _domain_table(
        "kind TEXT NOT NULL, evidence_id TEXT NOT NULL, method TEXT NOT NULL, "
        "artifact_sha256 TEXT NOT NULL, source_uri TEXT NOT NULL, declared_status TEXT NOT NULL",
        extra_key="kind,evidence_id",
    ),
}
DOMAIN_TABLES = frozenset(_TABLES) - {"game_versions", "knowledge_snapshots"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value: object, name: str, *, limit: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} requires nonempty text within {limit} characters")
    return value


def _integer(value: object, name: str, low: int = 0, high: int = 1_000_000_000) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} requires an integer in [{low},{high}]")
    return value


def _number(value: object, name: str, low: float = -1e12, high: float = 1e12) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} requires a finite number")
    try:
        number = float(value) if isinstance(value, (int, float)) else math.nan
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} requires a finite number") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} requires a finite number in [{low},{high}]")
    return number


def _optional_integer(value: object, name: str, low: int = 0) -> int | None:
    return None if value is None else _integer(value, name, low)


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _choice(value: object, name: str, choices: Iterable[str]) -> str:
    text = _text(value, name)
    if text not in choices:
        raise ValueError(f"unsupported {name}: {text}")
    return text


class _MissingDomainFields(ValueError):
    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        super().__init__(f"missing required fields: {fields}")


def _object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"{name} requires exactly {sorted(fields)}")
    if fields - set(value):
        raise _MissingDomainFields([f"{name}.{field}" for field in sorted(fields - set(value))])
    return value


def _entries(value: object, fields: set[str], name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_COLLECTION:
        raise ValueError(f"{name} requires a bounded array")
    return [_object(item, fields, name) for item in value]


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular nonsymlink file: {path.name}")
    if path.stat().st_size > limit:
        raise ValueError(f"file exceeds byte budget: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds byte budget: {path.name}")
    return data


def _sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("unmanifested SQLite journal sidecar")


def _capture_source(source: Path) -> tuple[bytes, bytes]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source snapshot must be a nonsymlink directory")
    database = source / "knowledge.sqlite"
    _sidecars(database)
    manifest = _read(source / "manifest.json", 64 * 1024)
    content = _read(database, MAX_SOURCE_BYTES)
    _sidecars(database)
    return manifest, content


def _validated_source(manifest_bytes: bytes, database_bytes: bytes) -> tuple[CatalogRecord, ...]:
    """Reuse v2 catalog semantics on exactly the bounded, hashed source bytes."""
    manifest = _object(
        strict_json(manifest_bytes.decode("utf-8")),
        {"manifest_version", "database", "sha256", "summary"},
        "source manifest",
    )
    if (
        type(manifest["manifest_version"]) is not int
        or manifest["manifest_version"] != 1
        or manifest["database"] != "knowledge.sqlite"
        or manifest["sha256"] != _sha(database_bytes)
    ):
        raise ValueError("source snapshot version or SHA256 mismatch")
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.deserialize(database_bytes)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        _validate_schema(connection)
        count = connection.execute("SELECT count(*) FROM catalog_records").fetchone()[0]
        legacy = connection.execute("SELECT count(*) FROM source_records").fetchone()[0]
        if count + legacy > MAX_SOURCE_RECORDS:
            raise ValueError("source snapshot exceeds record budget")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("source snapshot SQLite integrity failure")
        if canonical_json(_summary(connection)) != canonical_json(manifest["summary"]):
            raise ValueError("source snapshot summary mismatch")
        return tuple(
            _decode_record(row)
            for row in connection.execute(
                f"SELECT {_COLUMNS} FROM catalog_records ORDER BY record_id"
            )
        )


class _Projection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.rows = 0

    def add(self, table: str, row: dict[str, object]) -> None:
        self.rows += 1
        if self.rows > MAX_DOMAIN_ROWS:
            raise ValueError("domain projection exceeds row budget")
        columns = ",".join(row)
        placeholders = ",".join("?" for _ in row)
        self.connection.execute(
            f"INSERT INTO {table}({columns}) VALUES ({placeholders})", tuple(row.values())
        )

    def domain(self, table: str, key: dict[str, object], **fields: object) -> None:
        self.add(table, {**key, **fields})


def _root(projection: _Projection, record: CatalogRecord) -> None:
    payload = record.payload
    key: dict[str, object] = {
        "scope": record.scope,
        "game_version": record.game_version,
        "entity_id": _text(record.entity_id, "entity_id"),
    }
    name = _text(payload["name"], "name", limit=512)
    kind = record.kind
    if kind == "classes":
        projection.domain("classes", key, name=name, tier=_integer(payload["tier"], "tier", 0, 100))
        for row in _entries(
            payload["advancements"], {"to_class_id", "required_level"}, "advancements"
        ):
            projection.domain(
                "advancements",
                key,
                to_class_id=_text(row["to_class_id"], "to_class_id"),
                required_level=_integer(row["required_level"], "required_level", 0, 1000),
            )
    elif kind == "skills":
        maximum = _integer(payload["max_level"], "max_level", 1, 200)
        projection.domain(
            "skills",
            key,
            name=name,
            class_id=_text(payload["class_id"], "class_id"),
            max_level=maximum,
        )
        levels = _entries(
            payload["levels"],
            {
                "level",
                "hp_cost",
                "mp_cost",
                "ammo_item_id",
                "ammo_cost",
                "cooldown_ms",
                "cast_ms",
                "effects",
            },
            "skill levels",
        )
        # A missing level is missing knowledge, not zero cost or the preceding level's value.
        level_numbers = [_integer(row["level"], "skill level", 1, maximum) for row in levels]
        if len(set(level_numbers)) != len(level_numbers):
            raise ValueError("skill levels must not repeat a level")
        missing_levels = sorted(set(range(1, maximum + 1)) - set(level_numbers))
        if missing_levels:
            raise _MissingDomainFields([f"skill_levels.{level}" for level in missing_levels])
        for row in sorted(levels, key=lambda item: item["level"]):
            ammo_id = _optional_text(row["ammo_item_id"], "ammo_item_id")
            ammo_cost = _optional_integer(row["ammo_cost"], "ammo_cost")
            if ammo_cost is not None and ammo_cost > 0 and ammo_id is None:
                raise ValueError("positive ammo_cost requires ammo_item_id")
            projection.domain(
                "skill_levels",
                key,
                level=row["level"],
                hp_cost=_optional_integer(row["hp_cost"], "hp_cost"),
                mp_cost=_optional_integer(row["mp_cost"], "mp_cost"),
                ammo_item_id=ammo_id,
                ammo_cost=ammo_cost,
                cooldown_ms=_optional_integer(row["cooldown_ms"], "cooldown_ms"),
                cast_ms=_optional_integer(row["cast_ms"], "cast_ms"),
            )
            for effect in _entries(
                row["effects"],
                {"effect_id", "kind", "target", "magnitude", "duration_ms", "stat"},
                "skill effects",
            ):
                projection.domain(
                    "skill_effects",
                    key,
                    level=row["level"],
                    effect_id=_text(effect["effect_id"], "effect_id"),
                    effect_kind=_choice(
                        effect["kind"],
                        "effect kind",
                        {
                            "damage",
                            "heal",
                            "buff",
                            "debuff",
                            "summon",
                            "movement",
                            "resource",
                            "damage_over_time",
                        },
                    ),
                    target=_choice(
                        effect["target"], "effect target", {"self", "enemy", "ally", "area"}
                    ),
                    magnitude=None
                    if effect["magnitude"] is None
                    else _number(effect["magnitude"], "magnitude"),
                    duration_ms=_optional_integer(effect["duration_ms"], "duration_ms"),
                    stat=_optional_text(effect["stat"], "stat"),
                )
    elif kind == "monsters":
        projection.domain(
            "monsters",
            key,
            name=name,
            level=_integer(payload["level"], "level", 0, 1000),
            max_hp=_optional_integer(payload["max_hp"], "max_hp", 1),
            max_mp=_optional_integer(payload["max_mp"], "max_mp"),
            experience=_optional_integer(payload["experience"], "experience"),
        )
        for row in _entries(
            payload["drops"], {"item_id", "probability", "quantity_min", "quantity_max"}, "drops"
        ):
            low = _integer(row["quantity_min"], "quantity_min", 1)
            high = _integer(row["quantity_max"], "quantity_max", low)
            projection.domain(
                "monster_drops",
                key,
                item_id=_text(row["item_id"], "drop item_id"),
                probability=None
                if row["probability"] is None
                else _number(row["probability"], "probability", 0, 1),
                quantity_min=low,
                quantity_max=high,
            )
    elif kind == "maps":
        _map(projection, key, name, payload)
    elif kind == "items":
        _item(projection, key, name, payload)
    elif kind == "npcs":
        projection.domain(
            "npcs",
            key,
            name=name,
            map_id=_text(payload["map_id"], "map_id"),
            x=_number(payload["x"], "NPC x", 0, 1_000_000),
            y=_number(payload["y"], "NPC y", 0, 1_000_000),
        )
    elif kind == "quests":
        projection.domain(
            "quests",
            key,
            name=name,
            npc_id=_optional_text(payload["npc_id"], "npc_id"),
            min_level=_integer(payload["min_level"], "min_level", 0, 1000),
        )
        for row in _entries(
            payload["requirements"],
            {"requirement_id", "kind", "target_id", "quantity"},
            "requirements",
        ):
            requirement = _choice(
                row["kind"], "requirement kind", {"item", "monster", "quest", "class", "level"}
            )
            target = _optional_text(row["target_id"], "target_id")
            if (requirement == "level") != (target is None):
                raise ValueError("only level requirements must have null target_id")
            projection.domain(
                "quest_requirements",
                key,
                requirement_id=_text(row["requirement_id"], "requirement_id"),
                requirement_kind=requirement,
                item_id=target if requirement == "item" else None,
                monster_id=target if requirement == "monster" else None,
                prerequisite_quest_id=target if requirement == "quest" else None,
                class_id=target if requirement == "class" else None,
                quantity=_integer(row["quantity"], "requirement quantity", 1),
            )
    else:
        numeric = payload["numeric_value"]
        textual = _optional_text(payload["text_value"], "text_value")
        if numeric is not None and textual is not None:
            raise ValueError("mechanics may declare either a numeric or a text value")
        projection.domain(
            "mechanics_rules",
            key,
            name=name,
            rule_type=_text(payload["rule_type"], "rule_type"),
            unit=_text(payload["unit"], "unit"),
            numeric_value=None if numeric is None else _number(numeric, "numeric_value"),
            text_value=textual,
        )
    for asset in _entries(payload.get("assets", []), {"asset_id", "role"}, "assets"):
        identifier = _text(asset["asset_id"], "asset_id")
        if not _HASH.fullmatch(identifier):
            raise ValueError("asset_id must be a lowercase SHA256")
        projection.domain(
            "asset_index",
            key,
            kind=kind,
            asset_id=identifier,
            role=_text(asset["role"], "asset role"),
        )
    for evidence in _entries(
        payload.get("evidence", []),
        {"evidence_id", "method", "artifact_sha256", "source_uri", "declared_status"},
        "validation evidence",
    ):
        evidence_hash = _text(evidence["artifact_sha256"], "artifact_sha256")
        if not _HASH.fullmatch(evidence_hash):
            raise ValueError("evidence artifact_sha256 must be a lowercase SHA256")
        uri = _text(evidence["source_uri"], "evidence source_uri", limit=4096)
        validate_provenance(uri, record.permission, record.license)
        projection.domain(
            "validation_evidence",
            key,
            kind=kind,
            evidence_id=_text(evidence["evidence_id"], "evidence_id"),
            method=_text(evidence["method"], "evidence method"),
            artifact_sha256=evidence_hash,
            source_uri=uri,
            declared_status=_choice(evidence["declared_status"], "evidence status", STATUSES),
        )


def _map(
    projection: _Projection, key: dict[str, object], name: str, payload: dict[str, Any]
) -> None:
    width = _integer(payload["width"], "map width", 1, 1_000_000)
    height = _integer(payload["height"], "map height", 1, 1_000_000)
    projection.domain("maps", key, name=name, width=width, height=height)
    for row in _entries(
        payload["connections"],
        {"connection_id", "destination_map_id", "travel_kind"},
        "connections",
    ):
        projection.domain(
            "map_connections",
            key,
            connection_id=_text(row["connection_id"], "connection_id"),
            destination_map_id=_text(row["destination_map_id"], "destination_map_id"),
            travel_kind=_choice(row["travel_kind"], "travel_kind", {"portal", "walk", "other"}),
        )
    for row in _entries(payload["platforms"], {"platform_id", "x1", "x2", "y"}, "platforms"):
        left = _number(row["x1"], "platform x1", 0, width)
        right = _number(row["x2"], "platform x2", 0, width)
        if left >= right:
            raise ValueError("platform x1 must be less than x2")
        projection.domain(
            "map_platforms",
            key,
            platform_id=_text(row["platform_id"], "platform_id"),
            x1=left,
            x2=right,
            y=_number(row["y"], "platform y", 0, height),
        )
    for row in _entries(
        payload["portals"],
        {"portal_id", "destination_map_id", "destination_portal_id", "x", "y"},
        "portals",
    ):
        projection.domain(
            "map_portals",
            key,
            portal_id=_text(row["portal_id"], "portal_id"),
            destination_map_id=_text(row["destination_map_id"], "destination_map_id"),
            destination_portal_id=_optional_text(
                row["destination_portal_id"], "destination_portal_id"
            ),
            x=_number(row["x"], "portal x", 0, width),
            y=_number(row["y"], "portal y", 0, height),
        )


def _item(
    projection: _Projection, key: dict[str, object], name: str, payload: dict[str, Any]
) -> None:
    category = _choice(
        payload["category"],
        "item category",
        {"equipment", "consumable", "ammunition", "material", "currency", "quest", "other"},
    )
    equipment, consumable = payload["equipment"], payload["consumable"]
    if (category == "equipment") != (equipment is not None):
        raise ValueError("equipment details must agree with item category")
    if (category == "consumable") != (consumable is not None):
        raise ValueError("consumable details must agree with item category")
    projection.domain(
        "items",
        key,
        name=name,
        category=category,
        stack_limit=_integer(payload["stack_limit"], "stack_limit", 1, 1_000_000),
    )
    if equipment is not None:
        row = _object(
            equipment,
            {"slot", "required_level", "required_class_id", "attack", "defense"},
            "equipment",
        )
        projection.domain(
            "equipment",
            key,
            slot=_text(row["slot"], "slot"),
            required_level=_integer(row["required_level"], "required_level", 0, 1000),
            required_class_id=_optional_text(row["required_class_id"], "required_class_id"),
            attack=_optional_integer(row["attack"], "attack"),
            defense=_optional_integer(row["defense"], "defense"),
        )
    if consumable is not None:
        row = _object(consumable, {"hp_restore", "mp_restore", "duration_ms"}, "consumable")
        projection.domain(
            "consumables",
            key,
            hp_restore=_optional_integer(row["hp_restore"], "hp_restore"),
            mp_restore=_optional_integer(row["mp_restore"], "mp_restore"),
            duration_ms=_optional_integer(row["duration_ms"], "duration_ms"),
        )


def _reason(record: CatalogRecord) -> tuple[str, list[str]] | None:
    if record.verification_status != "VERIFIED":
        return "unverified_or_conflicted", [record.verification_status]
    payload = record.payload
    if "domain_schema_version" not in payload:
        return "unmapped_payload", ["domain_schema_version"]
    version = payload["domain_schema_version"]
    if type(version) is not int:
        raise ValueError("domain_schema_version must be an integer, not bool or float")
    if version != DOMAIN_SCHEMA_VERSION:
        return "unsupported_domain_schema", [str(version)]
    fields = _REQUIRED[record.kind] | {"domain_schema_version"}
    unknown = set(payload) - fields - {"assets", "evidence"}
    if unknown:
        raise ValueError(f"unsupported domain payload fields: {sorted(unknown)}")
    missing = fields - set(payload)
    if missing:
        return "missing_fields", sorted(missing)
    return None


def _acyclic(rows: Iterable[tuple[str, str, str, str]], name: str) -> None:
    graph: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    incoming: Counter[tuple[str, str, str]] = Counter()
    for scope, version, start, end in rows:
        before, after = (scope, version, start), (scope, version, end)
        if after not in graph[before]:
            graph[before].add(after)
            incoming[after] += 1
        incoming.setdefault(before, 0)
    queue = [node for node, degree in incoming.items() if degree == 0]
    removed = 0
    while queue:
        node = queue.pop()
        removed += 1
        for after in graph.get(node, ()):
            incoming[after] -= 1
            if incoming[after] == 0:
                queue.append(after)
    if removed != len(incoming):
        raise ValueError(f"cyclic {name}")


def _relationships(connection: sqlite3.Connection) -> None:
    broken = connection.execute("PRAGMA foreign_key_check").fetchone()
    if broken:
        raise ValueError(f"unresolved same-scope same-version reference in {broken[0]}")
    _acyclic(
        connection.execute("SELECT scope,game_version,entity_id,to_class_id FROM advancements"),
        "advancements",
    )
    _acyclic(
        connection.execute(
            "SELECT scope,game_version,entity_id,prerequisite_quest_id FROM quest_requirements "
            "WHERE prerequisite_quest_id IS NOT NULL"
        ),
        "quest prerequisites",
    )
    outside = connection.execute(
        "SELECT n.entity_id FROM npcs n JOIN maps m ON n.scope=m.scope AND "
        "n.game_version=m.game_version AND n.map_id=m.entity_id "
        "WHERE n.x>m.width OR n.y>m.height LIMIT 1"
    ).fetchone()
    if outside:
        raise ValueError("NPC position outside its referenced map")


def _project(
    records: tuple[CatalogRecord, ...], manifest_hash: str, database_hash: str
) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute(f"PRAGMA user_version={DOMAIN_SCHEMA_VERSION}")
        for name, columns in _TABLES.items():
            connection.execute(f"CREATE TABLE {name} ({columns}) STRICT")
        connection.execute("BEGIN")
        projection = _Projection(connection)
        projection.add(
            "knowledge_snapshots",
            {
                "snapshot_id": _sha((manifest_hash + database_hash).encode("ascii")),
                "source_manifest_sha256": manifest_hash,
                "source_database_sha256": database_hash,
                "source_schema": 2,
            },
        )
        versions = sorted({(record.scope, record.game_version) for record in records})
        for scope, version in versions:
            projection.add("game_versions", {"scope": scope, "game_version": version})
        mapped: set[tuple[str, str, str, str]] = set()
        for record in records:
            projection.add(
                "source_records", {column: getattr(record, column) for column in _SOURCE_COLUMNS}
            )
            issue = _reason(record)
            key: dict[str, object] = {
                "scope": record.scope,
                "game_version": record.game_version,
                "entity_id": record.entity_id,
            }
            if issue is not None:
                projection.domain(
                    "projection_issues",
                    key,
                    kind=record.kind,
                    record_id=record.record_id,
                    reason=issue[0],
                    details_json=canonical_json(issue[1]),
                )
                continue
            entity = (record.scope, record.game_version, record.kind, record.entity_id)
            if entity not in mapped:
                row_count = projection.rows
                connection.execute("SAVEPOINT domain_entity")
                try:
                    _root(projection, record)
                except _MissingDomainFields as exc:
                    connection.execute("ROLLBACK TO domain_entity")
                    connection.execute("RELEASE domain_entity")
                    projection.rows = row_count
                    projection.domain(
                        "projection_issues",
                        key,
                        kind=record.kind,
                        record_id=record.record_id,
                        reason="missing_fields",
                        details_json=canonical_json(exc.fields),
                    )
                    continue
                connection.execute("RELEASE domain_entity")
                mapped.add(entity)
            projection.domain("entity_sources", key, kind=record.kind, record_id=record.record_id)
        _relationships(connection)
        connection.commit()
        return connection
    except sqlite3.DatabaseError as exc:
        connection.close()
        raise ValueError(f"domain relational validation failed: {exc}") from exc
    except BaseException:
        connection.close()
        raise


def _table_rows(connection: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    columns = len(connection.execute(f"PRAGMA table_info({table})").fetchall())
    ordering = ",".join(str(index) for index in range(1, columns + 1))
    return connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall()


def _schema(connection: sqlite3.Connection) -> list[tuple[Any, ...]]:
    return connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema "
        "WHERE name NOT GLOB 'sqlite_*' ORDER BY type,name"
    ).fetchall()


def _report(connection: sqlite3.Connection) -> dict[str, object]:
    counts = {
        name: connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        for name in sorted(_TABLES)
    }
    source_entities = connection.execute(
        "SELECT count(*) FROM (SELECT DISTINCT scope,game_version,kind,entity_id "
        "FROM source_records)"
    ).fetchone()[0]
    mapped_entities = sum(
        counts["mechanics_rules" if kind == "mechanics" else kind] for kind in KINDS
    )
    return {
        "source_records": counts["source_records"],
        "source_record_denominator": "v2_catalog_records; legacy_rows_archived_only",
        "mapped_source_records": counts["entity_sources"],
        "unmapped_source_records": counts["projection_issues"],
        "source_entity_versions": source_entities,
        "projected_entities": mapped_entities,
        "unmapped_entity_versions": source_entities - mapped_entities,
        "all_source_records_mapped": (
            counts["source_records"] > 0 and counts["projection_issues"] == 0
        ),
        "mapping_status": "empty_source"
        if counts["source_records"] == 0
        else ("partial" if counts["projection_issues"] else "complete_for_declared_source"),
        "row_counts": counts,
        "issue_counts": dict(
            connection.execute(
                "SELECT reason,count(*) FROM projection_issues GROUP BY reason ORDER BY reason"
            )
        ),
        "scope_versions": [list(row) for row in _table_rows(connection, "game_versions")],
        "verification_scope": "source_declarations_and_structural_projection_only",
        "real_game_accuracy": "not_established",
        "asset_bytes_verified": False,
        "evidence_artifacts_verified": False,
        "source_confidence_available": False,
        "runtime_rules_compiled": False,
    }


def _manifest(files: dict[str, str], summary: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "kind": "domain-catalog-snapshot",
        "files": files,
        "summary": summary,
    }
    return {**body, "snapshot_id": _sha(canonical_json(body).encode("utf-8"))}


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_domain_snapshot(
    source_snapshot: str | Path, destination: str | Path
) -> dict[str, object]:
    """Publish a new self-contained projection without modifying the source catalog."""
    source, output = Path(source_snapshot), Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("domain output must not be inside the source snapshot")
    source_manifest, source_database = _capture_source(source)
    records = _validated_source(source_manifest, source_database)
    with closing(_project(records, _sha(source_manifest), _sha(source_database))) as connection:
        data = connection.serialize()
        if len(data) > MAX_DOMAIN_BYTES:
            raise ValueError("domain database exceeds byte budget")
        summary = _report(connection)
    stage = Path(tempfile.mkdtemp(prefix=".domain-build-", dir=output.parent))
    try:
        (stage / "source").mkdir()
        artifacts = {
            "domain.sqlite": data,
            "source/manifest.json": source_manifest,
            "source/knowledge.sqlite": source_database,
        }
        for name, content in artifacts.items():
            _write(stage / name, content)
        manifest = _manifest({name: _sha(content) for name, content in artifacts.items()}, summary)
        _write(stage / "manifest.json", (canonical_json(manifest) + "\n").encode("utf-8"))
        verify_domain_snapshot(stage)
        output.mkdir()
        try:
            for name in ("source", "domain.sqlite", "manifest.json"):
                os.replace(stage / name, output / name)
        except BaseException:
            shutil.rmtree(output)
            raise
        return manifest
    finally:
        shutil.rmtree(stage)


def _verified_connection(path: Path) -> tuple[dict[str, object], sqlite3.Connection]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("domain snapshot must be a nonsymlink directory")
    if {item.name for item in path.iterdir()} != {"manifest.json", "domain.sqlite", "source"}:
        raise ValueError("domain snapshot contains missing or unexpected files")
    if (path / "source").is_symlink() or {item.name for item in (path / "source").iterdir()} != {
        "manifest.json",
        "knowledge.sqlite",
    }:
        raise ValueError("invalid archived source directory")
    _sidecars(path / "domain.sqlite")
    manifest = _object(
        strict_json(_read(path / "manifest.json", MAX_MANIFEST_BYTES).decode("utf-8")),
        {"schema_version", "kind", "files", "summary", "snapshot_id"},
        "domain manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != DOMAIN_SCHEMA_VERSION
        or manifest["kind"] != "domain-catalog-snapshot"
    ):
        raise ValueError("unsupported domain snapshot schema or kind")
    files = _object(
        manifest["files"],
        {"domain.sqlite", "source/manifest.json", "source/knowledge.sqlite"},
        "domain files",
    )
    source_manifest, source_database = _capture_source(path / "source")
    database = _read(path / "domain.sqlite", MAX_DOMAIN_BYTES)
    actual_hashes = {
        "domain.sqlite": _sha(database),
        "source/manifest.json": _sha(source_manifest),
        "source/knowledge.sqlite": _sha(source_database),
    }
    if files != actual_hashes:
        raise ValueError("domain snapshot file SHA256 mismatch")
    records = _validated_source(source_manifest, source_database)
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(database)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA user_version").fetchone()[0] != DOMAIN_SCHEMA_VERSION:
            raise ValueError("domain database schema version mismatch")
        with closing(_project(records, _sha(source_manifest), _sha(source_database))) as expected:
            if _schema(connection) != _schema(expected):
                raise ValueError("domain database schema differs from independently rebuilt source")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("domain database SQLite integrity failure")
            for table in _TABLES:
                count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                if count > MAX_DOMAIN_ROWS or _table_rows(connection, table) != _table_rows(
                    expected, table
                ):
                    raise ValueError(f"domain semantic reprojection mismatch: {table}")
            expected_manifest = _manifest(actual_hashes, _report(expected))
            if canonical_json(manifest) != canonical_json(expected_manifest):
                raise ValueError("domain manifest summary or identity mismatch")
        _sidecars(path / "domain.sqlite")
        connection.set_authorizer(_read_only_authorizer)
        return dict(manifest), connection
    except BaseException:
        connection.close()
        raise


def _read_only_authorizer(
    action: int, first: str | None, second: str | None, database: str | None, context: str | None
) -> int:
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and second not in {"load_extension", "writefile"}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and first in {
        "table_info",
        "foreign_key_list",
        "foreign_key_check",
        "quick_check",
        "user_version",
        "query_only",
    }:
        if first in {"user_version", "query_only"} and second is not None:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def verify_domain_snapshot(path: str | Path) -> dict[str, object]:
    """Verify file integrity AND rebuild every relation from archived source evidence."""
    try:
        manifest, connection = _verified_connection(Path(path))
    except (sqlite3.DatabaseError, UnicodeError) as exc:
        raise ValueError(f"invalid domain snapshot: {exc}") from exc
    connection.close()
    return manifest


def open_domain_readonly(path: str | Path) -> sqlite3.Connection:
    """Return a verified in-memory byte snapshot with writes and ATTACH denied.

    The caller must close this connection. No file is reopened after validation;
    later mutation of the snapshot pathname cannot change returned query results.
    This is a database API boundary, not a process-permission sandbox.
    """
    try:
        _, connection = _verified_connection(Path(path))
    except (sqlite3.DatabaseError, UnicodeError) as exc:
        raise ValueError(f"invalid domain snapshot: {exc}") from exc
    return connection


def query_domain(
    path: str | Path,
    table: str,
    game_version: str,
    *,
    scope: str,
    entity_id: str | None = None,
    limit: int = 1000,
) -> tuple[dict[str, object], ...]:
    """Query a fixed table allowlist, with an explicit version and evidence scope."""
    if not isinstance(table, str) or table not in DOMAIN_TABLES:
        raise ValueError("unsupported domain query table")
    _text(game_version, "game_version", limit=4096)
    _choice(scope, "scope", SCOPES)
    if entity_id is not None:
        _text(entity_id, "entity_id", limit=4096)
    _integer(limit, "query limit", 1, MAX_QUERY_ROWS)
    with closing(open_domain_readonly(path)) as connection:
        connection.row_factory = sqlite3.Row
        columns = len(connection.execute(f"PRAGMA table_info({table})").fetchall())
        ordering = ",".join(str(index) for index in range(1, columns + 1))
        sql = f"SELECT * FROM {table} WHERE game_version=? AND scope=?"
        values: list[object] = [game_version, scope]
        if entity_id is not None:
            sql += " AND entity_id=?"
            values.append(entity_id)
        values.append(limit)
        return tuple(
            dict(row) for row in connection.execute(sql + f" ORDER BY {ordering} LIMIT ?", values)
        )
