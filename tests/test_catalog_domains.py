"""Independent, self-authored domain-export fixtures; no actual game data."""

import copy
import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from maple_vision import catalog_domains
from maple_vision.catalog import canonical_json, create_snapshot, import_catalog, initialize_catalog
from maple_vision.catalog_domains import (
    build_domain_snapshot,
    open_domain_readonly,
    query_domain,
    verify_domain_snapshot,
)


def evidence(kind, identifier, payload, **overrides):
    record = {
        "kind": kind,
        "entity_id": identifier,
        "game_version": "fixture-v1",
        "source_uri": "urn:self-authored:domain-catalog-tests",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0; synthetic test data, not official game rules",
        "verification_status": "VERIFIED",
        "scope": "synthetic_fixture",
        "payload": {"domain_schema_version": 1, **payload},
    }
    record.update(overrides)
    return record


def authored_export():
    """One explicit fixture covers every normalized business relation."""
    return [
        evidence(
            "classes",
            "beginner",
            {
                "name": "Test beginner",
                "tier": 0,
                "advancements": [{"to_class_id": "thrower", "required_level": 10}],
            },
        ),
        evidence("classes", "thrower", {"name": "Test thrower", "tier": 1, "advancements": []}),
        evidence(
            "skills",
            "throw",
            {
                "name": "Test throw",
                "class_id": "thrower",
                "max_level": 2,
                "levels": [
                    {
                        "level": level,
                        "hp_cost": 0,
                        "mp_cost": level,
                        "ammo_item_id": "star",
                        "ammo_cost": 1,
                        "cooldown_ms": 200,
                        "cast_ms": None,
                        "effects": [
                            {
                                "effect_id": "hit",
                                "kind": "damage",
                                "target": "enemy",
                                "magnitude": 10 * level,
                                "duration_ms": 0,
                                "stat": "hp",
                            }
                        ],
                    }
                    for level in (1, 2)
                ],
                "assets": [{"asset_id": "a" * 64, "role": "icon"}],
                "evidence": [
                    {
                        "evidence_id": "test-design",
                        "method": "authored_fixture",
                        "artifact_sha256": "b" * 64,
                        "source_uri": "urn:fixture:design",
                        "declared_status": "PROVISIONAL",
                    }
                ],
            },
        ),
        evidence(
            "items",
            "star",
            {
                "name": "Test star",
                "category": "ammunition",
                "stack_limit": 100,
                "equipment": None,
                "consumable": None,
            },
        ),
        evidence(
            "items",
            "potion",
            {
                "name": "Test potion",
                "category": "consumable",
                "stack_limit": 20,
                "equipment": None,
                "consumable": {"hp_restore": 30, "mp_restore": None, "duration_ms": 0},
            },
        ),
        evidence(
            "items",
            "dagger",
            {
                "name": "Test dagger",
                "category": "equipment",
                "stack_limit": 1,
                "equipment": {
                    "slot": "weapon",
                    "required_level": 10,
                    "required_class_id": "thrower",
                    "attack": 6,
                    "defense": 0,
                },
                "consumable": None,
            },
        ),
        evidence(
            "items",
            "coin",
            {
                "name": "Test coin",
                "category": "currency",
                "stack_limit": 999,
                "equipment": None,
                "consumable": None,
            },
        ),
        evidence(
            "monsters",
            "blob",
            {
                "name": "Test blob",
                "level": 3,
                "max_hp": 20,
                "max_mp": None,
                "experience": 7,
                "drops": [
                    {"item_id": "coin", "probability": 0.25, "quantity_min": 1, "quantity_max": 3}
                ],
            },
        ),
        evidence(
            "maps",
            "field",
            {
                "name": "Test field",
                "width": 640,
                "height": 320,
                "connections": [
                    {
                        "connection_id": "to-cave",
                        "destination_map_id": "cave",
                        "travel_kind": "portal",
                    }
                ],
                "platforms": [{"platform_id": "floor", "x1": 0, "x2": 640, "y": 300}],
                "portals": [
                    {
                        "portal_id": "exit",
                        "destination_map_id": "cave",
                        "destination_portal_id": "back",
                        "x": 600,
                        "y": 290,
                    }
                ],
            },
        ),
        evidence(
            "maps",
            "cave",
            {
                "name": "Test cave",
                "width": 320,
                "height": 192,
                "connections": [
                    {
                        "connection_id": "to-field",
                        "destination_map_id": "field",
                        "travel_kind": "portal",
                    }
                ],
                "platforms": [{"platform_id": "floor", "x1": 0, "x2": 320, "y": 168}],
                "portals": [
                    {
                        "portal_id": "back",
                        "destination_map_id": "field",
                        "destination_portal_id": "exit",
                        "x": 10,
                        "y": 160,
                    }
                ],
            },
        ),
        evidence("npcs", "guide", {"name": "Test guide", "map_id": "field", "x": 40, "y": 280}),
        evidence(
            "quests",
            "intro",
            {"name": "Test intro", "npc_id": "guide", "min_level": 1, "requirements": []},
        ),
        evidence(
            "quests",
            "collect",
            {
                "name": "Test collection",
                "npc_id": "guide",
                "min_level": 10,
                "requirements": [
                    {
                        "requirement_id": "previous",
                        "kind": "quest",
                        "target_id": "intro",
                        "quantity": 1,
                    },
                    {"requirement_id": "coins", "kind": "item", "target_id": "coin", "quantity": 3},
                    {
                        "requirement_id": "defeat",
                        "kind": "monster",
                        "target_id": "blob",
                        "quantity": 2,
                    },
                    {
                        "requirement_id": "job",
                        "kind": "class",
                        "target_id": "thrower",
                        "quantity": 1,
                    },
                    {"requirement_id": "level", "kind": "level", "target_id": None, "quantity": 10},
                ],
            },
        ),
        evidence(
            "mechanics",
            "gravity",
            {
                "name": "Test gravity",
                "rule_type": "physics",
                "unit": "px/s^2",
                "numeric_value": 650,
                "text_value": None,
            },
        ),
    ]


def selected(records, kind, identifier):
    return next(row for row in records if (row["kind"], row["entity_id"]) == (kind, identifier))


def source_snapshot(tmp_path, records=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "source.sqlite"
    initialize_catalog(database)
    source = tmp_path / "export.json"
    source.write_text(
        json.dumps(
            {"schema_version": 1, "records": authored_export() if records is None else records}
        ),
        encoding="utf-8",
    )
    import_catalog(database, source)
    snapshot = tmp_path / "source-snapshot"
    create_snapshot(database, snapshot)
    return snapshot


@pytest.fixture
def bundle(tmp_path):
    source = source_snapshot(tmp_path)
    target = tmp_path / "domains"
    manifest = build_domain_snapshot(source, target)
    return source, target, manifest


def query(
    bundle_path, table, identifier=None, version="fixture-v1", scope="synthetic_fixture", **kwargs
):
    return query_domain(bundle_path, table, version, scope=scope, entity_id=identifier, **kwargs)


def rehash_bundle(target):
    """Simulate an attacker who can repair EVERY outer integrity hash."""
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["files"] = {
        name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in manifest["files"]
    }
    manifest.pop("snapshot_id")
    manifest["snapshot_id"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    (target / "manifest.json").write_text(canonical_json(manifest) + "\n")


def test_full_authored_export_has_real_typed_relations_and_exact_denominators(bundle):
    _, target, manifest = bundle
    assert verify_domain_snapshot(target) == manifest
    summary = manifest["summary"]
    assert summary["source_records"] == summary["mapped_source_records"] == 14
    assert summary["projected_entities"] == 14
    assert summary["unmapped_source_records"] == 0
    assert summary["all_source_records_mapped"] is True
    for table in (
        "advancements",
        "skill_levels",
        "skill_effects",
        "monster_drops",
        "map_connections",
        "map_platforms",
        "map_portals",
        "equipment",
        "consumables",
        "npcs",
        "quests",
        "quest_requirements",
        "asset_index",
        "mechanics_rules",
        "validation_evidence",
    ):
        assert summary["row_counts"][table] > 0
    assert query(target, "advancements", "beginner")[0]["to_class_id"] == "thrower"
    assert [(row["level"], row["mp_cost"]) for row in query(target, "skill_levels", "throw")] == [
        (1, 1),
        (2, 2),
    ]
    assert query(target, "monster_drops", "blob")[0]["probability"] == 0.25
    assert query(target, "map_portals", "field")[0]["destination_portal_id"] == "back"
    assert query(target, "equipment", "dagger")[0]["required_class_id"] == "thrower"
    assert query(target, "quests", "collect")[0]["npc_id"] == "guide"
    assert len(query(target, "quest_requirements", "collect")) == 5
    assert query(target, "mechanics_rules", "gravity")[0]["numeric_value"] == 650.0
    with closing(open_domain_readonly(target)) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        columns = {row[1]: row[2] for row in connection.execute("PRAGMA table_info(skill_levels)")}
        assert columns["mp_cost"] == "INTEGER"
        assert "payload_json" not in columns
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name='sync_jobs'"
            ).fetchone()[0]
            == 0
        )


def test_unknown_values_and_evidence_declarations_are_not_promoted(bundle):
    _, target, manifest = bundle
    assert query(target, "monsters", "blob")[0]["max_mp"] is None
    assert query(target, "consumables", "potion")[0]["mp_restore"] is None
    assert query(target, "skill_levels", "throw")[0]["cast_ms"] is None
    assert query(target, "source_records", "throw")[0]["confidence"] is None
    assert query(target, "validation_evidence")[0]["declared_status"] == "PROVISIONAL"
    for field in (
        "asset_bytes_verified",
        "evidence_artifacts_verified",
        "source_confidence_available",
        "runtime_rules_compiled",
    ):
        assert manifest["summary"][field] is False
    assert manifest["summary"]["real_game_accuracy"] == "not_established"


def test_empty_source_does_not_claim_a_complete_game_database(tmp_path):
    output = tmp_path / "domains"
    report = build_domain_snapshot(source_snapshot(tmp_path, []), output)["summary"]
    assert report["source_records"] == report["projected_entities"] == 0
    assert report["mapping_status"] == "empty_source"
    assert report["all_source_records_mapped"] is False
    assert query(output, "classes") == ()


def test_snapshot_portable_and_deterministic_without_mutating_source(bundle, tmp_path):
    source, target, manifest = bundle
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    second = tmp_path / "second-domains"
    assert build_domain_snapshot(source, second) == manifest
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    shutil.rmtree(source)
    assert verify_domain_snapshot(second) == manifest
    assert query(second, "classes") == query(target, "classes")


def test_duplicate_provenance_kept_but_domain_entity_is_not_duplicated(tmp_path):
    records = authored_export()
    extra = copy.deepcopy(selected(records, "skills", "throw"))
    extra["source_uri"] = "urn:second-source:authored-test"
    records.append(extra)
    output = tmp_path / "domains"
    report = build_domain_snapshot(source_snapshot(tmp_path, records), output)["summary"]
    assert report["source_records"] == report["mapped_source_records"] == 15
    assert report["projected_entities"] == 14
    assert len(query(output, "entity_sources", "throw")) == 2
    assert len(query(output, "skills", "throw")) == 1
    assert len(query(output, "skill_levels", "throw")) == 2


def test_legacy_missing_unknown_and_conflicted_records_report_exact_unmapped_denominator(tmp_path):
    records = [
        evidence("classes", "legacy", {"label": "No domain schema"}),
        evidence("classes", "missing", {"name": "Partial", "tier": 0}),
        evidence("classes", "future", {"name": "Future", "tier": 0, "advancements": []}),
        evidence(
            "classes",
            "unknown",
            {"name": "Unknown", "tier": 0, "advancements": []},
            verification_status="UNKNOWN",
        ),
        evidence("classes", "conflict", {"name": "First", "tier": 0, "advancements": []}),
        evidence(
            "classes",
            "conflict",
            {"name": "Second", "tier": 0, "advancements": []},
            source_uri="urn:second",
        ),
    ]
    records[0]["payload"].pop("domain_schema_version")
    records[2]["payload"]["domain_schema_version"] = 99
    output = tmp_path / "domains"
    manifest = build_domain_snapshot(source_snapshot(tmp_path, records), output)
    summary = manifest["summary"]
    assert summary["source_records"] == summary["unmapped_source_records"] == 6
    assert summary["mapped_source_records"] == summary["projected_entities"] == 0
    assert summary["issue_counts"] == {
        "missing_fields": 1,
        "unmapped_payload": 1,
        "unsupported_domain_schema": 1,
        "unverified_or_conflicted": 3,
    }
    assert query(output, "classes") == ()
    assert len(query(output, "source_records", "conflict")) == 2
    assert {row["verification_status"] for row in query(output, "source_records", "conflict")} == {
        "CONFLICTED"
    }
    assert json.loads(query(output, "projection_issues", "missing")[0]["details_json"]) == [
        "advancements"
    ]


def test_existing_project_fixture_is_explicitly_unmapped_not_reinterpreted(tmp_path):
    records = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json").read_text()
    )["records"]
    output = tmp_path / "domains"
    summary = build_domain_snapshot(source_snapshot(tmp_path, records), output)["summary"]
    assert summary["source_records"] == summary["unmapped_source_records"] == len(records)
    assert summary["projected_entities"] == 0
    assert summary["issue_counts"] == {"unmapped_payload": len(records)}


@pytest.mark.parametrize("missing", ["level", "cost", "effect_field"])
def test_missing_nested_fields_remove_whole_partial_entity_and_report_missing_data(
    tmp_path, missing
):
    records = authored_export()
    skill = selected(records, "skills", "throw")["payload"]
    if missing == "level":
        skill["levels"].pop()
    elif missing == "cost":
        skill["levels"][1].pop("mp_cost")
    else:
        skill["levels"][1]["effects"][0].pop("magnitude")
    output = tmp_path / "domains"
    report = build_domain_snapshot(source_snapshot(tmp_path, records), output)["summary"]
    assert report["mapped_source_records"] == 13
    assert report["unmapped_source_records"] == 1
    assert query(output, "skills") == ()
    assert query(output, "skill_levels") == ()
    assert query(output, "skill_effects") == ()
    assert query(output, "asset_index") == ()
    issue = query(output, "projection_issues", "throw")[0]
    assert issue["reason"] == "missing_fields"
    assert json.loads(issue["details_json"])


@pytest.mark.parametrize(
    "scope,version", [("synthetic_fixture", "fixture-v2"), ("game_reference", "fixture-v1")]
)
def test_same_entity_names_are_separated_by_version_and_scope(tmp_path, scope, version):
    records = authored_export()
    extra = copy.deepcopy(records)
    for record in extra:
        record["scope"] = scope
        record["game_version"] = version
    output = tmp_path / "domains"
    summary = build_domain_snapshot(source_snapshot(tmp_path, records + extra), output)["summary"]
    assert summary["projected_entities"] == 28
    assert len(query(output, "classes")) == 2
    assert len(query(output, "classes", scope=scope, version=version)) == 2
    assert query(output, "classes", version="missing") == ()


@pytest.mark.parametrize(
    "change",
    [
        "missing_item",
        "different_version",
        "different_scope",
        "conflicted_parent",
        "missing_portal",
        "missing_map",
        "missing_class",
    ],
)
def test_foreign_keys_never_fall_back_to_another_version_scope_or_conflict(tmp_path, change):
    records = authored_export()
    coin = selected(records, "items", "coin")
    if change == "missing_item":
        records.remove(coin)
    elif change == "different_version":
        coin["game_version"] = "fixture-v2"
    elif change == "different_scope":
        coin["scope"] = "game_reference"
    elif change == "conflicted_parent":
        other = copy.deepcopy(coin)
        other["payload"]["stack_limit"] = 12
        records.append(other)
    elif change == "missing_portal":
        selected(records, "maps", "field")["payload"]["portals"][0]["destination_portal_id"] = (
            "missing"
        )
    elif change == "missing_map":
        selected(records, "npcs", "guide")["payload"]["map_id"] = "missing"
    else:
        selected(records, "skills", "throw")["payload"]["class_id"] = "missing"
    source = source_snapshot(tmp_path, records)
    output = tmp_path / "domains"
    with pytest.raises(ValueError, match="reference"):
        build_domain_snapshot(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".domain-build-*"))


@pytest.mark.parametrize(
    "kind,identifier,path,value",
    [
        ("classes", "beginner", ("tier",), True),
        ("classes", "beginner", ("tier",), 1.0),
        ("classes", "beginner", ("domain_schema_version",), True),
        ("classes", "beginner", ("domain_schema_version",), 1.0),
        ("classes", "beginner", ("unknown_field",), 1),
        ("skills", "throw", ("levels", 0, "mp_cost"), -1),
        ("skills", "throw", ("levels", 0, "ammo_item_id"), None),
        ("skills", "throw", ("levels", 0, "level"), 2),
        ("skills", "throw", ("levels", 0, "effects", 0, "magnitude"), True),
        ("monsters", "blob", ("drops", 0, "probability"), 1.01),
        ("monsters", "blob", ("drops", 0, "probability"), True),
        ("monsters", "blob", ("drops", 0, "quantity_min"), 4),
        ("maps", "field", ("platforms", 0, "x2"), 641),
        ("maps", "field", ("platforms", 0, "x1"), 640),
        ("maps", "field", ("portals", 0, "x"), -1),
        ("items", "dagger", ("category",), "material"),
        ("items", "potion", ("consumable", "hp_restore"), "30"),
        ("quests", "collect", ("requirements", 4, "target_id"), "coin"),
        ("npcs", "guide", ("x",), 641),
        ("skills", "throw", ("assets", 0, "asset_id"), "../x.png"),
        ("skills", "throw", ("evidence", 0, "source_uri"), "javascript:run"),
        ("mechanics", "gravity", ("text_value",), "also a text value"),
    ],
)
def test_recognized_payloads_reject_bad_types_geometry_and_relationship_semantics(
    tmp_path, kind, identifier, path, value
):
    records = authored_export()
    target = selected(records, kind, identifier)["payload"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    source = source_snapshot(tmp_path, records)
    output = tmp_path / "domains"
    with pytest.raises(ValueError):
        build_domain_snapshot(source, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["advancements", "quest"])
def test_cyclic_dependency_graphs_rejected_without_recursion_limit(tmp_path, kind):
    records = authored_export()
    if kind == "advancements":
        selected(records, "classes", "thrower")["payload"]["advancements"] = [
            {"to_class_id": "beginner", "required_level": 1}
        ]
    else:
        selected(records, "quests", "intro")["payload"]["requirements"] = [
            {"requirement_id": "cycle", "kind": "quest", "target_id": "collect", "quantity": 1}
        ]
    source = source_snapshot(tmp_path, records)
    with pytest.raises(ValueError, match="cyclic"):
        build_domain_snapshot(source, tmp_path / "domains")


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE skills SET max_level=99 WHERE entity_id='throw'",
        "UPDATE source_records SET confidence=1",
        "DELETE FROM validation_evidence",
        "CREATE TABLE sqliteXconcealed (value TEXT)",
        "PRAGMA user_version=9",
    ],
)
def test_semantic_reprojection_detects_tamper_even_after_every_outer_hash_is_repaired(bundle, sql):
    _, target, _ = bundle
    with closing(sqlite3.connect(target / "domain.sqlite")) as connection:
        connection.execute(sql)
        connection.commit()
    rehash_bundle(target)
    with pytest.raises(ValueError, match="reprojection|schema"):
        verify_domain_snapshot(target)


def test_source_record_tamper_cannot_be_hidden_by_rehashing_source_and_domain_manifests(bundle):
    _, target, _ = bundle
    source = target / "source"
    with closing(sqlite3.connect(source / "knowledge.sqlite")) as connection:
        connection.execute("UPDATE catalog_records SET license='new assertion'")
        connection.commit()
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["sha256"] = hashlib.sha256((source / "knowledge.sqlite").read_bytes()).hexdigest()
    (source / "manifest.json").write_text(canonical_json(manifest) + "\n")
    rehash_bundle(target)
    with pytest.raises(ValueError, match="modified"):
        verify_domain_snapshot(target)


def test_unbound_byte_tamper_and_missing_or_additional_files_rejected(bundle):
    _, target, _ = bundle
    (target / "surprise").write_text("x")
    with pytest.raises(ValueError, match="unexpected"):
        verify_domain_snapshot(target)
    (target / "surprise").unlink()
    with (target / "domain.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        verify_domain_snapshot(target)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM classes",
        "UPDATE classes SET tier=3",
        "DROP TABLE classes",
        "CREATE TABLE arbitrary (x INTEGER)",
        "PRAGMA query_only=OFF",
        "PRAGMA user_version=5",
        "ATTACH ':memory:' AS other",
        "VACUUM",
    ],
)
def test_read_only_connection_blocks_mutation_even_after_pragma_attempts(bundle, sql):
    _, target, _ = bundle
    before = (target / "domain.sqlite").read_bytes()
    with closing(open_domain_readonly(target)) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(sql)
        assert connection.execute("SELECT count(*) FROM classes").fetchone()[0] == 2
    assert (target / "domain.sqlite").read_bytes() == before


def test_query_connection_is_exact_verified_snapshot_not_a_reopened_path(bundle):
    _, target, _ = bundle
    with closing(open_domain_readonly(target)) as connection:
        (target / "domain.sqlite").write_bytes(b"replaced after verification")
        assert (
            connection.execute("SELECT name FROM classes WHERE entity_id='thrower'").fetchone()[0]
            == "Test thrower"
        )
    with pytest.raises(ValueError):
        open_domain_readonly(target)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"table": "classes; DROP TABLE classes"},
        {"table": "sqlite_schema"},
        {"table": "knowledge_snapshots"},
        {"scope": "official"},
        {"scope": True},
        {"game_version": ""},
        {"entity_id": False},
        {"limit": True},
        {"limit": 0},
        {"limit": 1001},
        {"limit": 1.0},
    ],
)
def test_public_query_validates_table_values_and_bounds(bundle, kwargs):
    _, target, _ = bundle
    arguments = {"table": "classes", "game_version": "fixture-v1", "scope": "synthetic_fixture"}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        query_domain(target, **arguments)


def test_query_values_are_parameters_not_sql(bundle):
    _, target, _ = bundle
    assert query(target, "classes", "' OR 1=1 --") == ()
    assert len(query(target, "classes", limit=1)) == 1
    assert query(target, "classes") == query(target, "classes")


def test_existing_destination_is_never_replaced_and_source_cannot_contain_output(bundle):
    source, target, manifest = bundle
    with pytest.raises(FileExistsError):
        build_domain_snapshot(source, target)
    assert verify_domain_snapshot(target) == manifest
    with pytest.raises(ValueError, match="inside"):
        build_domain_snapshot(source, source / "nested-domains")


def test_publication_failure_removes_partial_output_but_preserves_source(tmp_path, monkeypatch):
    source = source_snapshot(tmp_path)
    before = (source / "knowledge.sqlite").read_bytes()
    original = catalog_domains.os.replace
    calls = []

    def fail_second(start, end):
        calls.append((start, end))
        if len(calls) == 2:
            raise OSError("injected publication fault")
        return original(start, end)

    monkeypatch.setattr(catalog_domains.os, "replace", fail_second)
    output = tmp_path / "domains"
    with pytest.raises(OSError, match="injected"):
        build_domain_snapshot(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".domain-build-*"))
    assert (source / "knowledge.sqlite").read_bytes() == before


def test_database_insert_failure_cannot_publish_a_partial_projection(tmp_path, monkeypatch):
    source = source_snapshot(tmp_path)
    original = catalog_domains._Projection.add

    def injected(self, table, row):
        if table == "skills":
            raise sqlite3.IntegrityError("independent injected storage failure")
        return original(self, table, row)

    monkeypatch.setattr(catalog_domains._Projection, "add", injected)
    with pytest.raises(ValueError, match="storage failure"):
        build_domain_snapshot(source, tmp_path / "domains")
    assert not (tmp_path / "domains").exists()
    assert not list(tmp_path.glob(".domain-build-*"))


@pytest.mark.parametrize(
    "limit", ["MAX_SOURCE_BYTES", "MAX_SOURCE_RECORDS", "MAX_DOMAIN_ROWS", "MAX_DOMAIN_BYTES"]
)
def test_resource_bounds_apply_before_publishing(tmp_path, monkeypatch, limit):
    source = source_snapshot(tmp_path)
    monkeypatch.setattr(catalog_domains, limit, 1)
    with pytest.raises(ValueError, match="budget"):
        build_domain_snapshot(source, tmp_path / "domains")
    assert not (tmp_path / "domains").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_source_sidecars_rejected(tmp_path, suffix):
    source = source_snapshot(tmp_path)
    (source / ("knowledge.sqlite" + suffix)).write_bytes(b"journal")
    with pytest.raises(ValueError, match="sidecar"):
        build_domain_snapshot(source, tmp_path / "domains")


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges are environment-dependent")
def test_symlink_inputs_outputs_and_dangling_sidecars_rejected(bundle, tmp_path):
    source, target, _ = bundle
    linked = tmp_path / "linked-source"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        build_domain_snapshot(linked, tmp_path / "new")
    dangling = tmp_path / "reserved-output"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        build_domain_snapshot(source, dangling)
    (source / "knowledge.sqlite-wal").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="sidecar"):
        build_domain_snapshot(source, tmp_path / "new")
    original = target / "domain.sqlite"
    saved = tmp_path / "saved.sqlite"
    original.rename(saved)
    original.symlink_to(saved)
    with pytest.raises(ValueError, match="nonsymlink"):
        verify_domain_snapshot(target)
