"""Independent provenance, frozen snapshot and non-destructive migration checks."""

import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest
from PIL import Image

from maple_vision.asset_registry import import_asset, verify_asset
from maple_vision.catalog import (
    canonical_json,
    create_snapshot,
    import_catalog,
    initialize_catalog,
    migrate_catalog,
    query_catalog,
    validate_provenance,
    verify_snapshot,
)
from maple_vision.knowledge import migrate


def record(damage=10, status="VERIFIED"):
    return {
        "kind": "skills",
        "entity_id": "test",
        "game_version": "v1",
        "source_uri": "urn:self:test",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "scope": "synthetic_fixture",
        "verification_status": status,
        "payload": {"damage": damage},
    }


def import_record(db, path, item):
    path.write_text(json.dumps({"schema_version": 1, "records": [item]}), encoding="utf-8")
    return import_catalog(db, path)


@pytest.mark.parametrize(
    "uri", ["https://:secret@example.org", "https://@example.org", "http://:80/x"]
)
def test_review_provenance_rejects_empty_username_credentials_or_missing_host(uri):
    with pytest.raises(ValueError):
        validate_provenance(uri, "explicit_permission", "user-approved copy")


def test_review_invalid_asset_provenance_has_no_side_effects(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(image)
    registry = tmp_path / "registry"
    with pytest.raises(ValueError):
        import_asset(
            registry,
            image,
            entity_id="x",
            game_version="v1",
            source_uri="https://:secret@example.org",
            license="CC0-1.0",
            permission="explicit_permission",
        )
    assert not registry.exists()


def test_review_snapshots_from_wal_source_are_standalone(tmp_path):
    db = tmp_path / "source.sqlite"
    initialize_catalog(db)
    import_record(db, tmp_path / "import.json", record())
    with sqlite3.connect(db) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        snapshot = tmp_path / "snapshot"
        manifest = create_snapshot(db, snapshot)
        connection = sqlite3.connect(snapshot / "knowledge.sqlite")
        try:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            connection.close()
        assert verify_snapshot(snapshot) == manifest
        assert not list(snapshot.glob("knowledge.sqlite-*"))


def test_review_unhashed_wal_pages_cannot_change_a_verified_snapshot(tmp_path):
    db = tmp_path / "source.sqlite"
    initialize_catalog(db)
    source = tmp_path / "import.json"
    import_record(db, source, record())
    snapshot = tmp_path / "snapshot"
    manifest = create_snapshot(db, snapshot)
    target = snapshot / "knowledge.sqlite"
    # Reproduce an older or externally edited WAL-mode snapshot with a legitimate
    # hash for its base database but untracked committed pages in a sidecar.
    connection = sqlite3.connect(target)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        manifest["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        connection.execute("DELETE FROM catalog_records")
        connection.commit()
        import_record(target, source, record(damage=999))
        assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest["sha256"]
        assert query_catalog(target, "skills", "test", "v1")[0].payload["damage"] == 999
        with pytest.raises(ValueError, match="journal|sidecar"):
            verify_snapshot(snapshot)
    finally:
        connection.close()


@pytest.mark.parametrize("value", [True, 1.0])
def test_review_snapshot_summary_counts_are_not_coerced(tmp_path, value):
    db = tmp_path / "source.sqlite"
    initialize_catalog(db)
    import_record(db, tmp_path / "import.json", record())
    snapshot = tmp_path / "snapshot"
    manifest = create_snapshot(db, snapshot)
    manifest["summary"]["records"] = value
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        verify_snapshot(snapshot)


@pytest.mark.parametrize("foreign_keys", [False, True])
def test_review_v1_upgrade_does_not_retarget_extensions_to_a_dropped_table(foreign_keys):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"PRAGMA foreign_keys={int(foreign_keys)}")
        migrate(connection)
        connection.execute("CREATE TABLE child (source_id TEXT REFERENCES source_records(id))")
        original = connection.execute("SELECT sql FROM sqlite_schema WHERE name='child'").fetchone()
        with pytest.raises(ValueError, match="dependent"):
            migrate_catalog(connection)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute("SELECT sql FROM sqlite_schema WHERE name='child'").fetchone()
            == original
        )
        connection.execute("INSERT INTO source_records VALUES ('s', 'urn:s', 'v1', 'h', 'UNKNOWN')")
        connection.execute("INSERT INTO child VALUES ('s')")
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE VIEW source_view AS SELECT id FROM source_records",
        "CREATE INDEX source_uri_custom ON source_records(source_uri)",
        "CREATE TRIGGER source_audit AFTER INSERT ON source_records BEGIN SELECT 1; END",
    ],
)
def test_review_v1_upgrade_refuses_to_destroy_dependent_views_indexes_or_triggers(sql):
    connection = sqlite3.connect(":memory:")
    try:
        migrate(connection)
        connection.execute(sql)
        before = connection.execute("SELECT name, sql FROM sqlite_schema ORDER BY name").fetchall()
        with pytest.raises(ValueError, match="dependent"):
            migrate_catalog(connection)
        assert (
            connection.execute("SELECT name, sql FROM sqlite_schema ORDER BY name").fetchall()
            == before
        )
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        connection.close()


def test_review_unverified_query_requires_a_boolean_opt_in(tmp_path):
    db = tmp_path / "source.sqlite"
    initialize_catalog(db)
    import_record(db, tmp_path / "import.json", record(status="UNKNOWN"))
    with pytest.raises(ValueError, match="boolean"):
        query_catalog(db, "skills", "test", "v1", allow_unverified="false")
    assert query_catalog(db, "skills", "test", "v1") == ()


def test_review_payload_mutation_does_not_change_the_catalog_record(tmp_path):
    db = tmp_path / "source.sqlite"
    initialize_catalog(db)
    import_record(db, tmp_path / "import.json", record())
    result = query_catalog(db, "skills", "test", "v1")[0]
    value = result.payload
    value["damage"] = 999
    assert result.payload["damage"] == 10
    assert query_catalog(db, "skills", "test", "v1")[0].payload["damage"] == 10


def test_review_asset_path_traversal_is_rejected_even_with_recomputed_metadata_hash(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(image)
    registry = tmp_path / "registry"
    entry = import_asset(
        registry,
        image,
        entity_id="x",
        game_version="v1",
        source_uri="urn:self:x",
        license="CC0-1.0",
        permission="self_authored",
    )
    value = asdict(entry)
    value.pop("asset_id")
    value["relative_path"] = "../source.png"
    forged_id = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    value["asset_id"] = forged_id
    (registry / "records" / f"{forged_id}.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content-addressed"):
        verify_asset(registry, forged_id)
