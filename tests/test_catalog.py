import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from maple_vision.catalog import (
    catalog_summary,
    create_snapshot,
    import_catalog,
    initialize_catalog,
    migrate_catalog,
    open_catalog_readonly,
    query_catalog,
    verify_snapshot,
)
from maple_vision.knowledge import initialize


def record(**overrides):
    value = {
        "kind": "skills",
        "entity_id": "fixture.throw",
        "game_version": "fixture-v1",
        "source_uri": "urn:maple-vision:self-authored:fixture",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "verification_status": "VERIFIED",
        "scope": "synthetic_fixture",
        "payload": {"damage": 10, "label": "演示技能"},
    }
    value.update(overrides)
    return value


def source(tmp_path, *records):
    path = tmp_path / "import.json"
    path.write_text(json.dumps({"schema_version": 1, "records": list(records)}), encoding="utf-8")
    return path


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    initialize_catalog(path)
    return path


def test_v1_upgrade_preserves_records_and_hardens_null_ids(tmp_path):
    path = tmp_path / "old.sqlite"
    initialize(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO source_records VALUES ('old','urn:old','v1','abc','UNKNOWN')"
        )
    initialize_catalog(path)
    initialize_catalog(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT id FROM source_records").fetchall() == [("old",)]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO source_records VALUES (NULL,'urn:x','v1','x','UNKNOWN')"
            )


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_legacy_invalid_id_refuses_atomic_upgrade(tmp_path, bad_id):
    path = tmp_path / "old.sqlite"
    initialize(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO source_records VALUES (?,'urn:x','v1','abc','UNKNOWN')", (bad_id,)
        )
    with pytest.raises(ValueError, match="null/empty"):
        initialize_catalog(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT id FROM source_records").fetchone() == (bad_id,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE name='catalog_records'"
            ).fetchone()
            is None
        )


def test_migration_does_not_commit_outer_transaction():
    connection = sqlite3.connect(":memory:")
    connection.execute("BEGIN")
    migrate_catalog(connection)
    connection.rollback()
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    assert connection.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall() == []
    connection.close()


def test_unknown_schema_and_incompatible_v2_refused(tmp_path):
    path = tmp_path / "unknown.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=9")
    with pytest.raises(ValueError, match="unsupported"):
        initialize_catalog(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="incompatible"):
        initialize_catalog(path)


def test_import_is_idempotent_and_query_returns_canonical_payload(db, tmp_path):
    imported = import_catalog(db, source(tmp_path, record()))
    assert (imported.imported, imported.unchanged, imported.records_total) == (1, 0, 1)
    again = import_catalog(db, source(tmp_path, record()))
    assert (again.imported, again.unchanged, again.records_total) == (0, 1, 1)
    (result,) = query_catalog(db, "skills", "fixture.throw", "fixture-v1")
    assert result.payload == {"damage": 10, "label": "演示技能"}
    assert result.scope == "synthetic_fixture"
    assert result.payload_sha256 == hashlib.sha256(result.payload_json.encode()).hexdigest()
    assert catalog_summary(db)["records"] == 1


def test_different_versions_do_not_conflict(db, tmp_path):
    result = import_catalog(
        db, source(tmp_path, record(), record(game_version="fixture-v2", payload={"damage": 12}))
    )
    assert result.conflicted_entities == 0
    assert len(query_catalog(db, "skills", "fixture.throw", "fixture-v2")) == 1


def test_conflict_preserves_both_sources_and_filters_default_query(db, tmp_path):
    import_catalog(db, source(tmp_path, record()))
    imported = import_catalog(
        db, source(tmp_path, record(source_uri="urn:second", payload={"damage": 99}))
    )
    assert imported.conflicted_entities == 1
    assert query_catalog(db, "skills", "fixture.throw", "fixture-v1") == ()
    rows = query_catalog(db, "skills", "fixture.throw", "fixture-v1", allow_unverified=True)
    assert {row.payload["damage"] for row in rows} == {10, 99}
    assert {row.verification_status for row in rows} == {"CONFLICTED"}
    assert {row.declared_status for row in rows} == {"VERIFIED"}
    assert import_catalog(db, source(tmp_path, record())).unchanged == 1


def test_same_payload_different_provenance_retained_without_conflict(db, tmp_path):
    imported = import_catalog(
        db, source(tmp_path, record(), record(source_uri="https://example.org/reference"))
    )
    assert imported.imported == 2
    assert imported.conflicted_entities == 0
    assert len(query_catalog(db, "skills", "fixture.throw", "fixture-v1")) == 2


@pytest.mark.parametrize("status", ["UNKNOWN", "PROVISIONAL", "CONFLICTED"])
def test_unverified_data_requires_inspection_flag(db, tmp_path, status):
    import_catalog(db, source(tmp_path, record(verification_status=status)))
    assert query_catalog(db, "skills", "fixture.throw", "fixture-v1") == ()
    assert (
        query_catalog(db, "skills", "fixture.throw", "fixture-v1", allow_unverified=True)[
            0
        ].verification_status
        == status
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "exec"},
        {"entity_id": ""},
        {"game_version": " "},
        {"source_uri": "javascript:x"},
        {"source_uri": "https://user:secret@example.org"},
        {"source_uri": "https:///missing"},
        {"permission": "unknown"},
        {"license": ""},
        {"collected_at": "2026-11-24"},
        {"collected_at": "yesterday"},
        {"scope": "official"},
        {"verification_status": "verified"},
        {"payload": [1]},
        {"payload": {"damage": float("nan")}},
        {"extra": 1},
    ],
)
def test_invalid_batch_is_rejected_without_partial_write(db, tmp_path, overrides):
    with pytest.raises(ValueError):
        import_catalog(
            db, source(tmp_path, record(), record(**{"entity_id": "second", **overrides}))
        )
    assert catalog_summary(db)["records"] == 0


def test_duplicate_json_keys_and_bool_schema_rejected(db, tmp_path):
    path = tmp_path / "import.json"
    path.write_text('{"schema_version":1,"records":[],"records":[]}')
    with pytest.raises(ValueError, match="duplicate"):
        import_catalog(db, path)
    path.write_text('{"schema_version":true,"records":[]}')
    with pytest.raises(ValueError, match="schema"):
        import_catalog(db, path)


def test_import_requires_existing_initialized_catalog(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        import_catalog(missing, source(tmp_path, record()))
    assert not missing.exists()
    initialize(missing)
    with pytest.raises(ValueError, match="v2"):
        import_catalog(missing, source(tmp_path, record()))


def test_readonly_connection_and_uri_escaping(tmp_path):
    path = tmp_path / "知识 ?#.sqlite"
    initialize_catalog(path)
    connection = open_catalog_readonly(path)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM catalog_records")
    connection.close()


def test_tampered_record_and_missing_conflict_flags_detected(db, tmp_path):
    import_catalog(db, source(tmp_path, record()))
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE catalog_records SET license='changed'")
    with pytest.raises(ValueError, match="modified"):
        query_catalog(db, "skills", "fixture.throw", "fixture-v1")


def test_inconsistent_conflict_flags_detected(db, tmp_path):
    import_catalog(db, source(tmp_path, record(), record(payload={"damage": 99})))
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE catalog_records SET verification_status='VERIFIED'")
    with pytest.raises(ValueError, match="conflict"):
        catalog_summary(db)


def test_status_cannot_silently_promote_unknown(db, tmp_path):
    import_catalog(db, source(tmp_path, record(verification_status="UNKNOWN")))
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE catalog_records SET verification_status='VERIFIED'")
    with pytest.raises(ValueError, match="upgrade"):
        catalog_summary(db)


def test_snapshot_reproducible_manifest_and_independence(db, tmp_path):
    import_catalog(db, source(tmp_path, record()))
    snapshot = tmp_path / "snapshot"
    manifest = create_snapshot(db, snapshot)
    assert verify_snapshot(snapshot) == manifest
    import_catalog(db, source(tmp_path, record(entity_id="second")))
    assert verify_snapshot(snapshot)["summary"]["records"] == 1
    assert catalog_summary(db)["records"] == 2
    with pytest.raises(FileExistsError):
        create_snapshot(db, snapshot)
    assert verify_snapshot(snapshot) == manifest


def test_snapshot_tamper_detected(db, tmp_path):
    snapshot = tmp_path / "snapshot"
    create_snapshot(db, snapshot)
    with (snapshot / "knowledge.sqlite").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="SHA256"):
        verify_snapshot(snapshot)


def test_snapshot_manifest_traversal_and_summary_tamper_rejected(db, tmp_path):
    snapshot = tmp_path / "snapshot"
    manifest = create_snapshot(db, snapshot)
    bad = copy.deepcopy(manifest)
    bad["database"] = "../knowledge.sqlite"
    (snapshot / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest"):
        verify_snapshot(snapshot)
    manifest["summary"]["records"] = 999
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="summary"):
        verify_snapshot(snapshot)


def test_snapshot_refuses_corrupt_source_before_destination_created(db, tmp_path):
    import_catalog(db, source(tmp_path, record()))
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE catalog_records SET payload_sha256=?", ("f" * 64,))
    with pytest.raises(ValueError, match="modified"):
        create_snapshot(db, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_fixture_all_kinds_importable(tmp_path):
    path = tmp_path / "fixture.sqlite"
    initialize_catalog(path)
    result = import_catalog(path, Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json")
    assert result.imported >= 8
    summary = catalog_summary(path)
    assert summary["scopes"] == ["synthetic_fixture"]
    assert summary["game_versions"] == ["fixture-v1"]


def test_database_failure_rolls_back_entire_import(db, tmp_path):
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON catalog_records
            WHEN NEW.entity_id = 'second'
            BEGIN SELECT RAISE(ABORT, 'test database fault'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test database fault"):
        import_catalog(db, source(tmp_path, record(), record(entity_id="second")))
    assert catalog_summary(db)["records"] == 0
