import hashlib
import json
import os
import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from maple_vision.catalog import (
    canonical_json,
    catalog_summary,
    import_catalog,
    initialize_catalog,
    load_import,
    query_catalog,
)
from maple_vision.knowledge_sync import LocalExportSource, apply_sync, plan_sync


def record(entity_id="fixture.a", **overrides):
    value = {
        "kind": "skills",
        "entity_id": entity_id,
        "game_version": "fixture-v1",
        "source_uri": "urn:maple-vision:self-authored:sync",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "verification_status": "VERIFIED",
        "scope": "synthetic_fixture",
        "payload": {"damage": 10},
    }
    value.update(overrides)
    return value


def export(tmp_path, records, name="source.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "知识 # 100%20.sqlite"
    initialize_catalog(path)
    return path


def entries(db):
    with sqlite3.connect(db) as connection:
        return connection.execute("SELECT * FROM catalog_records ORDER BY record_id").fetchall()


def edit_plan(path, change, *, resign=True):
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    if resign:
        body = {key: value for key, value in document.items() if key != "plan_id"}
        document["plan_id"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    path.write_text(canonical_json(document), encoding="utf-8")


def test_adapter_captures_exact_immutable_bytes(tmp_path):
    source = export(tmp_path, [record()])
    snapshot = LocalExportSource(source).read()
    assert snapshot.content == source.read_bytes()
    assert snapshot.records == load_import(source)
    with pytest.raises(FrozenInstanceError):
        snapshot.content = b"changed"


def test_plan_deterministic_readonly_and_apply_idempotent(db, tmp_path):
    source = export(tmp_path, [record()])
    before = db.read_bytes()
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    planned = plan_sync(db, source, first)
    again = plan_sync(db, source, second)
    assert first.read_bytes() == second.read_bytes()
    assert planned["plan_id"] == again["plan_id"]
    assert db.read_bytes() == before
    assert planned["delta_counts"] == dict(
        added=1, unchanged=0, changed=0, missing=0, status_changes=0
    )
    document = json.loads(first.read_text())
    assert document["source"]["export_utf8"].encode() == source.read_bytes()
    assert "database" not in document and "path" not in document["source"]
    applied = apply_sync(db, first)
    assert applied["status"] == "applied" and applied["imported"] == 1
    assert applied["catalog_fingerprint"] == planned["expected_fingerprint"]
    existing = entries(db)
    repeated = apply_sync(db, first)
    assert repeated["status"] == "already_applied" and repeated["idempotent"] is True
    assert repeated["imported"] == 0 and entries(db) == existing


def test_added_changed_unchanged_missing_and_conflicts(db, tmp_path):
    baseline = [record("fixture.a"), record("fixture.b"), record("fixture.missing")]
    import_catalog(db, export(tmp_path, baseline, "baseline.json"))
    incoming = [
        record("fixture.a"),
        record("fixture.b", payload={"damage": 22}),
        record("fixture.new"),
    ]
    source = export(tmp_path, incoming)
    plan = tmp_path / "plan.json"
    summary = plan_sync(db, source, plan)
    assert summary["delta_counts"] == dict(
        added=1, unchanged=1, changed=1, missing=2, status_changes=2
    )
    assert summary["expected_import"] == dict(
        imported=2, unchanged=1, conflicted_entities=1, records_total=5
    )
    document = json.loads(plan.read_text())
    assert document["source"]["game_versions"] == ["fixture-v1"]
    assert document["source"]["provenance"][0]["permission"] == "self_authored"
    apply_sync(db, plan)
    assert len(query_catalog(db, "skills", "fixture.missing", "fixture-v1")) == 1
    assert query_catalog(db, "skills", "fixture.b", "fixture-v1") == ()
    conflicting = query_catalog(db, "skills", "fixture.b", "fixture-v1", allow_unverified=True)
    assert len(conflicting) == 2
    assert {entry.payload["damage"] for entry in conflicting} == {10, 22}
    assert {entry.verification_status for entry in conflicting} == {"CONFLICTED"}


def test_provenance_change_is_new_history_even_if_payload_matches(db, tmp_path):
    import_catalog(db, export(tmp_path, [record()], "baseline.json"))
    plan = tmp_path / "plan.json"
    summary = plan_sync(db, export(tmp_path, [record(source_uri="urn:second:evidence")]), plan)
    assert summary["delta_counts"]["changed"] == 1
    assert summary["delta_counts"]["missing"] == 1
    assert summary["expected_import"]["conflicted_entities"] == 0
    apply_sync(db, plan)
    assert len(query_catalog(db, "skills", "fixture.a", "fixture-v1")) == 2


@pytest.mark.parametrize("status", ["UNKNOWN", "PROVISIONAL", "CONFLICTED"])
def test_unverified_status_never_promoted(db, tmp_path, status):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record(verification_status=status)]), plan)
    apply_sync(db, plan)
    assert query_catalog(db, "skills", "fixture.a", "fixture-v1") == ()
    (result,) = query_catalog(db, "skills", "fixture.a", "fixture-v1", allow_unverified=True)
    assert result.verification_status == status and result.declared_status == status


def test_conflicted_history_is_not_resolved_by_a_single_matching_export(db, tmp_path):
    import_catalog(
        db, export(tmp_path, [record(), record(payload={"damage": 22})], "baseline.json")
    )
    plan = tmp_path / "plan.json"
    summary = plan_sync(db, export(tmp_path, [record()]), plan)
    assert summary["delta_counts"]["unchanged"] == 1
    assert summary["delta_counts"]["missing"] == 1
    assert apply_sync(db, plan)["status"] == "no_changes"
    assert catalog_summary(db)["status_counts"]["CONFLICTED"] == 2


def test_versions_separate_and_old_version_remains_queryable(db, tmp_path):
    import_catalog(db, export(tmp_path, [record()], "baseline.json"))
    plan = tmp_path / "plan.json"
    summary = plan_sync(
        db, export(tmp_path, [record(game_version="fixture-v2", payload={"damage": 22})]), plan
    )
    assert summary["delta_counts"]["added"] == 1
    assert summary["delta_counts"]["changed"] == 0
    apply_sync(db, plan)
    assert query_catalog(db, "skills", "fixture.a", "fixture-v1")[0].payload == {"damage": 10}
    assert query_catalog(db, "skills", "fixture.a", "fixture-v2")[0].payload == {"damage": 22}


def test_empty_export_is_auditable_no_delete(db, tmp_path):
    import_catalog(db, export(tmp_path, [record()], "baseline.json"))
    plan = tmp_path / "plan.json"
    summary = plan_sync(db, export(tmp_path, []), plan)
    assert summary["delta_counts"]["missing"] == 1
    assert summary["baseline_fingerprint"] == summary["expected_fingerprint"]
    assert apply_sync(db, plan)["status"] == "no_changes"
    assert len(entries(db)) == 1


def test_duplicate_source_records_preserve_import_counts(db, tmp_path):
    plan = tmp_path / "plan.json"
    summary = plan_sync(db, export(tmp_path, [record(), record()]), plan)
    assert summary["delta_counts"]["added"] == 1
    assert summary["expected_import"] == dict(
        imported=1, unchanged=1, conflicted_entities=0, records_total=1
    )
    assert apply_sync(db, plan)["imported"] == 1


def test_original_export_can_change_or_disappear_after_snapshot(db, tmp_path):
    source = export(tmp_path, [record()])
    plan = tmp_path / "plan.json"
    plan_sync(db, source, plan)
    source.write_text("changed after planning")
    source.unlink()
    apply_sync(db, plan)
    assert query_catalog(db, "skills", "fixture.a", "fixture-v1")[0].payload == {"damage": 10}


def test_stale_plan_detects_unverified_addition_without_partial_changes(db, tmp_path):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    import_catalog(
        db,
        export(tmp_path, [record("unverified", verification_status="UNKNOWN")], "interleaved.json"),
    )
    before = entries(db)
    with pytest.raises(ValueError, match="stale"):
        apply_sync(db, plan)
    assert entries(db) == before


def test_reapply_requires_exact_expected_state_not_just_present_input_ids(db, tmp_path):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    apply_sync(db, plan)
    import_catalog(db, export(tmp_path, [record("later")], "later.json"))
    before = entries(db)
    with pytest.raises(ValueError, match="stale"):
        apply_sync(db, plan)
    assert entries(db) == before


def test_legacy_rows_participate_in_fingerprint(db, tmp_path):
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO source_records VALUES ('legacy','urn:old','v0','legacy-hash','UNKNOWN')"
        )
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE source_records SET source_uri='urn:changed'")
    with pytest.raises(ValueError, match="stale"):
        apply_sync(db, plan)
    assert entries(db) == []


def test_schema_change_invalidates_plan(db, tmp_path):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE extra(value TEXT)")
    with pytest.raises(ValueError, match="stale"):
        apply_sync(db, plan)
    assert entries(db) == []


def test_database_failure_rolls_back_earlier_insert_in_same_batch(db, tmp_path):
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON catalog_records
            WHEN NEW.entity_id='second' BEGIN SELECT RAISE(ABORT,'injected failure'); END""")
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record("first"), record("second")]), plan)
    before = db.read_bytes()
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        apply_sync(db, plan)
    assert entries(db) == []
    assert db.read_bytes() == before


def test_postcondition_failure_rolls_back_all_effects(db, tmp_path, monkeypatch):
    import maple_vision.knowledge_sync as sync

    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record("first"), record("second")]), plan)
    original = sync._import_records

    def faulty(connection, records):
        result = original(connection, records)
        connection.execute("DELETE FROM catalog_records WHERE entity_id='second'")
        return result

    monkeypatch.setattr(sync, "_import_records", faulty)
    with pytest.raises(ValueError, match="postcondition"):
        apply_sync(db, plan)
    assert entries(db) == []


def test_growth_over_byte_budget_rolls_back_entire_import(db, tmp_path, monkeypatch):
    import maple_vision.knowledge_sync as sync

    source = export(tmp_path, [record("small"), record("large", payload={"text": "x" * 20_000})])
    plan = tmp_path / "plan.json"
    before = db.read_bytes()
    monkeypatch.setattr(sync, "MAX_CATALOG_BYTES", len(before))
    planned = plan_sync(db, source, plan)
    with pytest.raises(ValueError, match="logical size.*byte budget"):
        apply_sync(db, plan)
    assert entries(db) == []
    assert db.read_bytes() == before
    after = plan_sync(db, source, tmp_path / "after-rollback.json")
    assert after["baseline_fingerprint"] == planned["baseline_fingerprint"]
    assert after["plan_id"] == planned["plan_id"]


def test_trigger_change_to_legacy_history_is_rolled_back(db, tmp_path):
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO source_records VALUES ('legacy','urn:old','v0','old-hash','UNKNOWN')"
        )
        connection.execute("""CREATE TRIGGER alter_history AFTER INSERT ON catalog_records
            BEGIN UPDATE source_records SET source_uri='urn:wrong'; END""")
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    with pytest.raises(ValueError, match="postcondition"):
        apply_sync(db, plan)
    assert entries(db) == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT source_uri FROM source_records").fetchone() == (
            "urn:old",
        )


def test_writer_lock_covers_baseline_check_and_import(db, tmp_path, monkeypatch):
    import maple_vision.knowledge_sync as sync

    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    original = sync._import_records
    checked = []

    def contested(connection, records):
        other = sqlite3.connect(db, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute(
                    "INSERT INTO source_records VALUES ('racer','urn:racer','v','hash','UNKNOWN')"
                )
            checked.append(True)
        finally:
            other.close()
        return original(connection, records)

    monkeypatch.setattr(sync, "_import_records", contested)
    assert apply_sync(db, plan)["status"] == "applied"
    assert checked == [True]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(expected_fingerprint="0" * 64),
        lambda p: p.update(baseline_fingerprint="0" * 64),
        lambda p: p["changes"]["added"].clear(),
        lambda p: p["expected_import"].update(imported=999),
        lambda p: p["expected_import"].update(imported=True),
        lambda p: p["source"].update(sha256="0" * 64),
        lambda p: p["source"].update(size_bytes=True),
        lambda p: p["source"].update(export_utf8=p["source"]["export_utf8"] + " "),
        lambda p: p["source"]["provenance"][0].update(permission="public_license"),
        lambda p: p["source"].update(game_versions=["made-up"]),
        lambda p: p["source"].update(path="../../private.json"),
        lambda p: p["source"].update(name="../source.json"),
        lambda p: p.update(sql="DROP TABLE catalog_records"),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(adapter="network_fetcher"),
    ],
)
def test_tampered_evidence_or_metadata_rejected_even_with_recomputed_plan_id(db, tmp_path, change):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    edit_plan(plan, change)
    with pytest.raises(ValueError):
        apply_sync(db, plan)
    assert entries(db) == []


def test_outer_integrity_and_duplicate_keys_rejected(db, tmp_path):
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record()]), plan)
    edit_plan(plan, lambda p: p.update(expected_fingerprint="0" * 64), resign=False)
    with pytest.raises(ValueError, match="integrity"):
        apply_sync(db, plan)
    plan.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        apply_sync(db, plan)


def test_baseline_record_tamper_cannot_be_hidden_by_outer_checksum(db, tmp_path):
    import_catalog(db, export(tmp_path, [record()], "baseline.json"))
    plan = tmp_path / "plan.json"
    plan_sync(db, export(tmp_path, [record("new")]), plan)
    before = entries(db)
    edit_plan(plan, lambda p: p["baseline"]["records"][0].update(license="forged"))
    with pytest.raises(ValueError, match="modified"):
        apply_sync(db, plan)
    assert entries(db) == before


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecars_are_not_ignored_for_plan_or_apply(db, tmp_path, suffix):
    source, plan = export(tmp_path, [record()]), tmp_path / "plan.json"
    plan_sync(db, source, plan)
    sidecar = db.with_name(db.name + suffix)
    sidecar.write_bytes(b"unmanifested data")
    with pytest.raises(ValueError, match="sidecars"):
        plan_sync(db, source, tmp_path / "another.json")
    with pytest.raises(ValueError, match="sidecars"):
        apply_sync(db, plan)
    sidecar.unlink()
    assert entries(db) == []


def test_wal_mode_requires_explicit_standalone_conversion(db, tmp_path):
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    with pytest.raises(ValueError, match="DELETE-journal"):
        plan_sync(db, export(tmp_path, [record()]), tmp_path / "plan.json")


def test_invalid_source_does_not_publish_a_plan(db, tmp_path):
    source = export(tmp_path, [record(), record("bad", permission="unknown")])
    plan = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="permission"):
        plan_sync(db, source, plan)
    assert not plan.exists() and entries(db) == []


def test_plan_file_is_never_overwritten(db, tmp_path):
    source, plan = export(tmp_path, [record()]), tmp_path / "plan.json"
    plan.write_text("keep me")
    with pytest.raises(FileExistsError):
        plan_sync(db, source, plan)
    assert plan.read_text() == "keep me"


@pytest.mark.skipif(
    os.name == "nt", reason="unprivileged Windows may not create filesystem symlinks"
)
def test_source_plan_database_and_sidecar_symlinks_are_rejected(db, tmp_path):
    source, plan = export(tmp_path, [record()]), tmp_path / "plan.json"
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        plan_sync(db, source_link, plan)
    plan_sync(db, source, plan)
    plan_link = tmp_path / "plan-link.json"
    plan_link.symlink_to(plan)
    with pytest.raises(ValueError, match="symlink"):
        apply_sync(db, plan_link)
    database_link = tmp_path / "database-link.sqlite"
    database_link.symlink_to(db)
    with pytest.raises(ValueError, match="regular local"):
        apply_sync(database_link, plan)
    sidecar = db.with_name(db.name + "-wal")
    sidecar.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="sidecars"):
        apply_sync(db, plan)
    sidecar.unlink()
    assert entries(db) == []
