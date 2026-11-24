"""End-to-end contracts for lifecycle data, knowledge deltas and model validation CLI."""

import hashlib
import json

import pytest

from maple_vision.cli import main


def _run(arguments, capsys, expected=0):
    assert main([str(value) for value in arguments]) == expected
    captured = capsys.readouterr()
    if expected == 0:
        assert captured.err == ""
        return json.loads(captured.out)
    assert captured.out == ""
    return captured.err


def _item(entity_id="potion", payload=None, status="VERIFIED"):
    return {
        "kind": "items",
        "entity_id": entity_id,
        "game_version": "fixture-v1",
        "source_uri": "urn:maple-vision:sync-cli-fixture",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "scope": "synthetic_fixture",
        "verification_status": status,
        "payload": payload or {"name": "fixture potion"},
    }


def _export(path, records):
    path.write_text(json.dumps({"schema_version": 1, "records": records}), encoding="utf-8")
    return path


def test_lifecycle_cli_defaults_and_legacy_defaults_are_distinct(tmp_path, capsys):
    current = _run(
        ["dataset", "create", tmp_path / "lifecycle", "--profile", "lifecycle", "--sessions", 3],
        capsys,
    )
    assert current["schema_version"] == 2
    assert current["frames"] == 144
    assert current["validated"]
    for split in ("train", "val", "test"):
        counts = current["coverage"]["by_split"][split]["phase_counts"]
        assert all(
            counts[name] > 0
            for name in ("spawn", "combat", "loot", "cleared", "dialogue", "transition")
        )
    legacy = _run(["dataset", "create", tmp_path / "legacy", "--sessions", 3], capsys)
    assert legacy["schema_version"] == 1
    assert legacy["frames"] == 24
    assert "coverage" not in legacy


def test_sync_cli_is_self_contained_idempotent_and_preserves_conflicts(tmp_path, capsys):
    database = tmp_path / "knowledge.sqlite"
    source = _export(tmp_path / "source.json", [_item()])
    plan = tmp_path / "sync-plan.json"
    _run(["knowledge", "init", database], capsys)
    planned = _run(["knowledge", "plan-sync", database, source, plan], capsys)
    assert planned["delta_counts"]["added"] == 1
    source.unlink()
    applied = _run(["knowledge", "apply-sync", database, plan], capsys)
    assert applied["status"] == "applied"
    assert _run(["knowledge", "apply-sync", database, plan], capsys)["status"] == "already_applied"
    verified = _run(["knowledge", "query", database, "items", "potion", "fixture-v1"], capsys)
    assert len(verified["records"]) == 1

    _export(
        source, [_item(payload={"name": "conflicting fixture"}), _item("unknown", status="UNKNOWN")]
    )
    second_plan = tmp_path / "second-plan.json"
    _run(["knowledge", "plan-sync", database, source, second_plan], capsys)
    _run(["knowledge", "apply-sync", database, second_plan], capsys)
    assert not _run(["knowledge", "query", database, "items", "potion", "fixture-v1"], capsys)[
        "records"
    ]
    records = _run(
        ["knowledge", "query", database, "items", "potion", "fixture-v1", "--allow-unverified"],
        capsys,
    )["records"]
    assert len(records) == 2
    assert {record["verification_status"] for record in records} == {"CONFLICTED"}
    unknown = _run(
        ["knowledge", "query", database, "items", "unknown", "fixture-v1", "--allow-unverified"],
        capsys,
    )["records"]
    assert unknown[0]["verification_status"] == "UNKNOWN"


def test_stale_sync_cli_returns_error_without_importing_any_planned_row(tmp_path, capsys):
    database = tmp_path / "knowledge.sqlite"
    source = _export(tmp_path / "source.json", [_item()])
    plan = tmp_path / "sync-plan.json"
    _run(["knowledge", "init", database], capsys)
    _run(["knowledge", "plan-sync", database, source, plan], capsys)
    other = _export(tmp_path / "other.json", [_item("intervening")])
    _run(["knowledge", "import", database, other], capsys)
    error = _run(["knowledge", "apply-sync", database, plan], capsys, expected=2)
    assert "stale" in error.lower()
    assert not _run(["knowledge", "query", database, "items", "potion", "fixture-v1"], capsys)[
        "records"
    ]


@pytest.mark.parametrize("operation", ["suite", "soak", "reinfer"])
def test_model_validation_cli_rejects_an_unpaired_model_before_side_effects(
    tmp_path, capsys, operation
):
    output = tmp_path / "output"
    target = [output] if operation == "reinfer" else ["--output", output]
    error = _run(
        ["lab", operation, *target, "--model", tmp_path / "missing.onnx"], capsys, expected=2
    )
    assert "--model and --manifest" in error
    assert not output.exists()


def test_reinfer_cli_checks_actual_perception_without_changing_evidence(tmp_path, capsys):
    output = tmp_path / "mission"
    mission = _run(["lab", "run", "--output", output, "--record-stride", 16], capsys)
    assert mission["independent_goal_check"]
    before = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    result = _run(["lab", "reinfer", output], capsys)
    assert result["perception_reproduced"] is True
    assert result["policy_reproduced"] is True
    after = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    assert after == before
