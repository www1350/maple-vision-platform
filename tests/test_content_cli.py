"""External class content is explicit, portable, and bound at every evidence copy."""

import hashlib
import json
from pathlib import Path

import pytest

from maple_vision.class_package import ClassPackage, load_class_package
from maple_vision.cli import main
from maple_vision.lab import (
    CapabilityRunConfig,
    RunConfig,
    run_episode,
    run_soak,
    run_suite,
    verify_perception,
    verify_run,
)
from maple_vision.runtime_pipeline import RuntimeConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/classes"


def _rewrite(bundle, name, value):
    path = bundle / name
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def short_class_run(tmp_path):
    source = tmp_path / "class.json"
    source.write_bytes((EXAMPLES / "archer.json").read_bytes())
    package = load_class_package(source)
    output = tmp_path / "run"
    report = run_episode(
        output,
        CapabilityRunConfig(job=package.class_id, max_steps=7, record_stride=1),
        class_package=package,
        runtime=RuntimeConfig(),
    )
    source.unlink()
    return output, report, package


def test_deleted_package_replays_and_reinfers_actual_terminal(short_class_run):
    output, report, package = short_class_run
    assert report["status"] == "failed" and report["reason"] == "step_budget_exhausted"
    assert report["class_package"] == package.manifest()
    assert json.loads((output / "initial-state.json").read_text())["schema_version"] == 4
    assert json.loads((output / "manifest.json").read_text())["schema_version"] == 2
    assert verify_run(output)["verified"]
    reinfer = verify_perception(output)
    assert reinfer["structured_runtime_reproduced"]
    assert reinfer["belief_frames_recomputed"] == report["steps"] + 1
    assert reinfer["frames_reinferred"] == 8
    # A budget stop records the final observation and interruption; no extra
    # policy decision was evaluated on that terminal frame.
    assert reinfer["terminal_decision_compared"] is False


@pytest.mark.parametrize("name", ["summary.json", "initial-state.json", "class-package.json"])
def test_rehashed_disagreeing_class_copy_is_rejected(short_class_run, name):
    output, _, package = short_class_run
    document = json.loads((output / name).read_text())
    changed = package.to_dict()
    changed["version"] = "9.0.0"
    manifest = ClassPackage.from_dict(changed).manifest()
    if name == "class-package.json":
        document = manifest
    else:
        document["class_package"] = manifest
    _rewrite(output, name, document)
    with pytest.raises(ValueError, match="class package"):
        verify_run(output)


def test_class_copy_must_be_manifested(short_class_run):
    output, _, _ = short_class_run
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("class-package.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="class package"):
        verify_run(output)


@pytest.mark.parametrize(
    "location",
    [
        "step-state",
        "step-decision",
        "step-last-decision",
        "terminal-state",
        "terminal-interruption",
        "terminal-last-decision",
    ],
)
def test_static_capability_identity_is_checked_even_by_action_replay(short_class_run, location):
    output, _, _ = short_class_run
    terminal = location.startswith("terminal")
    name = "summary.json" if terminal else "trace.jsonl"
    if terminal:
        document = json.loads((output / name).read_text())
        data = document["runtime_evidence"]["terminal"]
    else:
        rows = [json.loads(line) for line in (output / name).read_text().splitlines()]
        data = rows[0]["runtime"]
    if location.endswith("-state"):
        target = data["state"]
    elif location.endswith("-last-decision"):
        target = data["state"]["last_decision"]["evidence"]
    else:
        target = data["interruption" if terminal else "decision"]["evidence"]
    target["capabilities"]["package_sha256"] = "0" * 64
    if terminal:
        _rewrite(output, name, document)
    else:
        path = output / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="capability identity"):
        verify_run(output)


def test_package_budget_is_checked_before_unbounded_read(short_class_run, monkeypatch):
    output, _, _ = short_class_run
    target = output / "class-package.json"
    target.write_bytes(b" " * (128 * 1024 + 1))
    original = Path.read_bytes

    def guarded(path):
        if path == target:
            raise AssertionError("must reject oversized class evidence before full read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="class package"):
        verify_run(output)


def test_builtin_structured_run_cannot_claim_an_external_capability_identity(tmp_path):
    output = tmp_path / "builtin"
    run_episode(output, RunConfig(max_steps=2), runtime=RuntimeConfig())
    summary = json.loads((output / "summary.json").read_text())
    summary["runtime_evidence"]["terminal"]["state"]["capabilities"] = {
        "class_id": "archer",
        "package_sha256": "0" * 64,
    }
    _rewrite(output, "summary.json", summary)
    with pytest.raises(ValueError, match="capability identity"):
        verify_run(output)


@pytest.mark.parametrize("factory", [run_suite, run_soak])
def test_external_batch_requires_unique_packages_and_structured_runtime(tmp_path, factory):
    package = load_class_package(EXAMPLES / "archer.json")
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="structured"):
        factory(output, class_packages=(package,))
    with pytest.raises(ValueError, match="unique"):
        factory(output, class_packages=(package, package), runtime=RuntimeConfig())
    assert not output.exists()


def test_invalid_package_fails_before_output_or_class_attribute_access(tmp_path):
    with pytest.raises(ValueError, match="class packages"):
        run_episode(tmp_path / "bad", class_package="invalid", runtime=RuntimeConfig())
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="unknown job"):
        RunConfig(job="archer")


def test_cli_class_compilation_is_distinct_from_mission_acceptance(capsys):
    assert main(["classes", "check", str(EXAMPLES / "archer.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verified"]
    assert result["capabilities"]["class_id"] == "archer"
    assert "mission_not_run" in result["validation_scope"]


def test_cli_infers_package_class_and_structured_runtime(tmp_path, capsys):
    output = tmp_path / "run"
    assert (
        main(
            [
                "lab",
                "run",
                "--output",
                str(output),
                "--class-pack",
                str(EXAMPLES / "bandit.json"),
                "--max-steps",
                "3",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["config"]["job"] == "bandit" and result["schema_version"] == 2
    assert result["status"] == "failed" and result["reason"] == "step_budget_exhausted"
    assert main(["lab", "reinfer", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["frames_reinferred"] == 4


@pytest.mark.parametrize(
    "arguments",
    [
        ["--runtime", "legacy"],
        ["--job", "assassin"],
        ["--knowledge-db", "unused.sqlite"],
    ],
)
def test_cli_rejects_incompatible_class_options_before_output(tmp_path, capsys, arguments):
    output = tmp_path / "run"
    assert (
        main(
            [
                "lab",
                "run",
                "--output",
                str(output),
                "--class-pack",
                str(EXAMPLES / "archer.json"),
                *arguments,
            ]
        )
        == 2
    )
    assert capsys.readouterr().err
    assert not output.exists()
