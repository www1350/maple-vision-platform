"""Public entry-point and configuration compatibility for the structured engine."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.lab import RunConfig, run_episode
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.state_estimation import StateConfig
from maple_vision.task_runtime import TaskRuntimeConfig


def test_cli_selects_structured_and_reinference_restores_it_without_override(tmp_path):
    bundle = tmp_path / "structured"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maple_vision",
            "lab",
            "run",
            "--output",
            str(bundle),
            "--runtime",
            "structured",
            "--max-steps",
            "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    summary = json.loads(result.stdout)
    assert summary["schema_version"] == 2 and summary["steps"] == 3
    assert summary["runtime_evidence"]["terminal"]["policy_evaluated"] is False
    before = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    verified = subprocess.run(
        [sys.executable, "-m", "maple_vision", "lab", "reinfer", str(bundle)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr
    report = json.loads(verified.stdout)
    assert report["structured_runtime_reproduced"]
    assert report["belief_frames_recomputed"] == 4
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}


def test_legacy_default_does_not_add_new_runtime_fields(tmp_path):
    summary = run_episode(tmp_path / "legacy", RunConfig(max_steps=2))
    assert summary["schema_version"] == 1
    assert "runtime_evidence" not in summary
    assert not (tmp_path / "legacy/runtime-config.json").exists()
    rows = (tmp_path / "legacy/trace.jsonl").read_text().splitlines()
    assert all("runtime" not in json.loads(row) for row in rows)


def test_nondefault_config_roundtrip_preserves_identity_and_copies_containers():
    config = RuntimeConfig(
        state=StateConfig(max_tracks=12, max_detections=12),
        tasks=TaskRuntimeConfig(intent_ttl_ms=125, clear_confirmations=5),
    )
    manifest = config.manifest()
    restored = RuntimeConfig.from_manifest(manifest)
    assert restored == config
    assert config.content_sha256 != RuntimeConfig().content_sha256
    manifest["settings"]["state"]["max_tracks"] = 13
    assert config.state.max_tracks == 12
    with pytest.raises(ValueError, match="checksum"):
        RuntimeConfig.from_manifest(manifest)


@pytest.mark.parametrize("bad", [None, [], 1, True, {"schema_version": 1}])
def test_malformed_runtime_documents_raise_public_value_error(bad):
    with pytest.raises(ValueError):
        RuntimeConfig.from_dict(bad)
    with pytest.raises(ValueError):
        RuntimeConfig.from_manifest(bad)


def test_configuration_never_accepts_implicit_engine_or_partial_defaults():
    for key in ("schema_version", "engine", "state", "tasks"):
        document = RuntimeConfig().to_dict()
        del document[key]
        with pytest.raises(ValueError):
            RuntimeConfig.from_dict(document)
    for field in ("intent_ttl_ms", "history_limit"):
        document = RuntimeConfig().to_dict()
        del document["tasks"][field]
        with pytest.raises(ValueError):
            RuntimeConfig.from_dict(document)


def test_invalid_public_config_is_rejected_before_creating_output(tmp_path):
    output: Path = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        run_episode(output, RunConfig(max_steps=1), runtime={})
    assert not output.exists()
    with pytest.raises(ValueError):
        replace(RuntimeConfig(), state={})
