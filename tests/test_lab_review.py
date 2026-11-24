"""Independent checks that run evidence cannot claim outcomes it did not produce."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.agent import AgentDecision
from maple_vision.lab import RunConfig, run_episode, verify_run
from maple_vision.sim_types import SimAction


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def update_manifest_for(output, relative):
    """A digest is not authentication: recomputed hashes must not hide contradictions."""
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    write_json(path, manifest)


@pytest.fixture
def short_bundle(tmp_path: Path):
    output = tmp_path / "run"
    result = run_episode(output, RunConfig(max_steps=2, record_stride=1))
    assert result["status"] == "failed"
    assert not result["independent_goal_check"]
    assert verify_run(output)["steps_replayed"] == 2
    return output


def test_review_false_policy_completion_is_rejected_by_independent_goal_oracle(monkeypatch):
    monkeypatch.setattr(
        "maple_vision.lab.VisualAgent.decide",
        lambda self, observation: AgentDecision(
            SimAction(), "done", "completed", "unearned_success"
        ),
    )
    result = run_episode(config=RunConfig(max_steps=1))
    assert result["status"] == "failed"
    assert result["reason"] == "independent_goal_check_failed"
    assert not result["independent_goal_check"]
    assert result["steps"] == 0


def test_review_changed_file_without_updated_hash_is_rejected(short_bundle):
    (short_bundle / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(short_bundle)


def test_review_manifest_path_cannot_escape_bundle(short_bundle, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    path = short_bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["../outside.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    write_json(path, manifest)
    with pytest.raises(ValueError, match="escapes"):
        verify_run(short_bundle)


def test_review_replay_rejects_a_forged_success_summary(short_bundle):
    path = short_bundle / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["status"] = "completed"
    summary["independent_goal_check"] = True
    summary["reason"] = "dialogue_observed"
    write_json(path, summary)
    update_manifest_for(short_bundle, "summary.json")
    with pytest.raises(ValueError):
        verify_run(short_bundle)


@pytest.mark.parametrize("field,value", [("seed", 1893), ("job", "cleric")])
def test_review_replay_configuration_must_match_initial_state(short_bundle, field, value):
    path = short_bundle / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["config"][field] = value
    write_json(path, summary)
    update_manifest_for(short_bundle, "summary.json")
    with pytest.raises(ValueError):
        verify_run(short_bundle)


@pytest.mark.parametrize("relative", ["frames/000000.png", "final.png"])
def test_review_saved_images_must_match_the_replayed_scene(short_bundle, relative):
    image = Image.new("RGB", (320, 192), (255, 255, 255))
    image.save(short_bundle / relative)
    update_manifest_for(short_bundle, relative)
    with pytest.raises(ValueError):
        verify_run(short_bundle)


@pytest.mark.parametrize("version", [True, 1.0])
def test_review_bundle_schema_requires_an_integer(short_bundle, version):
    path = short_bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = version
    write_json(path, manifest)
    with pytest.raises(ValueError):
        verify_run(short_bundle)
