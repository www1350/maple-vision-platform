"""Actual model re-inference, terminal evidence and bounded process accounting."""

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.asset_registry import import_asset
from maple_vision.catalog import import_catalog, initialize_catalog
from maple_vision.lab import (
    RunConfig,
    run_episode,
    run_soak,
    run_suite,
    verify_perception,
    verify_run,
)
from maple_vision.onnx_detector import OnnxDetector
from maple_vision.profiles import load_class_profile
from maple_vision.sprite_pack import load_sprite_pack
from maple_vision.validation_resources import (
    ProcessMemory,
    ProcessResourceMonitor,
    read_process_memory,
)
from maple_vision.vision import PixelPerception

os.environ["ORT_DISABLE_TELEMETRY"] = "1"


def hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def rewrite(directory, relative, value):
    path = directory / relative
    if relative.endswith("jsonl"):
        path.write_text("".join(json.dumps(row) + "\n" for row in value), encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def completed_pixel_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("pixel-reinfer") / "run"
    summary = run_episode(output)
    assert summary["status"] == "completed"
    return output, summary


@pytest.fixture
def short_run(tmp_path):
    output = tmp_path / "short"
    run_episode(output, RunConfig(max_steps=5))
    return output


def test_pixel_reinference_compares_full_trace_and_terminal_without_rewriting(completed_pixel_run):
    output, summary = completed_pixel_run
    before = hashes(output)
    assert verify_run(output)["perception_reproduced"] is False
    verified = verify_perception(output)
    assert verified["perception_reproduced"] and verified["policy_reproduced"]
    assert verified["frames_reinferred"] == summary["steps"] + 1
    assert verified["terminal_observation_compared"]
    assert verified["terminal_decision_compared"] and verified["terminal_guard_compared"]
    assert verified["model_hash_verified"] is False
    assert hashes(output) == before


@pytest.mark.parametrize(
    "fault", ["brief_dropout", "duplicate_frames", "persistent_dropout", "freeze_input"]
)
def test_reinference_reproduces_fault_supervision_and_termination(tmp_path, fault):
    output = tmp_path / fault
    summary = run_episode(output, RunConfig(fault=fault))
    verified = verify_perception(output)
    assert verified["recorded_status"] == summary["status"]
    assert verified["trace_steps_compared"] == summary["steps"]
    assert verified["frames_reinferred"] == summary["steps"] + 1
    assert verified["terminal_observation_compared"] and verified["terminal_guard_compared"]
    assert verified["terminal_decision_compared"] is (fault != "persistent_dropout")


@pytest.mark.parametrize("field", ["observation", "guard", "decision", "visual_effect"])
def test_reinference_detects_semantic_trace_tampering_even_with_new_outer_hash(short_run, field):
    rows = [json.loads(line) for line in (short_run / "trace.jsonl").read_text().splitlines()]
    if field == "observation":
        rows[2][field]["hp_ratio"] = 0.125
    elif field in {"guard", "decision"}:
        rows[2][field]["reason"] = "forged-but-actions-unchanged"
    else:
        rows[2][field] = "forged-effect"
    rewrite(short_run, "trace.jsonl", rows)
    assert verify_run(short_run)["verified"]
    with pytest.raises(ValueError, match="perception|policy"):
        verify_perception(short_run)


@pytest.mark.parametrize("field", ["observation", "decision", "guard", "cause"])
def test_terminal_evidence_is_compared_even_without_an_executed_action(
    tmp_path, completed_pixel_run, field
):
    output = tmp_path / "tampered-terminal"
    shutil.copytree(completed_pixel_run[0], output)
    summary = json.loads((output / "summary.json").read_text())
    terminal = summary["perception_evidence"]["terminal"]
    if field == "observation":
        terminal[field]["mp_ratio"] = 0.03125
    elif field in {"decision", "guard"}:
        terminal[field]["reason"] = "forged-final-reason"
    else:
        terminal[field] = "step_budget_exhausted"
    rewrite(output, "summary.json", summary)
    assert verify_run(output)["verified"]
    with pytest.raises(ValueError, match="terminal"):
        verify_perception(output)


def test_changed_step_budget_does_not_accept_a_trace_that_omits_the_next_step(short_run):
    summary = json.loads((short_run / "summary.json").read_text())
    summary["config"]["max_steps"] = 6
    rewrite(short_run, "summary.json", summary)
    assert verify_run(short_run)["verified"]
    with pytest.raises(ValueError, match="skipped trace step"):
        verify_perception(short_run)


def test_skipped_recorded_frame_is_rejected(short_run):
    rows = [json.loads(line) for line in (short_run / "trace.jsonl").read_text().splitlines()]
    del rows[2]
    rewrite(short_run, "trace.jsonl", rows)
    with pytest.raises(ValueError, match="sequence"):
        verify_perception(short_run)


def test_legacy_bundle_remains_action_replayable_but_is_not_full_perception_evidence(short_run):
    summary = json.loads((short_run / "summary.json").read_text())
    del summary["perception_evidence"]
    rewrite(short_run, "summary.json", summary)
    assert verify_run(short_run)["verified"]
    with pytest.raises(ValueError, match="regenerate"):
        verify_perception(short_run)


def test_manifest_replacement_after_action_preflight_is_rejected_before_following_new_paths(
    short_run, monkeypatch
):
    import maple_vision.lab as lab

    outside = short_run.parent / "outside.txt"
    outside.write_bytes(b"outside evidence")
    original_verify = lab.verify_run
    original_read = Path.read_bytes

    def replace_manifest_after_preflight(output):
        result = original_verify(output)
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["files"]["../outside.txt"] = hashlib.sha256(b"outside evidence").hexdigest()
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return result

    def never_read_outside(path):
        if path.resolve() == outside.resolve():
            pytest.fail("unvalidated replacement paths must never be followed")
        return original_read(path)

    monkeypatch.setattr(lab, "verify_run", replace_manifest_after_preflight)
    monkeypatch.setattr(Path, "read_bytes", never_read_outside)
    with pytest.raises(ValueError, match="manifest changed.*preflight"):
        verify_perception(short_run)


def test_action_preflight_stops_before_executing_an_over_budget_row(short_run, monkeypatch):
    from maple_vision.simulation import Simulator

    summary = json.loads((short_run / "summary.json").read_text())
    summary["config"]["max_steps"] = 1
    rewrite(short_run, "summary.json", summary)
    calls = []
    original_step = Simulator.step

    def counted_step(simulator, action):
        calls.append(action)
        return original_step(simulator, action)

    monkeypatch.setattr(Simulator, "step", counted_step)
    with pytest.raises(ValueError, match="step budget"):
        verify_perception(short_run)
    assert len(calls) == 1


def test_measurement_tolerance_does_not_waive_discrete_identity_or_decisions(short_run):
    rows = [json.loads(line) for line in (short_run / "trace.jsonl").read_text().splitlines()]
    rows[1]["observation"]["confidence"] -= 1e-7
    rewrite(short_run, "trace.jsonl", rows)
    assert verify_perception(short_run)["perception_reproduced"]
    rows[1]["observation"]["sequence"] = 1.0
    rewrite(short_run, "trace.jsonl", rows)
    with pytest.raises(ValueError, match="observation mismatch"):
        verify_perception(short_run)


def test_runtime_identity_difference_is_visible_not_falsely_equal(short_run):
    summary = json.loads((short_run / "summary.json").read_text())
    summary["runtime_identity"]["source_sha256"] = "0" * 64
    rewrite(short_run, "summary.json", summary)
    verified = verify_perception(short_run)
    assert verified["perception_reproduced"]
    assert verified["runtime_identity_matches"] is False


def test_embedded_profile_and_sprites_reinfer_after_original_inputs_are_deleted(tmp_path):
    database = tmp_path / "knowledge.sqlite"
    initialize_catalog(database)
    import_catalog(database, Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json")
    profile = load_class_profile(database, "assassin", "fixture-v1")
    source = tmp_path / "player.png"
    Image.new("RGBA", (14, 22), (20, 80, 200, 255)).save(source)
    registry = tmp_path / "registry"
    entry = import_asset(
        registry,
        source,
        entity_id="custom-player",
        game_version="fixture-v1",
        source_uri="urn:test:reinference-player",
        license="CC0-1.0",
        permission="self_authored",
    )
    mapping = tmp_path / "sprites.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic_fixture",
                "classes": {"player": entry.asset_id},
            }
        )
    )
    pack = load_sprite_pack(registry, mapping)
    output = tmp_path / "embedded"
    summary = run_episode(output, RunConfig(max_steps=30), class_profile=profile, sprite_pack=pack)
    assert summary["status"] == "stopped"
    assert json.loads((output / "initial-state.json").read_text())["schema_version"] == 2
    shutil.rmtree(registry)
    database.unlink()
    source.unlink()
    mapping.unlink()
    verified = verify_perception(output)
    assert verified["perception_reproduced"] and verified["policy_reproduced"]
    assert verified["independent_goal_check"] is False


class EmptyDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return ()


def test_unknown_custom_detector_cannot_be_claimed_as_reproduced(tmp_path):
    detector = EmptyDetector()
    output = tmp_path / "unknown-detector"
    run_episode(output, RunConfig(max_steps=3), detector)
    calls = detector.calls
    with pytest.raises(ValueError, match="custom detector identity"):
        verify_perception(output, detector)
    assert detector.calls == calls


def test_suite_uses_supplied_detector_for_normal_fault_and_failure_bundles(tmp_path):
    detector = EmptyDetector()
    output = tmp_path / "model-suite"
    report = run_suite(output, seeds=1, detector=detector)
    assert detector.calls > 0
    assert report["passed"] is False
    assert report["normal_episodes"] == 3 and report["episodes"] == 7
    assert all(row["detector"] == "onnx" for row in report["results"])
    assert all(row["status"] == "stopped" for row in report["results"])
    assert report["perception_settings"]["verifiable_model_binding"] is False
    for job in ("assassin", "dragon_knight", "cleric"):
        saved = json.loads((output / f"failure-{job}-0" / "summary.json").read_text())
        assert saved["detector"] == "onnx" and saved["status"] == "stopped"


def test_soak_uses_detector_and_preserves_failure_with_real_resource_sources(tmp_path):
    detector = EmptyDetector()
    output = tmp_path / "model-soak"
    report = run_soak(output, seconds=1, detector=detector, max_resource_samples=3)
    assert detector.calls > 0
    assert report["passed"] is False
    assert report["24h_wall_soak"] == "not_completed"
    assert report["perception_settings"]["pipeline"] == "onnx-objects+fixture-terrain-hud-v1"
    assert 2 <= report["resources"]["retained_samples"] <= 3
    assert report["resources"]["cpu_source"] == "time.process_time"
    assert report["resources"]["sampling_error"] is None
    saved = json.loads((output / "failure-1" / "summary.json").read_text())
    assert saved["detector"] == "onnx" and saved["status"] == "stopped"


@pytest.mark.parametrize("mode", ["suite", "soak"])
def test_transient_failure_keeps_original_bundle_without_running_detector_again(tmp_path, mode):
    class TransientDetector:
        def __init__(self):
            self.calls = 0
            self.pixel = PixelPerception()

        def detect(self, frame):
            self.calls += 1
            return () if self.calls <= 12 else self.pixel.detect(frame)

    detector = TransientDetector()
    output = tmp_path / mode
    if mode == "suite":
        report = run_suite(output, seeds=1, detector=detector)
        original = report["results"][0]
        failure = output / "failure-assassin-0"
        assert detector.calls == sum(row["steps"] + 1 for row in report["results"])
    else:
        report = run_soak(output, seconds=1, detector=detector)
        original = json.loads((output / "episodes.jsonl").read_text().splitlines()[0])
        failure = output / "failure-1"
        assert detector.calls == 12
    assert report["passed"] is False
    saved = json.loads((failure / "summary.json").read_text())
    assert original["status"] == saved["status"] == "stopped"
    assert original["steps"] == saved["steps"] == 11
    assert original["final_state_sha256"] == saved["final_state_sha256"]
    assert len((failure / "trace.jsonl").read_text().splitlines()) == 11
    assert saved["config"]["record_stride"] == 16
    assert report["episode_capture"]["wall_and_resources_include_capture_io"]
    assert not (output / "active-episode").exists()
    assert verify_run(failure)["verified"]
    before = hashes(failure)
    # The counterexample is real: a later run recovers, but must not replace the failure.
    assert run_episode(detector=detector)["status"] == "completed"
    assert hashes(failure) == before


def test_existing_staging_content_is_never_deleted(tmp_path):
    from maple_vision.lab import _record_episode

    staging = tmp_path / "active-episode"
    staging.mkdir()
    marker = staging / "keep.txt"
    marker.write_text("preexisting user content")
    with pytest.raises(FileExistsError):
        _record_episode(tmp_path, RunConfig(max_steps=1), None, failure_name="failure-1")
    assert marker.read_text() == "preexisting user content"


def test_resource_retention_is_bounded_but_retains_endpoints_and_discarded_peak(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("maple_vision.validation_resources.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "maple_vision.validation_resources.time.process_time", lambda: 10 + clock[0] / 4
    )
    monkeypatch.setattr(
        "maple_vision.validation_resources.read_process_memory",
        lambda: ProcessMemory(999 if clock[0] == 2 else 100, 7777, "test-current", "test-lifetime"),
    )
    monitor = ProcessResourceMonitor(interval_seconds=60, max_samples=3)
    monitor.start()
    for index in range(1, 10):
        clock[0] = float(index)
        monitor.sample()
    report = monitor.close()
    assert report["measurements"] == 11
    assert report["retained_samples"] <= 3
    assert report["discarded_samples"] == 11 - report["retained_samples"]
    assert report["samples"][0]["elapsed_wall_seconds"] == 0
    assert report["samples"][-1]["elapsed_wall_seconds"] == 9
    assert report["samples"][-1]["process_cpu_seconds"] == 2.25
    assert report["max_observed_current_rss_bytes"] == 999
    assert report["max_observed_cpu_percent"] == 25
    assert report["units"]["lifetime_peak_rss_bytes"] != report["units"]["current_rss_bytes"]


def test_unsupported_memory_is_explicitly_unknown(monkeypatch):
    monkeypatch.setattr(
        "maple_vision.validation_resources.sys.platform", "unsupported-test-platform"
    )
    measurement = read_process_memory()
    assert measurement.current_rss_bytes is None
    assert measurement.lifetime_peak_rss_bytes is None
    assert measurement.current_rss_source == "unavailable"
    assert measurement.unavailable_reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_seconds": 0},
        {"interval_seconds": True},
        {"interval_seconds": float("nan")},
        {"interval_seconds": 61},
        {"max_samples": 1},
        {"max_samples": True},
        {"max_samples": 10_001},
    ],
)
def test_resource_limits_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ProcessResourceMonitor(**kwargs)


def test_soak_exception_never_becomes_a_24h_pass_and_still_records_resources(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("maple_vision.lab.time.monotonic", lambda: clock[0])

    def failed_episode(output, config, detector, *, failure_name):
        clock[0] = 86401.0
        raise RuntimeError("inference failure")

    monkeypatch.setattr("maple_vision.lab._record_episode", failed_episode)
    output = tmp_path / "broken-soak"
    with pytest.raises(RuntimeError, match="inference failure"):
        run_soak(output, seconds=1)
    report = json.loads((output / "soak.json").read_text())
    assert report["passed"] is False
    assert report["24h_wall_soak"] == "not_completed"
    assert report["run_error"] == "RuntimeError"
    assert report["resources"]["retained_samples"] >= 2


@pytest.fixture(scope="module")
def actual_model_run(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    directory = Path(__file__).parents[1] / "examples" / "fixture-model"
    output = tmp_path_factory.mktemp("onnx-reinfer") / "run"
    detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")
    summary = run_episode(output, detector=detector)
    assert summary["status"] == "completed"
    return output, summary, directory


def test_actual_shipped_onnx_reinfers_every_frame_and_policy_terminal(
    actual_model_run, monkeypatch
):
    output, summary, directory = actual_model_run
    detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")
    calls = []
    detect = detector.detect

    def counted(frame):
        calls.append((frame.sequence, frame.timestamp_ms))
        return detect(frame)

    monkeypatch.setattr(detector, "detect", counted)
    before = hashes(output)
    verified = verify_perception(output, detector)
    assert verified["model_hash_verified"]
    assert verified["perception_reproduced"] and verified["policy_reproduced"]
    assert calls == [(index, index * 50) for index in range(summary["steps"] + 1)]
    assert verified["terminal_decision_compared"]
    assert verified["perception_settings"]["inference_options"] == {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    }
    assert hashes(output) == before


@pytest.mark.parametrize("mismatch", ["model_hash", "full_manifest", "threshold"])
def test_wrong_model_or_options_are_rejected_before_any_detect(
    actual_model_run, monkeypatch, mismatch
):
    output, _, directory = actual_model_run
    detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")
    if mismatch == "model_hash":
        detector.manifest = replace(detector.manifest, model_sha256="0" * 64)
    elif mismatch == "full_manifest":
        detector.manifest = replace(detector.manifest, scope="authorized_recordings")
    else:
        detector.confidence_threshold = 0.8

    def never_infer(frame):
        pytest.fail("mismatched model/options must fail before inference")

    monkeypatch.setattr(detector, "detect", never_infer)
    with pytest.raises(ValueError, match="model|manifest|options"):
        verify_perception(output, detector)
