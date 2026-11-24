"""Independent source-boundary and replay tests; fake models are not quality evidence."""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple_vision.agent import HybridPerception, Supervisor
from maple_vision.contracts import Detection, Status
from maple_vision.lab import (
    RunConfig,
    run_episode,
    run_soak,
    run_suite,
    verify_perception,
    verify_run,
)
from maple_vision.onnx_detector import ModelManifest, OnnxDetector
from maple_vision.scene import (
    CLASS_NAMES,
    DIALOGUE_COLOR,
    HP_Y,
    MP_Y,
    PALETTE,
    ImageFrame,
    SceneRenderer,
)
from maple_vision.simulation import Simulator
from maple_vision.vision import PixelPerception

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

PLAYER = Detection("player", (24.0, 146.0, 38.0, 168.0), 0.91)
MODEL_SOURCE = "onnx-all-objects+fixture-hud-v1"


class FixedDetector:
    def __init__(self, detections):
        self.detections = detections
        self.frames = []

    def detect(self, frame):
        assert isinstance(frame, ImageFrame)
        self.frames.append((frame.sequence, frame.timestamp_ms))
        return self.detections


@pytest.fixture
def frame():
    # Truth constructs the test image only. The adapter receives this RGB value.
    return SceneRenderer().render(Simulator().snapshot())


def pixels(frame):
    return np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()


def hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def rewrite_summary(directory, summary):
    path = directory / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_hidden_terrain_comes_only_from_model_without_running_palette_detection(frame, monkeypatch):
    rgb = pixels(frame)
    for category in ("platform", "ladder"):
        rgb[np.all(rgb == PALETTE[category], axis=2)] = (6, 9, 13)
    hidden = replace(frame, pixels=rgb.tobytes())
    assert not any(
        item.class_id in {"platform", "ladder"} for item in PixelPerception().detect(hidden)
    )
    # Deliberately different from authored map coordinates: output must preserve
    # these model measurements, not substitute the simulator's platform geometry.
    supplied = (
        PLAYER,
        Detection("platform", (7.0, 134.0, 305.0, 139.0), 0.72),
        Detection("ladder", (151.0, 82.0, 158.0, 137.0), 0.68),
    )
    detector = FixedDetector(supplied)

    def forbidden_palette(*args):
        pytest.fail("model terrain must not call the pixel object detector")

    monkeypatch.setattr(PixelPerception, "detect", forbidden_palette)
    observation = HybridPerception(detector, terrain_source="model").observe(hidden)
    assert observation.detections == supplied
    assert observation.source_id == MODEL_SOURCE
    assert observation.status is Status.KNOWN
    assert observation.hp_ratio == observation.mp_ratio == 1.0
    assert observation.map_id == "meadow"
    assert detector.frames == [(0, 0)]


def test_model_misses_are_not_repaired_from_visible_terrain_or_dynamic_colors(frame):
    detector = FixedDetector((PLAYER,))
    model = HybridPerception(detector, terrain_source="model").observe(frame)
    fixture = HybridPerception(detector).observe(frame)
    assert model.detections == (PLAYER,)
    assert any(item.class_id == "platform" for item in fixture.detections)
    assert any(item.class_id == "ladder" for item in fixture.detections)
    assert not any(item.class_id == "monster" for item in model.detections)
    assert fixture.source_id == "onnx-objects+fixture-terrain-hud-v1"


@pytest.mark.parametrize("category", CLASS_NAMES)
def test_every_supported_object_class_preserves_model_box_and_confidence(frame, category):
    detection = Detection(category, (11.0, 37.0, 31.0, 48.0), 0.63)
    supplied = (detection,) if category == "player" else (PLAYER, detection)
    actual = HybridPerception(FixedDetector(supplied), terrain_source="model").observe(frame)
    assert actual.detections == supplied


@pytest.mark.parametrize("players", [(), (PLAYER, replace(PLAYER, bbox=(55, 140, 69, 162)))])
def test_missing_or_ambiguous_model_player_is_unknown_even_with_visible_palette(frame, players):
    observation = HybridPerception(FixedDetector(players), terrain_source="model").observe(frame)
    assert observation.status is Status.UNKNOWN
    assert observation.confidence == 0
    assert Supervisor().inspect(observation, frame.timestamp_ms).state == "paused"


@pytest.mark.parametrize("field", ["hp_ratio", "mp_ratio", "map_id"])
def test_unreadable_fixture_hud_field_stays_unknown_in_model_mode(frame, field):
    rgb = pixels(frame)
    if field == "map_id":
        rgb[6:18, frame.width - 20 : frame.width - 8] = (2, 3, 5)
    else:
        rgb[(HP_Y if field == "hp_ratio" else MP_Y) + 2, 13] = (2, 3, 5)
    broken = replace(frame, pixels=rgb.tobytes())
    observation = HybridPerception(FixedDetector((PLAYER,)), terrain_source="model").observe(broken)
    assert getattr(observation, field) is None
    assert observation.status is Status.UNKNOWN
    assert Supervisor().inspect(observation, frame.timestamp_ms).state == "paused"


def test_hud_only_reader_is_explicitly_not_object_or_terrain_evidence(frame, monkeypatch):
    rgb = pixels(frame)
    rgb[34, frame.width // 4 : frame.width * 3 // 4] = DIALOGUE_COLOR
    dialogue = replace(frame, pixels=rgb.tobytes())
    monkeypatch.setattr(PixelPerception, "detect", lambda _: pytest.fail("unexpected detection"))
    result = PixelPerception().observe_hud(dialogue)
    assert result.detections == ()
    assert result.status is Status.UNKNOWN and result.confidence == 0
    assert result.dialogue_visible
    assert result.source_id == "synthetic-fixture-hud-v1"


@pytest.mark.parametrize(
    "supplied,match",
    [
        ([PLAYER], "immutable tuple"),
        ((object(),), "Detection"),
        ((Detection("invented", (1, 1, 8, 8), 0.9),), "class"),
        ((Detection("platform", (0, 168, 321, 172), 0.9),), "input frame"),
        ((Detection("ladder", (0, 180, 8, 193), 0.9),), "input frame"),
    ],
)
def test_invalid_model_output_fails_explicitly_instead_of_falling_back(frame, supplied, match):
    with pytest.raises(ValueError, match=match):
        HybridPerception(FixedDetector(supplied), terrain_source="model").observe(frame)


@pytest.mark.parametrize("runner", [run_episode, run_suite, run_soak, verify_perception])
@pytest.mark.parametrize("selection", ["pixel", "MODEL", "", None, True, []])
def test_unknown_terrain_selection_fails_before_detector_or_output(tmp_path, runner, selection):
    detector = FixedDetector((PLAYER,))
    output = tmp_path / "invalid-selection"
    with pytest.raises(ValueError, match="terrain_source"):
        runner(output, detector=detector, terrain_source=selection)
    assert not output.exists()
    assert detector.frames == []


@pytest.mark.parametrize("runner", [run_episode, run_suite, run_soak, verify_perception])
def test_model_mode_requires_detector_before_creating_output(tmp_path, runner):
    output = tmp_path / "missing-detector"
    with pytest.raises(ValueError, match="requires a detector"):
        runner(output, terrain_source="model")
    assert not output.exists()


@pytest.mark.parametrize("classes", [("player",), (*CLASS_NAMES[:-1], "unknown"), CLASS_NAMES[:-1]])
def test_wrong_declared_model_classes_fail_before_inference_or_output(tmp_path, classes):
    detector = FixedDetector((PLAYER,))
    detector.manifest = SimpleNamespace(class_names=classes)
    output = tmp_path / "wrong-model-schema"
    with pytest.raises(ValueError, match="all seven"):
        run_episode(output, detector=detector, terrain_source="model")
    assert detector.frames == [] and not output.exists()


@pytest.fixture(scope="module")
def native_model(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    directory = tmp_path_factory.mktemp("terrain-native-contract")
    # Constant native model intentionally cannot solve the mission. Its purpose
    # is real inference/evidence checks with known boxes and bounded failure.
    values = np.array(
        [
            [
                [24, 146, 38, 168, 0.95, 0],
                [200, 150, 220, 168, 0.95, 1],
                [0, 168, 320, 172, 0.95, 5],
                [210, 90, 218, 168, 0.95, 6],
            ]
        ],
        dtype=np.float32,
    )
    node = onnx.helper.make_node(
        "Constant", [], ["detections"], value=onnx.numpy_helper.from_array(values)
    )
    graph = onnx.helper.make_graph(
        [node],
        "model-terrain-source-contract",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 192, 320])],
        [onnx.helper.make_tensor_value_info("detections", onnx.TensorProto.FLOAT, [1, 4, 6])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )
    model_path, manifest_path = directory / "model.onnx", directory / "manifest.json"
    onnx.save(model, model_path)
    manifest = ModelManifest(
        hashlib.sha256(model_path.read_bytes()).hexdigest(),
        CLASS_NAMES,
        320,
        192,
        "end2end_xyxy",
        "synthetic_fixture",
    )
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    return model_path, manifest_path


@pytest.mark.parametrize("recorded_source", ["fixture", "model"])
def test_reinference_rejects_wrong_terrain_source_before_any_detect(
    tmp_path, native_model, recorded_source, monkeypatch
):
    detector = OnnxDetector(*native_model)
    output = tmp_path / "run"
    run_episode(output, RunConfig(max_steps=5), detector, terrain_source=recorded_source)
    before = hashes(output)

    def forbidden(frame):
        pytest.fail("mode mismatch must be rejected before detect")

    monkeypatch.setattr(detector, "detect", forbidden)
    opposite = "model" if recorded_source == "fixture" else "fixture"
    with pytest.raises(ValueError, match="terrain mode"):
        verify_perception(output, detector, terrain_source=opposite)
    assert hashes(output) == before


def test_default_v02_identity_remains_reinferable_without_a_new_mode_field(tmp_path, native_model):
    detector = OnnxDetector(*native_model)
    output = tmp_path / "legacy-settings"
    summary = run_episode(output, RunConfig(max_steps=5), detector)
    settings = summary["perception_evidence"]["settings"]
    assert settings == {
        "pipeline": "onnx-objects+fixture-terrain-hud-v1",
        "implementation": "maple_vision.onnx_detector.OnnxDetector",
        "model_manifest": asdict(detector.manifest),
        "loaded_model_sha256": detector.loaded_model_sha256,
        "inference_options": {"confidence_threshold": 0.25, "iou_threshold": 0.45},
        "verifiable_model_binding": True,
    }
    assert verify_perception(output, detector)["perception_reproduced"]
    assert verify_run(output)["perception_reproduced"] is False


@pytest.mark.parametrize(
    "fault", ["none", "brief_dropout", "persistent_dropout", "duplicate_frames", "freeze_input"]
)
def test_actual_model_terrain_reinference_preserves_original_fault_and_terminal_evidence(
    tmp_path, native_model, fault, monkeypatch
):
    detector = OnnxDetector(*native_model)
    output = tmp_path / fault
    native_detect = detector.detect
    calls = []

    def counted(frame):
        calls.append((frame.sequence, frame.timestamp_ms))
        return native_detect(frame)

    monkeypatch.setattr(detector, "detect", counted)
    summary = run_episode(
        output,
        RunConfig(max_steps=100, fault=fault, record_stride=16),
        detector,
        terrain_source="model",
    )
    original_calls = calls.copy()
    calls.clear()
    before = hashes(output)
    replay = verify_perception(output, detector, terrain_source="model")
    assert replay["perception_reproduced"] and replay["policy_reproduced"]
    assert replay["frames_reinferred"] == summary["steps"] + 1 == len(calls)
    assert calls == original_calls
    assert replay["terminal_observation_compared"]
    assert replay["terminal_guard_compared"]
    assert replay["terminal_decision_compared"] is (fault != "persistent_dropout")
    assert replay["recorded_status"] == summary["status"]
    assert not replay["independent_goal_check"]
    assert summary["status"] == ("stopped" if fault == "persistent_dropout" else "failed")
    assert hashes(output) == before
    settings = summary["perception_evidence"]["settings"]
    assert settings["pipeline"] == MODEL_SOURCE and settings["terrain_source"] == "model"
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert all(row["observation"]["source_id"] == MODEL_SOURCE for row in rows)
    if fault in {"brief_dropout", "persistent_dropout"}:
        assert rows[20]["observation"]["hp_ratio"] is None
        assert rows[20]["guard"]["state"] == "paused"
    if fault == "duplicate_frames":
        assert [row["observation"]["sequence"] for row in rows[20:26]] == [19] * 6
    if fault == "freeze_input":
        assert rows[20]["execution"]["applied_action"]["kind"] == "idle"


def test_model_terminal_observation_is_compared_after_budget_without_fake_policy_call(
    tmp_path, native_model
):
    detector = OnnxDetector(*native_model)
    output = tmp_path / "terminal-budget"
    summary = run_episode(output, RunConfig(max_steps=5), detector, terrain_source="model")
    report = verify_perception(output, detector, terrain_source="model")
    assert report["frames_reinferred"] == 6
    assert report["terminal_observation_compared"]
    assert not report["terminal_guard_compared"] and not report["terminal_decision_compared"]
    terminal = summary["perception_evidence"]["terminal"]
    terrain = next(d for d in terminal["observation"]["detections"] if d["class_id"] == "platform")
    terrain["bbox"] = (0, 167, 320, 172)
    rewrite_summary(output, summary)
    assert verify_run(output)["verified"]
    with pytest.raises(ValueError, match="terminal.observation"):
        verify_perception(output, detector, terrain_source="model")


def test_missing_explicit_model_mode_identity_cannot_be_silently_inferred(tmp_path, native_model):
    detector = OnnxDetector(*native_model)
    output = tmp_path / "mode-removed"
    summary = run_episode(output, RunConfig(max_steps=1), detector, terrain_source="model")
    del summary["perception_evidence"]["settings"]["terrain_source"]
    rewrite_summary(output, summary)
    with pytest.raises(ValueError, match="terrain mode"):
        verify_perception(output, detector, terrain_source="model")


@pytest.mark.parametrize("runner", [run_suite, run_soak])
def test_suite_and_soak_use_explicit_model_terrain_for_every_original_failure(tmp_path, runner):
    detector = FixedDetector(())
    output = tmp_path / "validation"
    options = {"seeds": 1} if runner is run_suite else {"seconds": 1}
    report = runner(output, detector=detector, terrain_source="model", **options)
    assert not report["passed"]
    assert report["perception_settings"]["terrain_source"] == "model"
    assert report["perception_settings"]["pipeline"] == MODEL_SOURCE
    saved = sorted(output.rglob("summary.json"))
    assert saved
    steps = 0
    for path in saved:
        summary = json.loads(path.read_text())
        assert summary["perception_evidence"]["settings"]["terrain_source"] == "model"
        assert (
            summary["perception_evidence"]["terminal"]["observation"]["source_id"] == MODEL_SOURCE
        )
        assert summary["status"] == "stopped"
        steps += summary["steps"] + 1
    assert len(detector.frames) == steps, "retaining failures must not rerun the model"


def test_real_shipped_model_can_reinfer_opt_in_mode_without_quality_promotion(tmp_path):
    pytest.importorskip("onnxruntime")
    directory = Path(__file__).parents[1] / "examples" / "lifecycle-model"
    detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")
    output = tmp_path / "actual-model"
    summary = run_episode(output, RunConfig(max_steps=8), detector, terrain_source="model")
    replay = verify_perception(output, detector, terrain_source="model")
    assert replay["frames_reinferred"] == 9
    assert replay["model_hash_verified"] and replay["files_unchanged"]
    assert summary["status"] == "failed" and not replay["independent_goal_check"]
