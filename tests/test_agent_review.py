"""Independent adversarial cases for observation ordering and model provenance."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from maple_vision.agent import Supervisor, verify_visual_effect
from maple_vision.contracts import Detection, Status
from maple_vision.onnx_detector import ModelManifest, OnnxDetector
from maple_vision.sim_types import SimAction
from maple_vision.vision import VisualObservation


def observation(sequence=0, timestamp_ms=0):
    return VisualObservation(
        sequence,
        timestamp_ms,
        Status.KNOWN,
        0.95,
        (Detection("player", (20.0, 146.0, 34.0, 168.0), 0.95),),
        1.0,
        1.0,
        "meadow",
    )


def test_review_capture_clock_rollback_is_not_a_fresh_observation():
    guard = Supervisor()
    assert guard.inspect(observation(1, 100), 100).state == "running"
    decision = guard.inspect(observation(2, 90), 150)
    assert decision.state != "running", "capture time rollback must not become trusted state"


def test_review_future_frame_cannot_poison_recovery_sequence():
    guard = Supervisor(max_unknown_steps=3)
    assert guard.inspect(observation(0, 0), 0).state == "running"
    assert guard.inspect(observation(100_000, 10_000), 50).state == "paused"
    assert guard.inspect(observation(2, 100), 100).state == "running", (
        "reject the malformed frame without latching its impossible sequence"
    )


def test_review_bad_observations_have_bounded_stop_and_do_not_execute():
    guard = Supervisor(max_unknown_steps=2)
    assert guard.inspect(observation(0, 0), 0).state == "running"
    assert guard.inspect(observation(0, 0), 50).state == "paused"
    assert guard.inspect(observation(0, 0), 100).state == "stopped"
    assert guard.inspect(observation(3, 150), 150).state == "stopped"


@pytest.mark.parametrize("sequence,timestamp", [(4, 100), (3, 100), (5, 99)])
def test_review_old_or_duplicate_frames_cannot_verify_a_map_change(sequence, timestamp):
    before = observation(4, 100)
    after = replace(before, sequence=sequence, timestamp_ms=timestamp, map_id="cavern")
    assert verify_visual_effect(before, after, SimAction("portal")) != "observed"


def test_review_missing_map_reading_is_not_an_observed_portal_effect():
    before = replace(observation(0, 0), map_id=None)
    after = replace(observation(1, 50), map_id="cavern")
    assert verify_visual_effect(before, after, SimAction("portal")) == "unknown"


def test_review_existing_dialogue_is_not_a_new_interaction_effect():
    before = replace(observation(0, 0), dialogue_visible=True)
    after = replace(observation(1, 50), dialogue_visible=True)
    assert verify_visual_effect(before, after, SimAction("interact")) != "observed"


def manifest_payload(payload):
    return {
        "schema_version": 1,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "class_names": ["player"],
        "input_width": 32,
        "input_height": 32,
        "output_format": "raw_xywh",
        "scope": "synthetic_fixture",
    }


def test_review_manifest_schema_version_is_an_integer(tmp_path):
    manifest = tmp_path / "manifest.json"
    value = manifest_payload(b"fixture")
    value["schema_version"] = 1.0
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        ModelManifest.load(manifest)


def test_review_session_loads_the_exact_bytes_that_passed_the_checksum(tmp_path, monkeypatch):
    """Changing a model after verification must not silently switch inference artifacts."""
    original = b"verified fixture model bytes"
    model = tmp_path / "model.onnx"
    model.write_bytes(original)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload(original)), encoding="utf-8")
    loaded = []

    def options():
        # A concurrent writer replaces the pathname after checksum verification.
        model.write_bytes(b"different model artifact")
        return SimpleNamespace(intra_op_num_threads=0)

    def create_session(content, **kwargs):
        loaded.append(content if isinstance(content, bytes) else Path(content).read_bytes())
        return SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(type="tensor(float)", shape=[1, 3, 32, 32], name="x")
            ],
            get_outputs=lambda: [SimpleNamespace()],
        )

    fake_runtime = SimpleNamespace(
        SessionOptions=options,
        InferenceSession=create_session,
        disable_telemetry_events=lambda: None,
    )
    monkeypatch.setattr(
        "maple_vision.onnx_detector.importlib.import_module", lambda name: fake_runtime
    )
    OnnxDetector(model, manifest)
    assert loaded == [original], "hash verification must bind to the bytes actually loaded"
