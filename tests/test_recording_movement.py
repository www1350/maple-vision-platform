"""Imported authored pixels exercise the optional movement replay end to end.

The detector reads colored regions without frame-number tables or state labels.
These fixtures validate engineering behavior, not real-game detection accuracy.
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from maple_vision.contracts import Detection
from maple_vision.movement_hypotheses import MOVEMENT_ALGORITHM_VERSION, MovementConfig
from maple_vision.recording_tracking import track_recording, verify_recording_tracking
from maple_vision.recordings import import_recording

COLORS = {"player": (240, 40, 220), "rope": (220, 180, 30), "platform": (30, 180, 230)}


class GeometryMarkerDetector:
    """Visible color bounds bridge a small foreground occlusion of authored terrain."""

    def detect(self, frame):
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        detections = []
        for kind, color in COLORS.items():
            mask = np.all(rgb == color, axis=2)
            rows = np.flatnonzero(np.any(mask, axis=1))
            if not len(rows):
                continue
            groups = [rows]
            if kind == "platform":
                groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
            for group in groups:
                ys, xs = np.nonzero(mask[group[0] : group[-1] + 1])
                detections.append(
                    Detection(
                        kind,
                        (int(xs.min()), int(group[0]), int(xs.max()) + 1, int(group[-1]) + 1),
                        0.95,
                    )
                )
        return tuple(detections)


def authored_recording(tmp_path, *, camera_follows=False):
    source = tmp_path / "source"
    source.mkdir()
    feet = [130 - 4 * step for step in range(21)] + [50] * 4
    positions = [130] * 17 + feet
    for sequence, bottom in enumerate(positions):
        camera_y = 130 - bottom if camera_follows and sequence >= 17 else 0
        image = Image.new("RGB", (320, 256), (12, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((78, 30 + camera_y, 81, 149 + camera_y), fill=COLORS["rope"])
        draw.rectangle((50, 50 + camera_y, 99, 54 + camera_y), fill=COLORS["platform"])
        draw.rectangle((160, 130 + camera_y, 199, 134 + camera_y), fill=COLORS["platform"])
        draw.rectangle(
            (75, bottom - 16 + camera_y, 84, bottom - 1 + camera_y), fill=COLORS["player"]
        )
        image.save(source / f"{sequence:04d}.png")
        image.close()
    recording = tmp_path / "recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:authored-movement-pixels",
        permission="self_authored",
        license="CC0",
        fps=20,
        start_frame=17,
        max_frames=len(feet),
    )
    return recording


def trace(output):
    return [json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("camera_follows", [False, True])
def test_movement_replay_processes_every_source_frame_and_recomputes_candidates(
    tmp_path, camera_follows
):
    recording = authored_recording(tmp_path, camera_follows=camera_follows)
    output = tmp_path / "movement"
    report = track_recording(
        recording,
        output,
        GeometryMarkerDetector(),
        movement_config=MovementConfig(),
        preview_max_frames=3,
    )
    rows = trace(output)
    assert len(rows) == report["movement_analysis"]["frames_processed"] == 25
    assert report["preview_frames"] <= 3
    assert report["movement_analysis"]["algorithm_version"] == MOVEMENT_ALGORITHM_VERSION
    assert sum(report["movement_analysis"]["state_counts"].values()) == len(rows)
    assert report["preview"]["renderer"] == "recording_movement_overlay_v1"
    assert "climbing_candidate" in {row["movement"]["state"] for row in rows}
    assert rows[-1]["movement"]["state"] == "platform_arrival_candidate"
    for index, row in enumerate(rows):
        movement = row["movement"]
        assert movement["source_sequence"] == row["source_sequence"] == index + 17
        assert movement["timestamp_ms"] == row["timestamp_ms"] == index * 50
        assert movement["source_id"] == row["belief"]["observation"]["source_id"]
        assert movement["actionable"] is False
        assert row["belief"]["actionable"] is False
        assert movement["action_execution"] == "none"
    climbing = next(
        row["movement"] for row in rows if row["movement"]["state"] == "climbing_candidate"
    )
    if camera_follows:
        assert climbing["camera"]["delta_px"] == [0, 4]
        assert climbing["relative_motion"]["actor_screen_delta_px"] == [0, 0]
        assert climbing["relative_motion"]["actor_camera_relative_delta_px"] == [0, -4]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    verified = verify_recording_tracking(recording, output, GeometryMarkerDetector())
    assert verified["verified"] is True
    assert verified["movement_recomputed"] is True
    assert verified["movement_state_counts"] == report["movement_analysis"]["state_counts"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_default_tracking_keeps_movement_absent_and_distinct_window_identity(tmp_path):
    recording = authored_recording(tmp_path)
    plain = tmp_path / "plain"
    optional = tmp_path / "optional"
    original = track_recording(recording, plain, GeometryMarkerDetector(), preview_max_frames=2)
    extended = track_recording(
        recording,
        optional,
        GeometryMarkerDetector(),
        movement_config=MovementConfig(),
        preview_max_frames=2,
    )
    assert "movement_analysis" not in original
    assert original["preview"]["renderer"] == "recording_tracking_overlay_v1"
    assert all("movement" not in row for row in trace(plain))
    assert original["tracking_window_id"] != extended["tracking_window_id"]
    verified = verify_recording_tracking(recording, plain, GeometryMarkerDetector())
    assert "movement_recomputed" not in verified


def test_forged_movement_state_rejected_even_after_trace_hashes_are_refreshed(tmp_path):
    recording = authored_recording(tmp_path)
    output = tmp_path / "movement"
    track_recording(recording, output, GeometryMarkerDetector(), movement_config=MovementConfig())
    rows = trace(output)
    rows[-1]["movement"]["state"] = "unknown"
    encoded = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    # Preserve the exact hashed bytes across platforms before checking semantic tampering.
    (output / "tracks.jsonl").write_bytes(encoded.encode("utf-8"))
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    report["tracks_sha256"] = report["files"]["tracks.jsonl"] = digest
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory mismatch"):
        verify_recording_tracking(recording, output, GeometryMarkerDetector())


@pytest.mark.parametrize("mutation", ["config", "algorithm", "counts", "remove"])
def test_verification_binds_movement_configuration_algorithm_and_summary(tmp_path, mutation):
    recording = authored_recording(tmp_path)
    output = tmp_path / "movement"
    track_recording(recording, output, GeometryMarkerDetector(), movement_config=MovementConfig())
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    analysis = report["movement_analysis"]
    if mutation == "config":
        analysis["config"]["max_climb_to_arrival_ms"] += 1
    elif mutation == "algorithm":
        analysis["algorithm_version"] = "unrecognized-algorithm"
    elif mutation == "counts":
        analysis["state_counts"] = {"unknown": report["frames_processed"]}
    else:
        del report["movement_analysis"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch|recomputed evidence"):
        verify_recording_tracking(recording, output, GeometryMarkerDetector())
