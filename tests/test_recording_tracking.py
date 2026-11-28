"""Independent authored-pixel checks for read-only temporal recording replay.

The marker detector reads connected colored regions in each supplied RGB image.
It has no trajectory table, expected track IDs, simulator state, or frame-number
dependent predictions. These fixtures test engineering behavior, not game accuracy.
"""

import hashlib
import json
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageSequence

from maple_vision.contracts import Detection
from maple_vision.onnx_detector import ModelManifest, OnnxDetector
from maple_vision.recording_tracking import track_recording, verify_recording_tracking
from maple_vision.recordings import import_recording, validate_recording
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator
from maple_vision.state_estimation import StateConfig


class MarkerDetector:
    """Tiny independent pixel detector; connected magenta areas are player boxes."""

    def __init__(self):
        self.seen = []

    def detect(self, frame):
        self.seen.append((frame.sequence, frame.timestamp_ms))
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        mask = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] > 150)
        points = {(int(x), int(y)) for y, x in np.argwhere(mask)}
        detections = []
        while points:
            first = points.pop()
            component, pending = [first], [first]
            while pending:
                x, y = pending.pop()
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbor in points:
                        points.remove(neighbor)
                        component.append(neighbor)
                        pending.append(neighbor)
            xs, ys = zip(*component, strict=True)
            detections.append(
                Detection("player", (min(xs), min(ys), max(xs) + 1, max(ys) + 1), 0.95)
            )
        return tuple(sorted(detections, key=lambda detection: detection.bbox))


def authored_recording(
    root, positions, *, name="recording", fps=20, start_frame=0, max_frames=10000
):
    source = root / (name + "-source")
    source.mkdir()
    for sequence, position in enumerate(positions):
        image = Image.new("RGB", (160, 96), (12, 18, 24))
        draw = ImageDraw.Draw(image)
        positions_in_frame = () if position is None else (position,)
        if isinstance(position, tuple):
            positions_in_frame = position
        for x in positions_in_frame:
            draw.rectangle((x, 30, x + 9, 45), fill=(240, 30, 220))
        image.save(source / f"{sequence:04d}.png")
    recording = root / name
    import_recording(
        source,
        recording,
        source_uri="urn:self:independently-authored-marker:" + name,
        permission="self_authored",
        license="CC0",
        fps=fps,
        start_frame=start_frame,
        max_frames=max_frames,
    )
    return recording


def trace(replay):
    return [json.loads(line) for line in (replay / "tracks.jsonl").read_text().splitlines()]


def field(row, name):
    return next(item for item in row["belief"]["fields"] if item["name"] == name)


def player_tracks(row, *, observed_only=False):
    return [
        item
        for item in row["belief"]["tracks"]
        if item["class_id"] == "player" and (not observed_only or item["observed_this_frame"])
    ]


def rewrite_report(replay, change):
    path = replay / "report.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")


def refresh_file_hash(replay, name, report_key):
    digest = hashlib.sha256((replay / name).read_bytes()).hexdigest()

    def change(report):
        report[report_key] = digest
        report["files"][name] = digest

    rewrite_report(replay, change)


def test_pixel_motion_preserves_identity_and_measured_screen_velocity(tmp_path):
    recording = authored_recording(tmp_path, [12, 17, 22, 27])
    source_before = {
        path.relative_to(recording): path.read_bytes()
        for path in recording.rglob("*")
        if path.is_file()
    }
    replay = tmp_path / "replay"
    detector = MarkerDetector()
    report = track_recording(recording, replay, detector)
    rows = trace(replay)

    assert detector.seen == [(0, 0), (1, 50), (2, 100), (3, 150)]
    assert report["frames_processed"] == len(rows) == 4
    assert report["perception_mode"] == "detector_only"
    assert report["identity_scope"] == "window_local_only"
    tracks = [player_tracks(row, observed_only=True)[0] for row in rows]
    assert len({item["track_id"] for item in tracks}) == 1
    assert [item["association"] for item in tracks] == ["new", "matched", "matched", "matched"]
    assert [item["bbox"] for item in tracks] == [
        [12, 30, 22, 46],
        [17, 30, 27, 46],
        [22, 30, 32, 46],
        [27, 30, 37, 46],
    ]
    assert tracks[0]["velocity_px_per_second"] is None
    assert all(item["velocity_px_per_second"] == [100.0, 0.0] for item in tracks[1:])
    assert all(not item["identity_uncertain"] for item in tracks)
    for row in rows:
        belief = row["belief"]
        assert belief["actionable"] is False
        assert belief["observation"]["status"] == "unknown"
        assert belief["last_accepted_sequence"] is None
        for name in ("map_id", "hp_ratio", "mp_ratio", "inventory", "skills", "buffs"):
            assert field(row, name)["value"] is None
            assert field(row, name)["status"] == "unknown"
        assert belief["observation"]["map_id"] is None
        assert belief["observation"]["hp_ratio"] is None
        assert belief["observation"]["mp_ratio"] is None
    replay_before = {path.name: path.read_bytes() for path in replay.iterdir()}
    verified = verify_recording_tracking(recording, replay, MarkerDetector())
    assert verified["verified"] is True
    assert verified["trajectory_recomputed"] is True
    assert verified["preview_verified"] is True
    assert {path.name: path.read_bytes() for path in replay.iterdir()} == replay_before
    assert {
        path.relative_to(recording): path.read_bytes()
        for path in recording.rglob("*")
        if path.is_file()
    } == source_before


def test_missing_pixels_retain_a_hypothesis_then_reacquire_or_expire(tmp_path):
    recording = authored_recording(tmp_path, [10, 15, None, 25, None, None, None, 45], fps=10)
    replay = tmp_path / "replay"
    config = StateConfig(max_age_ms=50, field_ttl_ms=500, track_ttl_ms=250)
    track_recording(recording, replay, MarkerDetector(), state_config=config)
    rows = trace(replay)
    original = player_tracks(rows[0])[0]["track_id"]

    lost = player_tracks(rows[2])[0]
    assert lost["track_id"] == original
    assert lost["association"] == "unobserved"
    assert lost["observed_this_frame"] is False
    assert lost["bbox"] == [15, 30, 25, 46]
    assert lost["observed_at_ms"] == 100
    assert lost["expires_at_ms"] == 350
    assert rows[2]["belief"]["observation"]["detections"] == []
    assert field(rows[2], "player_bbox")["observed_this_frame"] is False

    returned = player_tracks(rows[3], observed_only=True)[0]
    assert returned["track_id"] == original
    assert returned["association"] == "reacquired"
    assert returned["velocity_px_per_second"] == [50.0, 0.0]
    assert player_tracks(rows[5])[0]["expires_at_ms"] == 550
    assert player_tracks(rows[6]) == []
    restarted = player_tracks(rows[7], observed_only=True)[0]
    assert restarted["track_id"] != original
    assert restarted["association"] == "new"
    assert restarted["velocity_px_per_second"] is None


def test_multiple_visible_players_do_not_arbitrarily_refresh_one_player(tmp_path):
    recording = authored_recording(tmp_path, [20, (23, 75), 26])
    replay = tmp_path / "replay"
    track_recording(recording, replay, MarkerDetector())
    first, ambiguous, recovered = trace(replay)

    assert len(ambiguous["belief"]["observation"]["detections"]) == 2
    assert ambiguous["belief"]["actionable"] is False
    assert player_tracks(ambiguous, observed_only=True) == []
    assert field(ambiguous, "player_bbox")["value"] == [20, 30, 30, 46]
    assert field(ambiguous, "player_bbox")["observed_this_frame"] is False
    original = player_tracks(first)[0]["track_id"]
    assert player_tracks(ambiguous)[0]["track_id"] == original
    assert player_tracks(ambiguous)[0]["association"] == "unobserved"
    assert player_tracks(recovered)[0]["track_id"] == original
    assert player_tracks(recovered)[0]["association"] == "reacquired"
    assert player_tracks(recovered)[0]["velocity_px_per_second"] == [60.0, 0.0]


def test_preview_sampling_keeps_endpoints_without_skipping_temporal_updates(tmp_path):
    recording = authored_recording(tmp_path, [10 + index * 2 for index in range(17)])
    replay = tmp_path / "replay"
    detector = MarkerDetector()
    report = track_recording(recording, replay, detector, preview_stride=4, preview_max_frames=4)
    rows = trace(replay)

    assert len(detector.seen) == len(rows) == report["frames_processed"] == 17
    assert [row["sequence"] for row in rows] == list(range(17))
    assert [row["belief"]["now_ms"] for row in rows] == [index * 50 for index in range(17)]
    assert all(player_tracks(row)[0]["association"] == "matched" for row in rows[1:])
    assert all(player_tracks(row)[0]["velocity_px_per_second"] == [40.0, 0.0] for row in rows[1:])
    selected = report["preview"]["sequences"]
    assert selected == sorted(set(selected))
    assert selected[0] == 0 and selected[-1] == 16
    assert 2 <= len(selected) == report["preview_frames"] <= 4
    assert set(selected).issubset(range(17))
    with Image.open(replay / "preview.gif") as preview:
        assert preview.format == "GIF"
        assert preview.n_frames == len(selected)
        assert preview.size == (report["preview"]["width"], report["preview"]["height"])
        for image in ImageSequence.Iterator(preview):
            image.load()
    assert verify_recording_tracking(recording, replay, MarkerDetector())["verified"] is True


def test_source_windows_keep_original_offsets_and_start_fresh_tracks(tmp_path):
    positions = [10, 15, 20, 25, 30, 35]
    first = authored_recording(tmp_path, positions, name="first", max_frames=2)
    second = authored_recording(tmp_path, positions, name="second", start_frame=2, max_frames=4)
    detector = MarkerDetector()
    one, two = tmp_path / "one", tmp_path / "two"
    report_one = track_recording(first, one, detector)
    report_two = track_recording(second, two, detector)
    rows = trace(two)

    assert [row["sequence"] for row in rows] == [0, 1, 2, 3]
    assert [row["source_sequence"] for row in rows] == [2, 3, 4, 5]
    assert [row["source_start_frame"] for row in rows] == [2] * 4
    assert [row["timestamp_ms"] for row in rows] == [0, 50, 100, 150]
    assert [row["source_timestamp_ms"] for row in rows] == [100, 150, 200, 250]
    assert [row["belief"]["observation"]["sequence"] for row in rows] == [2, 3, 4, 5]
    assert [row["source_name"] for row in rows] == ["0002.png", "0003.png", "0004.png", "0005.png"]
    assert player_tracks(rows[0])[0]["association"] == "new"
    assert player_tracks(rows[0])[0]["velocity_px_per_second"] is None
    assert report_one["tracking_window_id"] != report_two["tracking_window_id"]
    assert report_two["identity_scope"] == "window_local_only"
    assert verify_recording_tracking(second, two, MarkerDetector())["verified"] is True


def test_variable_pts_window_preserves_original_clock_and_local_elapsed_time(tmp_path):
    av = pytest.importorskip("av")
    source = tmp_path / "variable.mp4"
    clock = Fraction(1, 1000)
    with av.open(str(source), "w", format="mp4") as writer:
        stream = writer.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = clock
        for pts, shade in [(140, 20), (180, 75), (300, 130), (520, 200)]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (shade,) * 3))
            frame.pts, frame.time_base = pts, clock
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    recording = tmp_path / "recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:independent-variable-clock",
        permission="self_authored",
        license="CC0",
        presentation_time=True,
        start_frame=1,
        max_frames=3,
    )
    replay = tmp_path / "replay"
    report = track_recording(recording, replay, MarkerDetector())
    rows = trace(replay)

    assert [row["source_sequence"] for row in rows] == [1, 2, 3]
    assert [row["timestamp_ms"] for row in rows] == [0, 120, 340]
    assert [row["source_timestamp_ms"] for row in rows] == [180, 300, 520]
    assert [
        row["pts"] * Fraction(row["video_clock"]["numerator"], row["video_clock"]["denominator"])
        for row in rows
    ] == [Fraction(9, 50), Fraction(3, 10), Fraction(13, 25)]
    assert [row["belief"]["now_ms"] for row in rows] == [0, 120, 340]
    assert report["frames_processed"] == 3
    assert verify_recording_tracking(recording, replay, MarkerDetector())["verified"] is True


def test_fixture_perception_is_explicit_and_replay_has_no_control_trace(tmp_path):
    source = tmp_path / "fixture-source"
    source.mkdir()
    simulator, renderer = Simulator(seed=23), SceneRenderer()
    for index in range(2):
        (source / f"{index}.png").write_bytes(renderer.render(simulator.snapshot()).png_bytes())
        simulator.step(SimAction("move", direction=1))
    recording = tmp_path / "recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:fixture-only",
        permission="self_authored",
        license="CC0",
        fps=20,
    )
    replay = tmp_path / "replay"
    report = track_recording(recording, replay)
    assert report["perception_mode"] == "synthetic_fixture"
    assert report["action_execution"] == "none"
    assert report["accuracy_metrics_available"] is False
    assert report["effect_success_rate"] is None
    assert {path.name for path in replay.iterdir()} == {
        "tracks.jsonl",
        "report.json",
        "preview.gif",
    }
    forbidden = {"intent", "input", "controller", "execution", "action_trace", "task_trace"}

    def inspect(value):
        if isinstance(value, dict):
            assert not (forbidden & set(value))
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for row in trace(replay):
        inspect(row)
    assert verify_recording_tracking(recording, replay)["verified"] is True


@pytest.mark.parametrize(
    "options",
    [
        {"preview_stride": 0},
        {"preview_stride": True},
        {"preview_stride": 10001},
        {"preview_max_frames": 1},
        {"preview_max_frames": 121},
        {"preview_max_frames": False},
        {"state_config": {}},
    ],
)
def test_invalid_options_fail_without_predictions_or_partial_publication(tmp_path, options):
    recording = authored_recording(tmp_path, [10, 15])
    detector = MarkerDetector()
    with pytest.raises(ValueError):
        track_recording(recording, tmp_path / "rejected", detector, **options)
    assert detector.seen == []
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".rejected-*"))


@pytest.mark.parametrize("fault", ["mutable", "outside", "exception"])
def test_bad_later_detector_result_leaves_no_partial_bundle(tmp_path, fault):
    class BadLaterDetector(MarkerDetector):
        def detect(self, frame):
            detections = super().detect(frame)
            if detections[0].bbox[0] >= 20:
                if fault == "mutable":
                    return list(detections)
                if fault == "outside":
                    return (Detection("player", (0, 0, frame.width + 1, 10), 0.9),)
                raise ValueError("intentional detector failure on a later pixel image")
            return detections

    recording = authored_recording(tmp_path, [10, 20])
    replay = tmp_path / "rejected"
    detector = BadLaterDetector()
    with pytest.raises(ValueError):
        track_recording(recording, replay, detector)
    assert len(detector.seen) == 2
    assert not replay.exists()
    assert not list(tmp_path.glob(".rejected-*"))


def test_detection_and_trace_budgets_fail_without_partial_publication(tmp_path, monkeypatch):
    import maple_vision.recording_tracking as module

    recording = authored_recording(tmp_path, [10, (10, 40, 70)])
    config = StateConfig(max_detections=2, max_tracks=2)
    with pytest.raises(ValueError, match="budget"):
        track_recording(recording, tmp_path / "too-many", MarkerDetector(), state_config=config)
    assert not (tmp_path / "too-many").exists()
    assert not list(tmp_path.glob(".too-many-*"))
    monkeypatch.setattr(module, "MAX_TRACK_BYTES", 64)
    with pytest.raises(ValueError):
        track_recording(recording, tmp_path / "too-large", MarkerDetector())
    assert not (tmp_path / "too-large").exists()
    assert not list(tmp_path.glob(".too-large-*"))


def test_output_never_overwrites_or_writes_inside_source(tmp_path):
    recording = authored_recording(tmp_path, [10, 15])
    protected = tmp_path / "existing"
    protected.mkdir()
    marker = protected / "keep.txt"
    marker.write_text("keep this exact content")
    detector = MarkerDetector()
    with pytest.raises(FileExistsError):
        track_recording(recording, protected, detector)
    assert marker.read_text() == "keep this exact content"
    with pytest.raises(ValueError, match="outside"):
        track_recording(recording, recording / "nested", detector)
    assert not (recording / "nested").exists()
    assert detector.seen == []


def test_verification_detects_source_pixels_changed_after_tracking(tmp_path):
    recording = authored_recording(tmp_path, [10, 15])
    replay = tmp_path / "replay"
    track_recording(recording, replay, MarkerDetector())
    image_path = recording / "frames" / "000001.png"
    with Image.open(image_path) as original:
        changed = original.convert("RGB")
    changed.putpixel((15, 30), (0, 0, 0))
    changed.save(image_path)
    with pytest.raises(ValueError):
        verify_recording_tracking(recording, replay, MarkerDetector())


@pytest.mark.parametrize("operation", ["track", "verify"])
@pytest.mark.parametrize("changed_source", ["consumed_png", "manifest"])
def test_late_source_mutation_cannot_publish_or_verify_stale_source_identity(
    tmp_path, operation, changed_source
):
    recording = authored_recording(tmp_path, [10, 15, 20])
    replay = tmp_path / "replay"
    original_session = validate_recording(recording)["session_id"]

    class LateMutatingDetector(MarkerDetector):
        armed = False
        changed = False

        def detect(self, frame):
            detections = super().detect(frame)
            if self.armed and frame.sequence == 2:
                if changed_source == "consumed_png":
                    path = recording / "frames" / "000000.png"
                    with Image.open(path) as original:
                        changed = original.convert("RGB")
                    changed.putpixel((0, 0), (33, 44, 55))
                    changed.save(path)
                else:
                    path = recording / "manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["source_uri"] = "urn:self:source-replaced-during-inference"
                    body = {key: value for key, value in manifest.items() if key != "session_id"}
                    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                    manifest["session_id"] = hashlib.sha256(encoded).hexdigest()
                    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
                self.changed = True
            return detections

    detector = LateMutatingDetector()
    if operation == "verify":
        track_recording(recording, replay, detector)
        replay_before = {path.name: path.read_bytes() for path in replay.iterdir()}
        detector.seen.clear()
    detector.armed = True
    with pytest.raises(ValueError):
        if operation == "track":
            track_recording(recording, replay, detector)
        else:
            verify_recording_tracking(recording, replay, detector)

    assert detector.changed is True
    assert detector.seen == [(0, 0), (1, 50), (2, 100)]
    if changed_source == "manifest":
        # A different, internally valid session must not inherit the old source claim.
        assert validate_recording(recording)["session_id"] != original_session
    if operation == "track":
        assert not replay.exists()
        assert not list(tmp_path.glob(".replay-*"))
    else:
        assert {path.name: path.read_bytes() for path in replay.iterdir()} == replay_before


@pytest.mark.parametrize("changed_file", ["consumed_trace_row", "report"])
def test_late_replay_mutation_cannot_verify_earlier_read_evidence(tmp_path, changed_file):
    recording = authored_recording(tmp_path, [10, 15, 20])
    replay = tmp_path / "replay"

    class LateReplayMutatingDetector(MarkerDetector):
        armed = False
        changed = False

        def detect(self, frame):
            detections = super().detect(frame)
            if self.armed and frame.sequence == 2:
                if changed_file == "consumed_trace_row":
                    with (replay / "tracks.jsonl").open("r+b") as stream:
                        first = stream.readline()
                        row = json.loads(first)
                        row["belief"]["tracks"][0]["confidence"] = 0.94
                        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                        assert len(encoded) + 1 == len(first)
                        # Equal-length in-place mutation leaves all unread row offsets intact.
                        stream.seek(0)
                        stream.write(encoded + b"\n")
                else:
                    rewrite_report(replay, lambda report: report.update(frames_processed=999))
                self.changed = True
            return detections

    detector = LateReplayMutatingDetector()
    track_recording(recording, replay, detector)
    detector.seen.clear()
    detector.armed = True
    with pytest.raises(ValueError, match="replay changed during verification"):
        verify_recording_tracking(recording, replay, detector)
    assert detector.changed is True
    assert detector.seen == [(0, 0), (1, 50), (2, 100)]
    if changed_file == "consumed_trace_row":
        assert trace(replay)[0]["belief"]["tracks"][0]["confidence"] == 0.94
    else:
        assert json.loads((replay / "report.json").read_text())["frames_processed"] == 999


def test_verification_recomputes_beliefs_after_trace_hash_is_recomputed(tmp_path):
    recording = authored_recording(tmp_path, [10, 15, 20])
    replay = tmp_path / "replay"
    track_recording(recording, replay, MarkerDetector())
    rows = trace(replay)
    rows[1]["belief"]["tracks"][0]["velocity_px_per_second"] = [12345.0, 0.0]
    trace_path = replay / "tracks.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    refresh_file_hash(replay, "tracks.jsonl", "tracks_sha256")
    with pytest.raises(ValueError, match="trajectory mismatch"):
        verify_recording_tracking(recording, replay, MarkerDetector())


def test_verification_rejects_replaced_gif_even_with_an_updated_hash(tmp_path):
    recording = authored_recording(tmp_path, [10, 15, 20])
    replay = tmp_path / "replay"
    track_recording(recording, replay, MarkerDetector(), preview_stride=1)
    preview_path = replay / "preview.gif"
    with Image.open(preview_path) as original:
        durations = [frame.info.get("duration", 100) for frame in ImageSequence.Iterator(original)]
        original.seek(0)
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(original)]
    frames[0].putpixel((0, 0), (255, 255, 0))
    frames[0].save(
        preview_path, save_all=True, append_images=frames[1:], duration=durations, loop=0
    )
    refresh_file_hash(replay, "preview.gif", "preview_sha256")
    with pytest.raises(ValueError, match="preview differs"):
        verify_recording_tracking(recording, replay, MarkerDetector())


def test_verification_binds_state_configuration_and_detector_identity(tmp_path):
    class OtherMarkerDetector(MarkerDetector):
        pass

    recording = authored_recording(tmp_path, [10, 15])
    replay = tmp_path / "replay"
    track_recording(recording, replay, MarkerDetector())
    with pytest.raises(ValueError):
        verify_recording_tracking(recording, replay, OtherMarkerDetector())
    rewrite_report(replay, lambda report: report["state_config"].update(max_match_distance=1.0))
    with pytest.raises(ValueError):
        verify_recording_tracking(recording, replay, MarkerDetector())


def test_native_model_hash_and_thresholds_are_bound_even_for_same_predictions(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    # Constant output is solely an adapter-identity fixture, not tracking/accuracy evidence.
    node = onnx.helper.make_node(
        "Constant",
        [],
        ["detections"],
        value=onnx.numpy_helper.from_array(np.array([[[4, 5, 20, 25, 0.9, 0]]], dtype=np.float32)),
    )
    graph = onnx.helper.make_graph(
        [node],
        "recording-tracking-model-identity",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 64, 64])],
        [onnx.helper.make_tensor_value_info("detections", onnx.TensorProto.FLOAT, [1, 1, 6])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )
    model_path, manifest_path = tmp_path / "identity.onnx", tmp_path / "model-manifest.json"
    onnx.save(model, model_path)
    manifest = ModelManifest(
        hashlib.sha256(model_path.read_bytes()).hexdigest(),
        ("player",),
        64,
        64,
        "end2end_xyxy",
        "synthetic_fixture",
    )
    manifest_path.write_text(json.dumps(asdict(manifest)))
    recording = authored_recording(tmp_path, [10, 15])
    replay = tmp_path / "replay"
    detector = OnnxDetector(model_path, manifest_path)
    track_recording(recording, replay, detector)
    assert verify_recording_tracking(recording, replay, detector)["verified"] is True

    different_threshold = OnnxDetector(model_path, manifest_path, confidence_threshold=0.30)
    with pytest.raises(ValueError):
        verify_recording_tracking(recording, replay, different_threshold)
    detector.manifest = replace(detector.manifest, model_sha256="0" * 64)
    with pytest.raises(ValueError):
        verify_recording_tracking(recording, replay, detector)
