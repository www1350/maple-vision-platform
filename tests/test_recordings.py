import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.catalog import canonical_json
from maple_vision.contracts import Detection
from maple_vision.recordings import (
    import_recording,
    iter_recording,
    observe_recording,
    validate_recording,
)
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    simulator = Simulator(seed=9)
    renderer = SceneRenderer()
    (source / "0001.png").write_bytes(renderer.render(simulator.snapshot()).png_bytes())
    simulator.step(SimAction("move", direction=1))
    (source / "0002.png").write_bytes(renderer.render(simulator.snapshot()).png_bytes())
    Image.new("RGB", (320, 192)).save(source / "0003.png")
    return source


def ingest(source, output, **kwargs):
    options = dict(
        source_uri="urn:self:recording", permission="self_authored", license="CC0-1.0", fps=20
    )
    options.update(kwargs)
    return import_recording(source, output, **options)


@pytest.fixture
def recording(source, tmp_path):
    output = tmp_path / "recording"
    ingest(source, output)
    return output


def modify_manifest(recording, edit):
    path = recording / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    body = {key: value for key, value in manifest.items() if key != "session_id"}
    manifest["session_id"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    path.write_text(canonical_json(manifest))


def test_png_import_and_replay_pixel_identity(recording, source):
    summary = validate_recording(recording)
    assert summary["frame_count"] == 3
    assert summary["timing_accuracy"] == "nominal_fps_only"
    frames = list(iter_recording(recording))
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_ms for frame in frames] == [0, 50, 100]
    for frame, source_path in zip(frames, sorted(source.glob("*.png")), strict=True):
        with Image.open(source_path) as image:
            assert frame.pixels == image.convert("RGB").tobytes()
        assert set(frame.__dataclass_fields__) == {
            "sequence",
            "timestamp_ms",
            "width",
            "height",
            "pixels",
        }


def test_import_is_reproducible_and_truncation_explicit(source, tmp_path):
    a = ingest(source, tmp_path / "a", max_frames=2)
    b = ingest(source, tmp_path / "b", max_frames=2)
    assert a["session_id"] == b["session_id"]
    assert a["truncated"] is True
    assert a["completion_reason"] == "frame_limit"
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["total_source_frames"] == 3
    assert manifest["source_hash_basis"] == "selected_png_inputs"


@pytest.mark.parametrize("fps", [None, True, 0, -1, 241, float("nan"), float("inf"), "20"])
def test_directory_invalid_fps_rejected(source, tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        ingest(source, tmp_path / "bad", fps=fps)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("maximum", [0, True, 10001, -1, 1.5])
def test_invalid_frame_limit(source, tmp_path, maximum):
    with pytest.raises(ValueError, match="max_frames"):
        ingest(source, tmp_path / "bad", max_frames=maximum)


def test_fractional_fps_uses_explicit_nominal_timeline(source, tmp_path):
    output = tmp_path / "recording"
    ingest(source, output, fps=29.97)
    assert [frame.timestamp_ms for frame in iter_recording(output)] == [0, 33, 67]


def test_missing_source_url_and_unlicensed_input_rejected(source, tmp_path):
    with pytest.raises(ValueError, match="local"):
        ingest(Path("https://example.org/video.mp4"), tmp_path / "bad")
    with pytest.raises(ValueError, match="permission"):
        ingest(source, tmp_path / "bad", permission="unknown")


def test_output_never_overwrites_and_cannot_be_inside_source(source, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        ingest(source, output)
    assert (output / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        ingest(source, source / "nested")


@pytest.mark.parametrize("mode", ["corrupt", "resize", "transparent"])
def test_invalid_later_frame_does_not_publish_partial_session(source, tmp_path, mode):
    target = source / "0002.png"
    if mode == "corrupt":
        target.write_bytes(b"bad png")
    elif mode == "resize":
        Image.new("RGB", (319, 192)).save(target)
    else:
        Image.new("RGBA", (320, 192), (255, 0, 0, 100)).save(target)
    with pytest.raises(ValueError):
        ingest(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_total_pixel_budget_is_enforced_before_publication(source, tmp_path, monkeypatch):
    import maple_vision.recordings as module

    monkeypatch.setattr(module, "MAX_TOTAL_PIXELS", 1000)
    with pytest.raises(ValueError, match="budget"):
        ingest(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_manifest_hash_detects_changes(recording):
    path = recording / "manifest.json"
    value = json.loads(path.read_text())
    value["fps"] = 30
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        validate_recording(recording)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["frames"][1].update(sequence=0),
        lambda m: m["frames"][1].update(sequence=True),
        lambda m: m["frames"][1].update(timestamp_ms=12),
        lambda m: m["frames"][1].update(file="../outside.png"),
        lambda m: m["frames"][1].update(width=319),
        lambda m: m.update(frame_count=2),
        lambda m: m.update(truncated=True),
        lambda m: m.update(total_source_frames=False),
        lambda m: m.update(timebase=[]),
        lambda m: m["frames"][1].update(source_name=m["frames"][0]["source_name"]),
    ],
)
def test_internal_invariants_checked_even_when_manifest_digest_is_recomputed(recording, edit):
    modify_manifest(recording, edit)
    with pytest.raises(ValueError):
        validate_recording(recording)


def test_frame_content_and_unlisted_files_rejected(recording):
    (recording / "frames" / "extra.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra"):
        validate_recording(recording)
    (recording / "frames" / "extra.png").unlink()
    (recording / "frames" / "000001.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        validate_recording(recording)


def test_frame_is_rechecked_at_consumption_time(recording):
    iterator = iter_recording(recording)
    assert next(iterator).sequence == 0
    (recording / "frames" / "000001.png").write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="SHA256"):
        next(iterator)


def test_pixel_replay_reports_perception_only(recording, tmp_path):
    output = tmp_path / "observed"
    report = observe_recording(recording, output)
    assert report["frames_processed"] == 3
    assert report["status_counts"] == {"known": 2, "unknown": 1}
    assert report["accuracy_metrics_available"] is False
    assert report["effect_success_rate"] is None
    assert report["action_execution"] == "none"
    lines = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(lines) == 3
    assert all("intent" not in line and "execution" not in line for line in lines)
    with pytest.raises(FileExistsError):
        observe_recording(recording, output)


def test_custom_detector_does_not_manufacture_world_state(recording, tmp_path):
    class Detector:
        def detect(self, frame):
            assert not hasattr(frame, "annotations")
            return (Detection("player", (0.0, 0.0, 10.0, 10.0), 0.7),)

    output = tmp_path / "custom"
    report = observe_recording(recording, output, Detector())
    assert report["status_counts"] == {"unknown": 3}
    assert report["detections_by_class"] == {"player": 3}
    lines = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert all(line["observation"]["map_id"] is None for line in lines)
    assert all(line["observation"]["hp_ratio"] is None for line in lines)


def test_bad_detector_result_does_not_publish_success(recording, tmp_path):
    class Detector:
        def detect(self, frame):
            return (Detection("player", (0.0, 0.0, 10000.0, 10.0), 0.9),)

    with pytest.raises(ValueError, match="outside"):
        observe_recording(recording, tmp_path / "bad", Detector())
    assert not (tmp_path / "bad").exists()


def test_mp4_rejects_playlist_before_decoder_import(tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_text("#EXTM3U\nhttps://example.org/video")
    with pytest.raises(ValueError, match="header"):
        ingest(path, tmp_path / "bad")


def test_real_local_mp4_decode(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2", reason="optional local MP4 decoder")
    import numpy as np

    path = tmp_path / "authored.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("local OpenCV build has no MP4 encoder for this test fixture")
    try:
        for value in (255, 220, 180):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 2] = value
            writer.write(image)
    finally:
        writer.release()
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file")
    output = tmp_path / "recorded"
    report = ingest(path, output, fps=None, max_frames=2)
    assert report["fps"] == pytest.approx(10.0)
    assert report["truncated"] is True
    assert report["timebase"] == "nominal_fps_video_metadata"
    frames = list(iter_recording(output))
    assert [frame.timestamp_ms for frame in frames] == [0, 100]
    assert frames[0].pixels[0] > 200 and frames[0].pixels[1] < 20
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "protocol_whitelist;file"
    path.unlink()  # capture released; Windows must not retain a decoder file handle.


@pytest.fixture
def variable_rate_mp4(tmp_path):
    from fractions import Fraction

    av = pytest.importorskip("av")
    path = tmp_path / "variable.mp4"
    clock = Fraction(1, 1000)
    with av.open(str(path), "w", format="mp4") as writer:
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
    return path


@pytest.mark.parametrize("maximum,expected", [(3, [0, 40, 160]), (4, [0, 40, 160, 380])])
def test_exact_pts_come_from_the_same_decoded_frame(variable_rate_mp4, tmp_path, maximum, expected):
    from fractions import Fraction

    from maple_vision.cli import main

    output = tmp_path / "exact"
    assert (
        main(
            [
                "recording",
                "import",
                str(variable_rate_mp4),
                str(output),
                "--source-uri",
                "urn:self:vfr",
                "--permission",
                "self_authored",
                "--license",
                "CC0",
                "--presentation-time",
                "--max-frames",
                str(maximum),
            ]
        )
        == 0
    )
    result = validate_recording(output)
    assert result["timing_accuracy"] == "source_pts" and result["fps"] is None
    assert result["truncated"] == (maximum < 4)
    manifest = json.loads((output / "manifest.json").read_bytes())
    info = manifest["video_clock"]
    unit = Fraction(info["numerator"], info["denominator"])
    origin = manifest["frames"][0]["pts"]
    assert origin * unit == Fraction(14, 100)
    assert [(f["pts"] - origin) * unit * 1000 for f in manifest["frames"]] == expected
    frames = list(iter_recording(output))
    assert [f.timestamp_ms for f in frames] == expected
    for frame, shade in zip(frames, [20, 75, 130, 200][:maximum], strict=True):
        assert abs(frame.pixels[0] - shade) <= 4


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["frames"][1].update(pts=d["frames"][0]["pts"]),
        lambda d: d["frames"][2].update(timestamp_ms=80),
        lambda d: d["video_clock"].update(denominator=0),
        lambda d: d.update(timebase="nominal_fps_user"),
    ],
)
def test_inconsistent_exact_timeline_fails_after_rehash(variable_rate_mp4, tmp_path, edit):
    output = tmp_path / "exact"
    ingest(variable_rate_mp4, output, fps=None, presentation_time=True)
    modify_manifest(output, edit)
    with pytest.raises(ValueError):
        validate_recording(output)


def test_pts_rejects_a_directory_or_fps_override(source, tmp_path):
    with pytest.raises(ValueError, match="without an fps override"):
        ingest(source, tmp_path / "out", presentation_time=True)
    assert not (tmp_path / "out").exists()
