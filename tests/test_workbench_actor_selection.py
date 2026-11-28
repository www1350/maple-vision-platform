"""Independent HTTP checks for source-bound target selection in the workbench."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from maple_vision.catalog import canonical_json
from maple_vision.contracts import Detection
from maple_vision.recording_evaluation import summarize_recording_windows
from maple_vision.recording_tracking import track_recording, verify_recording_tracking
from maple_vision.recordings import import_recording
from maple_vision.workbench_api import create_app

ORIGIN = "http://127.0.0.1"
FIRST = (30, 60, 42, 80)
SECOND = (110, 60, 122, 80)
COLORS = ((218, 51, 77), (44, 203, 230))


class VisiblePlayers:
    """Detect two authored colored figures using pixels, never review data."""

    def detect(self, frame):
        pixels = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        detections = []
        for color in COLORS:
            ys, xs = np.nonzero(np.all(pixels == color, axis=2))
            if len(xs):
                detections.append(
                    Detection(
                        "player",
                        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                        0.95,
                    )
                )
        return tuple(detections)


def _image():
    image = Image.new("RGB", (160, 120), (12, 18, 24))
    draw = ImageDraw.Draw(image)
    for box, color in zip((FIRST, SECOND), COLORS, strict=True):
        left, top, right, bottom = box
        draw.rectangle((left, top, right - 1, bottom - 1), fill=color)
    return image


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, document):
    path.write_bytes((canonical_json(document) + "\n").encode("utf-8"))


def _payload(manifest, local=1):
    frame = manifest["frames"][local]
    return {
        "source_sequence": manifest.get("source_start_frame", 0) + local,
        "rgb_sha256": frame["rgb_sha256"],
        "bbox": list(FIRST),
        "reviewer": "independent HTTP review",
        "subject": "red visible figure",
        "user_confirmed": True,
    }


@pytest.fixture
def http_recording(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        with _image() as image:
            image.save(source / f"{index:04d}.png")
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    recording = recordings / "selected-window"
    import_recording(
        source,
        recording,
        source_uri="urn:self:workbench-independent-two-players",
        permission="self_authored",
        license="CC0",
        fps=20,
        start_frame=8,
        max_frames=4,
    )
    return recordings, recording, _json(recording / "manifest.json")


def test_http_selection_uses_absolute_source_frame_and_server_local_timestamp(
    http_recording, tmp_path
):
    recordings, recording, manifest = http_recording
    before = {
        path.relative_to(recording): path.read_bytes()
        for path in recording.rglob("*")
        if path.is_file()
    }
    identity = manifest["session_id"]
    with TestClient(
        create_app(tmp_path / "workspace", recordings=recordings), base_url=ORIGIN
    ) as http:
        detail = http.get(f"/api/recordings/{identity}")
        assert detail.status_code == 200
        frame = detail.json()["frames"][1]
        assert frame["source_sequence"] == 9
        assert frame["rgb_sha256"] == manifest["frames"][1]["rgb_sha256"]
        response = http.post(f"/api/recordings/{identity}/actor-selection", json=_payload(manifest))
        assert response.status_code == 200, response.text
        selection = response.json()
        assert selection["kind"] == "recording_actor_selection"
        assert selection["source"]["session_id"] == identity
        assert selection["source"]["source_start_frame"] == 8
        assert selection["anchor"]["source_sequence"] == 9
        assert selection["anchor"]["timestamp_ms"] == 50
        assert selection["anchor"]["rgb_sha256"] == frame["rgb_sha256"]
        assert selection["anchor"]["bbox"] == list(FIRST)
        assert selection["anchor"]["reviewer"] == "independent HTTP review"
        assert "user_confirmed" not in selection["anchor"]
    assert {
        path.relative_to(recording): path.read_bytes()
        for path in recording.rglob("*")
        if path.is_file()
    } == before

    replay = tmp_path / "replay"
    report = track_recording(
        recording, replay, VisiblePlayers(), actor_selection=selection, preview_max_frames=2
    )
    assert report["actor_selection_analysis"]["status_counts"] == {"waiting": 1, "selected": 3}
    verified = verify_recording_tracking(recording, replay, VisiblePlayers())
    assert verified["verified"] and verified["actor_selection_recomputed"]
    assert "movement_recomputed" not in verified


def _exact_window(root, name, start, pts, *, truncated):
    """Independent source-clock fixture with two actual PNG bodies per frame."""
    session = root / name
    frames = session / "frames"
    frames.mkdir(parents=True)
    rows = []
    for sequence, timestamp in enumerate(pts):
        with _image() as image:
            pixels = image.tobytes()
            file = f"frames/{sequence:06d}.png"
            image.save(session / file)
        data = (session / file).read_bytes()
        rows.append(
            {
                "sequence": sequence,
                "timestamp_ms": timestamp - pts[0],
                "file": file,
                "sha256": hashlib.sha256(data).hexdigest(),
                "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "source_name": f"video-frame-{start + sequence:06d}",
                "source_sha256": None,
                "width": 160,
                "height": 120,
                "pts": timestamp,
            }
        )
    manifest = {
        "schema_version": 2 if start == 0 else 3,
        "kind": "offline_recording",
        "scope": "user_supplied_recording",
        "source_kind": "mp4",
        "source_uri": "urn:self:independent-two-player-window-clock",
        "permission": "self_authored",
        "license": "CC0",
        "source_name": "authored-clock-fixture.mp4",
        "source_sha256": "b" * 64,
        "source_hash_basis": "mp4_file",
        "fps": None,
        "timebase": "video_presentation_ticks",
        "frame_count": len(rows),
        "width": 160,
        "height": 120,
        "max_frames": len(rows),
        "total_source_frames": None,
        "truncated": truncated,
        "completion_reason": "frame_limit" if truncated else "decoder_stopped",
        "frames": rows,
        "video_clock": {"stream_index": 0, "numerator": 1, "denominator": 1000},
    }
    if start:
        manifest["source_start_frame"] = start
    manifest["session_id"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    _write(session / "manifest.json", manifest)
    return session, manifest


def test_coverage_frame_routes_selection_to_its_second_window(tmp_path):
    recordings = tmp_path / "coverage-recordings"
    recordings.mkdir()
    first, _ = _exact_window(recordings, "first", 0, [100, 140], truncated=True)
    second, manifest = _exact_window(recordings, "second", 2, [220, 360], truncated=False)
    coverage_path = tmp_path / "coverage.json"
    summarize_recording_windows([second, first], coverage_path)
    app = create_app(
        tmp_path / "coverage-workspace", recordings=recordings, recording_coverages=[coverage_path]
    )
    with TestClient(app, base_url=ORIGIN) as http:
        coverages = http.get("/api/recording-coverages").json()
        coverage_id = coverages[0]["coverage_sha256"]
        detail = http.get(f"/api/recording-coverages/{coverage_id}").json()
        frame = detail["frames"][3]
        assert frame["session_id"] == manifest["session_id"]
        assert frame["source_sequence"] == 3
        assert frame["rgb_sha256"] == manifest["frames"][1]["rgb_sha256"]
        response = http.post(
            f"/api/recordings/{frame['session_id']}/actor-selection", json=_payload(manifest)
        )
        assert response.status_code == 200, response.text
        selection = response.json()
        assert selection["anchor"]["source_sequence"] == 3
        assert selection["anchor"]["timestamp_ms"] == 140
        assert selection["source"]["source_start_timestamp_ms"] == 220
        assert selection["source"]["session_id"] == manifest["session_id"]
    replay = tmp_path / "second-window-replay"
    track_recording(
        second, replay, VisiblePlayers(), actor_selection=selection, preview_max_frames=2
    )
    assert verify_recording_tracking(second, replay, VisiblePlayers())["actor_selection_recomputed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_rgb",
        "uppercase_rgb",
        "local_instead_of_source",
        "source_boolean",
        "outside_source",
        "unconfirmed",
        "integer_confirmation",
        "string_confirmation",
        "missing_confirmation",
        "extra_timestamp",
        "extra_source",
        "empty_reviewer",
        "long_subject",
        "missing_subject",
        "negative_box",
        "outside_box",
        "empty_box",
        "reversed_box",
        "boolean_box",
        "string_box",
        "short_box",
        "nan_box",
    ],
)
def test_bad_or_unconfirmed_actor_selection_requests_are_rejected(
    http_recording, tmp_path, mutation
):
    recordings, _, manifest = http_recording
    payload = deepcopy(_payload(manifest))
    if mutation == "wrong_rgb":
        payload["rgb_sha256"] = "0" * 64
    elif mutation == "uppercase_rgb":
        payload["rgb_sha256"] = "A" * 64
    elif mutation == "local_instead_of_source":
        payload["source_sequence"] = 1
    elif mutation == "source_boolean":
        payload["source_sequence"] = True
    elif mutation == "outside_source":
        payload["source_sequence"] = 12
    elif mutation == "unconfirmed":
        payload["user_confirmed"] = False
    elif mutation == "integer_confirmation":
        payload["user_confirmed"] = 1
    elif mutation == "string_confirmation":
        payload["user_confirmed"] = "true"
    elif mutation == "missing_confirmation":
        del payload["user_confirmed"]
    elif mutation == "extra_timestamp":
        payload["timestamp_ms"] = 999
    elif mutation == "extra_source":
        payload["source"] = {"session_id": manifest["session_id"]}
    elif mutation == "empty_reviewer":
        payload["reviewer"] = "   "
    elif mutation == "long_subject":
        payload["subject"] = "s" * 129
    elif mutation == "missing_subject":
        del payload["subject"]
    else:
        payload["bbox"] = {
            "negative_box": [-1, 60, 42, 80],
            "outside_box": [30, 60, 161, 80],
            "empty_box": [30, 60, 30, 80],
            "reversed_box": [42, 60, 30, 80],
            "boolean_box": [True, 60, 42, 80],
            "string_box": ["30", 60, 42, 80],
            "short_box": [30, 60, 42],
            "nan_box": [float("nan"), 60, 42, 80],
        }[mutation]
    with TestClient(
        create_app(tmp_path / "workspace", recordings=recordings), base_url=ORIGIN
    ) as http:
        response = http.post(
            f"/api/recordings/{manifest['session_id']}/actor-selection",
            content=json.dumps(payload, allow_nan=True),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, response.text


def test_unknown_recording_selection_is_not_resolved_from_user_supplied_paths(
    http_recording, tmp_path
):
    recordings, _, manifest = http_recording
    with TestClient(
        create_app(tmp_path / "workspace", recordings=recordings), base_url=ORIGIN
    ) as http:
        response = http.post(f"/api/recordings/{'f' * 64}/actor-selection", json=_payload(manifest))
        assert response.status_code == 404


@pytest.mark.parametrize("coherent_replacement", [False, True])
def test_registered_recording_mutation_cannot_rebind_selection(
    http_recording, tmp_path, coherent_replacement
):
    recordings, recording, manifest = http_recording
    with TestClient(
        create_app(tmp_path / "workspace", recordings=recordings), base_url=ORIGIN
    ) as http:
        payload = _payload(manifest)
        frame = manifest["frames"][1]
        path = recording / frame["file"]
        with Image.open(path) as original:
            changed = original.convert("RGB")
        changed.putpixel((0, 0), (91, 92, 93))
        changed.save(path)
        if coherent_replacement:
            frame["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame["rgb_sha256"] = hashlib.sha256(changed.tobytes()).hexdigest()
            old_identity = manifest.pop("session_id")
            manifest["session_id"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
            _write(recording / "manifest.json", manifest)
            payload["rgb_sha256"] = frame["rgb_sha256"]
        else:
            old_identity = manifest["session_id"]
        changed.close()
        response = http.post(f"/api/recordings/{old_identity}/actor-selection", json=payload)
        assert response.status_code == 422, response.text
