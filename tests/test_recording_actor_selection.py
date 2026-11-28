"""Source pixels, explicit actor review and replay must agree in multi-player recordings.

The two figures use independent visible colors solely to make detector plumbing
reproducible. No detector reads frame numbers, target selection or expected state.
The authored scene is an engineering fixture, not real-domain identification data.
"""

import hashlib
import json
import shutil
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from maple_vision.catalog import canonical_json
from maple_vision.contracts import Detection
from maple_vision.movement_hypotheses import MovementConfig
from maple_vision.recording_movement_evaluation import (
    benchmark_movement_recording,
    create_movement_label_template,
)
from maple_vision.recording_tracking import (
    create_recording_actor_selection,
    track_recording,
    verify_recording_tracking,
)
from maple_vision.recordings import import_recording
from maple_vision.scene import PALETTE, SceneRenderer
from maple_vision.simulation import Simulator

COLORS = {
    "climber": (240, 40, 220),
    "bystander": (90, 230, 60),
    "rope": (220, 180, 30),
    "platform": (30, 180, 230),
}
SOURCE_START = 17
FRAME_COUNT = 25


class ColoredPlayersDetector:
    def detect(self, frame):
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        detections = []
        for marker, color in COLORS.items():
            mask = np.all(rgb == color, axis=2)
            rows = np.flatnonzero(np.any(mask, axis=1))
            if not len(rows):
                continue
            groups = [rows]
            if marker == "platform":
                groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
            for group in groups:
                ys, xs = np.nonzero(mask[group[0] : group[-1] + 1])
                detections.append(
                    Detection(
                        "player" if marker in {"climber", "bystander"} else marker,
                        (
                            int(xs.min()),
                            int(group[0] + ys.min()),
                            int(xs.max()) + 1,
                            int(group[0] + ys.max()) + 1,
                        ),
                        0.95,
                    )
                )
        return tuple(detections)


def _climber_box(local):
    feet = 130 - 4 * min(local, 20)
    return 75, feet - 16, 85, feet


def _make_recording(root, *, missing_climber=None):
    source = root / "source"
    source.mkdir(parents=True)
    for source_sequence in range(SOURCE_START + FRAME_COUNT):
        local = max(0, source_sequence - SOURCE_START)
        image = Image.new("RGB", (320, 256), (12, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((78, 30, 81, 149), fill=COLORS["rope"])
        draw.rectangle((50, 50, 99, 54), fill=COLORS["platform"])
        draw.rectangle((160, 130, 199, 134), fill=COLORS["platform"])
        if local != missing_climber:
            left, top, right, bottom = _climber_box(local)
            draw.rectangle((left, top, right - 1, bottom - 1), fill=COLORS["climber"])
        draw.rectangle((225, 114, 234, 129), fill=COLORS["bystander"])
        image.save(source / f"{source_sequence:04d}.png")
        image.close()
    recording = root / "recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:two-visible-player-pixels",
        permission="self_authored",
        license="CC0",
        fps=20,
        start_frame=SOURCE_START,
        max_frames=FRAME_COUNT,
    )
    return recording


def _selection(recording, output, *, local=0, bystander=False):
    return create_recording_actor_selection(
        recording,
        output,
        source_sequence=SOURCE_START + local,
        bbox=(225, 114, 235, 130) if bystander else _climber_box(local),
        reviewer="independent pixel review",
        subject="stationary visible figure" if bystander else "rising visible figure",
    )


def _track(recording, output, specification):
    return track_recording(
        recording,
        output,
        ColoredPlayersDetector(),
        movement_config=MovementConfig(),
        actor_selection=specification,
        preview_max_frames=3,
    )


def _rows(replay):
    return [json.loads(line) for line in (replay / "tracks.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def selected_recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("two-visible-player-recording")
    recording = _make_recording(root)
    specification = _selection(recording, root / "selection.json")
    replay = root / "replay"
    report = _track(recording, replay, specification)
    return recording, replay, specification, report


@pytest.mark.parametrize("bystander", [False, True])
def test_visible_other_player_is_retained_but_only_selected_actor_supplies_movement(
    selected_recording, tmp_path, bystander
):
    recording, replay, specification, report = selected_recording
    if bystander:
        specification = _selection(recording, tmp_path / "other-selection.json", bystander=True)
        replay = tmp_path / "other-replay"
        report = _track(recording, replay, specification)
    rows = _rows(replay)
    selected_ids = {row["actor_selection"]["actor_track_id"] for row in rows}
    assert len(selected_ids) == 1
    assert None not in selected_ids
    assert len(rows) == FRAME_COUNT
    for local, row in enumerate(rows):
        players = [
            track
            for track in row["belief"]["tracks"]
            if track["class_id"] == "player" and track["observed_this_frame"]
        ]
        assert len(players) == 2
        assert row["actor_selection"]["status"] == "selected"
        assert row["source_sequence"] == SOURCE_START + local
        selected = next(track for track in players if track["track_id"] in selected_ids)
        assert selected["bbox"] == list((225, 114, 235, 130) if bystander else _climber_box(local))
        assert row["movement"]["actor_selection"] == row["actor_selection"]
        assert row["belief"]["actionable"] is False
    states = {row["movement"]["state"] for row in rows}
    if bystander:
        assert not states & {"climbing_candidate", "platform_arrival_candidate"}
    else:
        assert "climbing_candidate" in states
        assert rows[-1]["movement"]["state"] == "platform_arrival_candidate"
    assert report["actor_selection_analysis"]["specification"] == specification
    assert report["actor_selection_analysis"]["status_counts"] == {"selected": FRAME_COUNT}
    assert report["preview"]["renderer"] == "recording_selected_actor_overlay_v1"
    checked = verify_recording_tracking(recording, replay, ColoredPlayersDetector())
    assert checked["verified"] is True
    assert checked["actor_selection_recomputed"] is True
    assert checked["movement_recomputed"] is True
    assert checked["frames_verified"] == FRAME_COUNT


def test_middle_anchor_does_not_retroactively_select_or_inherit_earlier_climb(
    selected_recording, tmp_path
):
    recording, _, _, _ = selected_recording
    specification = _selection(recording, tmp_path / "middle-selection.json", local=10)
    assert specification["anchor"]["source_sequence"] == 27
    assert specification["anchor"]["timestamp_ms"] == 500
    replay = tmp_path / "middle-replay"
    report = _track(recording, replay, specification)
    rows = _rows(replay)
    assert all(row["actor_selection"]["status"] == "waiting" for row in rows[:10])
    assert all(row["movement"]["state"] == "unknown" for row in rows[:10])
    assert rows[10]["actor_selection"]["status"] == "selected"
    assert rows[10]["movement"]["state"] == "unknown"
    assert rows[11]["movement"]["state"] != "climbing_candidate"
    assert any(row["movement"]["state"] == "climbing_candidate" for row in rows[12:20])
    assert rows[-1]["movement"]["state"] == "platform_arrival_candidate"
    assert report["actor_selection_analysis"]["status_counts"] == {"waiting": 10, "selected": 15}
    assert verify_recording_tracking(recording, replay, ColoredPlayersDetector())["verified"]


def test_selected_dropout_keeps_other_person_observed_but_never_reselects_after_return(tmp_path):
    recording = _make_recording(tmp_path, missing_climber=10)
    specification = _selection(recording, tmp_path / "selection.json")
    replay = tmp_path / "replay"
    report = _track(recording, replay, specification)
    rows = _rows(replay)
    assert all(row["actor_selection"]["status"] == "selected" for row in rows[:10])
    assert all(row["actor_selection"]["status"] == "invalidated" for row in rows[10:])
    assert all(row["actor_selection"]["actor_track_id"] is None for row in rows[10:])
    assert all(row["movement"]["state"] == "unknown" for row in rows[10:])
    for row in rows:
        assert any(
            track["class_id"] == "player"
            and track["observed_this_frame"]
            and track["bbox"] == [225, 114, 235, 130]
            for track in row["belief"]["tracks"]
        )
    assert report["actor_selection_analysis"]["status_counts"] == {
        "selected": 10,
        "invalidated": 15,
    }
    assert verify_recording_tracking(recording, replay, ColoredPlayersDetector())["verified"]


def test_selected_id_forgery_is_rejected_after_refreshing_all_trace_hashes(
    selected_recording, tmp_path
):
    recording, original, _, _ = selected_recording
    replay = tmp_path / "forged-replay"
    shutil.copytree(original, replay)
    rows = _rows(replay)
    row = rows[10]
    other_id = next(
        track["track_id"]
        for track in row["belief"]["tracks"]
        if track["class_id"] == "player" and track["bbox"] == [225, 114, 235, 130]
    )
    assert other_id != row["actor_selection"]["actor_track_id"]
    row["actor_selection"]["actor_track_id"] = other_id
    row["movement"]["actor_selection"]["actor_track_id"] = other_id
    row["movement"]["actor_track_id"] = other_id
    encoded = "".join(canonical_json(value) + "\n" for value in rows).encode("utf-8")
    (replay / "tracks.jsonl").write_bytes(encoded)
    report_path = replay / "report.json"
    report = json.loads(report_path.read_text())
    report["tracks_sha256"] = report["files"]["tracks.jsonl"] = hashlib.sha256(encoded).hexdigest()
    report_path.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="trajectory mismatch"):
        verify_recording_tracking(recording, replay, ColoredPlayersDetector())


@pytest.mark.parametrize("mismatch", ["source", "rgb", "timestamp", "bounds"])
def test_misbound_selection_fails_before_any_replay_output_is_published(
    selected_recording, tmp_path, mismatch
):
    recording, _, original, _ = selected_recording
    specification = deepcopy(original)
    if mismatch == "source":
        specification["source"]["source_sha256"] = "f" * 64
    elif mismatch == "rgb":
        specification["anchor"]["rgb_sha256"] = "f" * 64
    elif mismatch == "timestamp":
        specification["anchor"]["timestamp_ms"] += 1
    else:
        specification["anchor"]["bbox"] = [75, 114, 321, 130]
    output = tmp_path / "invalid-replay"
    with pytest.raises(ValueError):
        _track(recording, output, specification)
    assert not output.exists()


def test_default_detector_selected_mode_keeps_fixture_map_and_resources_unknown(tmp_path):
    source = tmp_path / "fixture-source"
    source.mkdir()
    frame = SceneRenderer().render(Simulator(seed=9).snapshot())
    for index in range(3):
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels)
        ImageDraw.Draw(image).rectangle((250, 110, 263, 131), fill=PALETTE["player"])
        image.save(source / f"{index:04d}.png")
        image.close()
    recording = tmp_path / "fixture-recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:two-fixture-players",
        permission="self_authored",
        license="CC0",
        fps=20,
    )
    specification = create_recording_actor_selection(
        recording,
        tmp_path / "fixture-selection.json",
        source_sequence=0,
        bbox=(250, 110, 264, 132),
        reviewer="independent added figure",
        subject="upper right colored figure",
    )
    replay = tmp_path / "fixture-replay"
    report = track_recording(
        recording,
        replay,
        actor_selection=specification,
        movement_config=MovementConfig(),
        preview_max_frames=2,
    )
    assert report["perception_mode"] == "detector_only"
    assert report["detector_metadata"]["observation_adapter"] == "DetectorOnlyPerception.observe"
    for row in _rows(replay):
        observation = row["belief"]["observation"]
        assert observation["map_id"] is observation["hp_ratio"] is observation["mp_ratio"] is None
        assert row["belief"]["actionable"] is False
        assert row["actor_selection"]["status"] == "selected"
    assert verify_recording_tracking(recording, replay)["actor_selection_recomputed"] is True


def test_movement_benchmark_retains_selected_actor_contract_and_independent_reference(
    selected_recording, tmp_path
):
    recording, replay, _, tracking = selected_recording
    labels_path = tmp_path / "labels.json"
    labels = create_movement_label_template(recording, labels_path)
    labels.update(
        domain="authored_fixture",
        reviewer="independent state review",
        subject="rising magenta figure",
    )
    labels["frames"][10]["review"] = {
        "actor_bbox": list(_climber_box(10)),
        "rope_climb_visible": True,
        "arrival_after_climb_visible": False,
        "note": "The independently authored figure is halfway through its ascent.",
    }
    labels_path.write_bytes((canonical_json(labels) + "\n").encode("utf-8"))
    report = benchmark_movement_recording(
        recording, replay, labels_path, tmp_path / "benchmark.json", ColoredPlayersDetector()
    )
    assert report["verification"]["actor_selection_recomputed"] is True
    assert report["bindings"]["actor_selection_analysis"] == tracking["actor_selection_analysis"]
    assert report["predicates"]["rope_climb_visible"]["true_positives"] == 1
    assert report["predicates"]["rope_climb_visible"]["unreviewed"] == 24
    assert report["diagnostics"][10]["actor_match"]["status"] == "matched"
    assert report["owner_identity_verified"] is False
