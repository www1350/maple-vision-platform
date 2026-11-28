"""Independent pixel and review probes for selected-person identity evaluation.

The red and cyan people are authored outside the estimator. In the swap scene
they exchange places between observations: geometric positions stay continuous
while visible color identities do not. A stable internal track ID must not hide
that independently reviewed identity change. These are engineering fixtures.
"""

import hashlib
import json
import shutil
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image, ImageDraw

from maple_vision.catalog import canonical_json
from maple_vision.contracts import Detection
from maple_vision.movement_hypotheses import MovementConfig
from maple_vision.recording_actor_evaluation import (
    benchmark_actor_recording,
    create_actor_label_template,
)
from maple_vision.recording_tracking import (
    create_recording_actor_selection,
    track_recording,
    verify_recording_tracking,
)
from maple_vision.recordings import import_recording

SOURCE_START = 41
FRAME_COUNT = 8
LEFT = [30, 60, 42, 80]
RIGHT = [110, 60, 122, 80]
COLORS = {
    "red": (218, 51, 77),
    "cyan": (44, 203, 230),
    "platform": (190, 185, 60),
    "rope": (205, 70, 190),
}
OUTCOMES = {
    "correct_target",
    "wrong_other_actor",
    "localization_mismatch",
    "ambiguous_reference_match",
    "missed_visible_target",
    "correct_abstention_not_visible",
    "selected_when_not_visible",
    "unreviewed",
    "reference_uncertain",
}


class TwoColorDetector:
    def __init__(self, *, during_inference=None):
        self.during_inference = during_inference
        self.seen = []

    def detect(self, frame):
        self.seen.append((frame.sequence, frame.timestamp_ms))
        assert not hasattr(frame, "review")
        assert not hasattr(frame, "ground_truth")
        if self.during_inference is not None:
            callback, self.during_inference = self.during_inference, None
            callback()
        pixels = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        detections = []
        for name, color in COLORS.items():
            mask = np.all(pixels == color, axis=2)
            rows = np.flatnonzero(np.any(mask, axis=1))
            if not len(rows):
                continue
            groups = (
                np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
                if name == "platform"
                else [rows]
            )
            for group in groups:
                ys, xs = np.nonzero(mask[group[0] : group[-1] + 1])
                detections.append(
                    Detection(
                        "player" if name in {"red", "cyan"} else name,
                        (
                            int(xs.min()),
                            int(ys.min() + group[0]),
                            int(xs.max()) + 1,
                            int(ys.max() + group[0]) + 1,
                        ),
                        0.95,
                    )
                )
        return tuple(detections)


def _boxes(local, swap_at=None):
    return (RIGHT, LEFT) if swap_at is not None and local >= swap_at else (LEFT, RIGHT)


def _case(root, *, swap_at=None, missing_at=None, movement_config=None, selected_other=False):
    source = root / "source"
    source.mkdir(parents=True)
    for source_sequence in range(SOURCE_START + FRAME_COUNT):
        local = max(0, source_sequence - SOURCE_START)
        image = Image.new("RGB", (160, 120), (12, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 20, 37, 99), fill=COLORS["rope"])
        draw.rectangle((15, 40, 74, 43), fill=COLORS["platform"])
        draw.rectangle((90, 100, 144, 103), fill=COLORS["platform"])
        for name, box in zip(("red", "cyan"), _boxes(local, swap_at), strict=True):
            if name == "red" and local == missing_at:
                continue
            left, top, right, bottom = box
            draw.rectangle((left, top, right - 1, bottom - 1), fill=COLORS[name])
        image.save(source / f"{source_sequence:04d}.png")
        image.close()
    recording, replay = root / "recording", root / "replay"
    import_recording(
        source,
        recording,
        source_uri="urn:self:independent-actor-identity-pixels",
        permission="self_authored",
        license="CC0",
        fps=20,
        start_frame=SOURCE_START,
        max_frames=FRAME_COUNT,
    )
    selection = create_recording_actor_selection(
        recording,
        root / "selection.json",
        source_sequence=SOURCE_START,
        bbox=tuple(RIGHT if selected_other else LEFT),
        reviewer="independent anchor review",
        subject="cyan person" if selected_other else "red person",
    )
    track_recording(
        recording,
        replay,
        TwoColorDetector(),
        actor_selection=selection,
        movement_config=movement_config,
        preview_max_frames=2,
    )
    return recording, replay


@pytest.fixture(scope="module")
def actor_case(tmp_path_factory):
    return _case(tmp_path_factory.mktemp("independent-actor-review"))


@pytest.fixture(scope="module")
def swapped_case(tmp_path_factory):
    return _case(tmp_path_factory.mktemp("independent-actor-swap"), swap_at=3)


def _write(path, value):
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _rows(replay):
    return [json.loads(line) for line in (replay / "tracks.jsonl").read_text().splitlines()]


def _visible(local=0, *, swap_at=None):
    target, other = _boxes(local, swap_at)
    return {
        "visibility": "visible",
        "actor_bbox": list(target),
        "other_actors": [{"actor_id": "cyan", "bbox": list(other)}],
        "note": "Independent review of the red figure and the other cyan figure.",
    }


def _hidden(visibility="not_visible"):
    return {
        "visibility": visibility,
        "actor_bbox": None,
        "other_actors": [{"actor_id": "cyan", "bbox": list(RIGHT)}],
        "note": "The target is absent or cannot be independently identified in this frame.",
    }


def _labels(recording, path, reviews=None):
    document = create_actor_label_template(recording, path)
    document.update(
        domain="authored_fixture", reviewer="independent identity reviewer", subject="red person"
    )
    for local, review in (reviews or {}).items():
        document["frames"][local]["review"] = review
    _write(path, document)
    return document


def _score(case, labels, output, detector=None, **options):
    return benchmark_actor_recording(
        *case, labels, output, TwoColorDetector() if detector is None else detector, **options
    )


def test_template_inventories_every_frame_without_making_identity_claims(actor_case, tmp_path):
    recording, _ = actor_case
    path = tmp_path / "labels.json"
    labels = create_actor_label_template(recording, path)
    manifest = json.loads((recording / "manifest.json").read_text())
    assert json.loads(path.read_text()) == labels
    assert labels["schema_version"] == 1 and labels["kind"] == "recording_actor_labels"
    assert labels["session_id"] == manifest["session_id"]
    assert labels["source_sha256"] == manifest["source_sha256"]
    assert labels["domain"] is labels["reviewer"] is labels["subject"] is None
    assert [row["source_sequence"] for row in labels["frames"]] == list(range(41, 49))
    assert [row["timestamp_ms"] for row in labels["frames"]] == list(range(0, 400, 50))
    for row, frame in zip(labels["frames"], manifest["frames"], strict=True):
        assert row["review"] is None
        assert row["rgb_sha256"] == frame["rgb_sha256"]
        assert row["pts"] is None


def test_identity_is_scored_without_movement_and_all_inputs_are_preserved(actor_case, tmp_path):
    recording, replay = actor_case
    before = {path.name: path.read_bytes() for path in replay.iterdir()}
    rows = _rows(replay)
    assert all("movement" not in row for row in rows)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {local: _visible(local) for local in range(FRAME_COUNT)})
    output = tmp_path / "evaluation.json"
    report = _score(actor_case, labels, output)
    assert json.loads(output.read_text()) == report
    assert report["kind"] == "recording_actor_benchmark"
    assert report["evaluation_status"] == "evaluated"
    assert report["verification"]["actor_selection_recomputed"] is True
    assert "movement_recomputed" not in report["verification"]
    assert report["coverage"] == {
        "reviewed": 8,
        "unreviewed": 0,
        "visible": 8,
        "not_visible": 0,
        "uncertain": 0,
    }
    assert set(report["outcomes"]) == OUTCOMES
    assert report["outcomes"]["correct_target"] == 8
    assert sum(report["outcomes"].values()) == 8
    assert report["selection_status_counts"] == {"waiting": 0, "selected": 8, "invalidated": 0}
    metrics = report["metrics"]
    assert metrics["target_recall"] == metrics["visible_selection_coverage"] == 1.0
    assert metrics["visible_target_frames"] == metrics["correct_target_frames"] == 8
    assert report["continuity"]["eligible_adjacent_identity_pairs"] == 7
    assert report["continuity"]["reference_identity_changes"] == 0
    assert report["continuity"]["longest_correct_target_run_frames"] == 8
    assert report["owner_identity_verified"] is False
    assert report["read_only"] is True
    assert report["actionable"] is False
    assert {path.name: path.read_bytes() for path in replay.iterdir()} == before


def test_selected_actor_is_correct_even_when_movement_confidence_abstains(tmp_path):
    case = _case(tmp_path, movement_config=MovementConfig(min_actor_confidence=0.99))
    rows = _rows(case[1])
    assert all(row["actor_selection"]["status"] == "selected" for row in rows)
    assert all(row["movement"]["state"] == "unknown" for row in rows)
    labels = tmp_path / "labels.json"
    _labels(case[0], labels, {local: _visible(local) for local in range(FRAME_COUNT)})
    report = _score(case, labels, tmp_path / "evaluation.json")
    assert report["verification"]["movement_recomputed"] is True
    assert report["outcomes"]["correct_target"] == 8
    assert report["metrics"]["target_recall"] == 1.0


def test_stable_geometric_track_id_cannot_hide_a_changed_reference_person(swapped_case, tmp_path):
    rows = _rows(swapped_case[1])
    assert len({row["actor_selection"]["actor_track_id"] for row in rows}) == 1
    assert all(row["actor_selection"]["status"] == "selected" for row in rows)
    labels = tmp_path / "labels.json"
    _labels(
        swapped_case[0], labels, {local: _visible(local, swap_at=3) for local in range(FRAME_COUNT)}
    )
    report = _score(swapped_case, labels, tmp_path / "evaluation.json")
    assert report["outcomes"]["correct_target"] == 3
    assert report["outcomes"]["wrong_other_actor"] == 5
    assert report["metrics"]["target_recall"] == 3 / 8
    assert report["metrics"]["visible_selection_coverage"] == 1.0
    assert report["continuity"]["reference_identity_changes"] == 1
    assert report["continuity"]["eligible_adjacent_identity_pairs"] == 7
    assert report["continuity"]["longest_correct_target_run_frames"] == 3
    assert report["diagnostics"][2]["reference_match"]["actor_id"] == "target"
    assert report["diagnostics"][3]["reference_match"]["actor_id"] == "cyan"


@pytest.mark.parametrize("gap", ["unreviewed", "uncertain"])
def test_identity_continuity_never_bridges_unreviewed_or_uncertain_frames(
    swapped_case, tmp_path, gap
):
    labels = tmp_path / "labels.json"
    reviews = {local: _visible(local, swap_at=3) for local in range(FRAME_COUNT)}
    reviews[2] = None if gap == "unreviewed" else _hidden("uncertain")
    _labels(swapped_case[0], labels, reviews)
    report = _score(swapped_case, labels, tmp_path / "evaluation.json")
    assert report["continuity"]["reference_identity_changes"] == 0
    assert report["continuity"]["eligible_adjacent_identity_pairs"] == 5
    assert report["continuity"]["longest_correct_target_run_frames"] == 2
    assert report["diagnostics"][2]["outcome"] == (
        "unreviewed" if gap == "unreviewed" else "reference_uncertain"
    )
    assert report["metrics"]["visible_target_frames"] == 7
    assert report["metrics"]["target_recall"] == 2 / 7


def test_target_loss_abstention_stays_in_visible_recall_denominator_after_return(tmp_path):
    case = _case(tmp_path, missing_at=3)
    rows = _rows(case[1])
    assert all(row["actor_selection"]["status"] == "invalidated" for row in rows[3:])
    labels = tmp_path / "labels.json"
    reviews = {local: _visible(local) for local in range(FRAME_COUNT)}
    reviews[3] = _hidden()
    _labels(case[0], labels, reviews)
    report = _score(case, labels, tmp_path / "evaluation.json")
    assert report["outcomes"]["correct_target"] == 3
    assert report["outcomes"]["correct_abstention_not_visible"] == 1
    assert report["outcomes"]["missed_visible_target"] == 4
    assert report["metrics"]["visible_target_frames"] == 7
    assert report["metrics"]["target_recall"] == 3 / 7
    assert report["metrics"]["visible_selection_coverage"] == 3 / 7
    assert report["metrics"]["not_visible_selection_rate"] == 0.0
    assert report["selection_status_counts"] == {"waiting": 0, "selected": 3, "invalidated": 5}


def test_visible_targets_before_a_middle_anchor_remain_in_the_recall_denominator(
    actor_case, tmp_path
):
    recording, _ = actor_case
    selection = create_recording_actor_selection(
        recording,
        tmp_path / "later-selection.json",
        source_sequence=SOURCE_START + 2,
        bbox=tuple(LEFT),
        reviewer="independent later-frame choice",
        subject="red person",
    )
    replay = tmp_path / "later-replay"
    track_recording(
        recording, replay, TwoColorDetector(), actor_selection=selection, preview_max_frames=2
    )
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {local: _visible(local) for local in range(FRAME_COUNT)})
    report = _score((recording, replay), labels, tmp_path / "evaluation.json")
    assert report["selection_status_counts"] == {"waiting": 2, "selected": 6, "invalidated": 0}
    assert report["outcomes"]["missed_visible_target"] == 2
    assert report["outcomes"]["correct_target"] == 6
    assert report["metrics"]["visible_target_frames"] == 8
    assert report["metrics"]["target_recall"] == 6 / 8
    assert report["continuity"]["longest_correct_target_run_frames"] == 6
    assert report["continuity"]["longest_correct_target_interval"]["start_source_sequence"] == 43


def test_selected_other_person_when_target_is_absent_is_a_false_selection(tmp_path):
    case = _case(tmp_path, missing_at=3, selected_other=True)
    labels = tmp_path / "labels.json"
    _labels(case[0], labels, {3: _hidden()})
    report = _score(case, labels, tmp_path / "evaluation.json")
    assert report["outcomes"]["selected_when_not_visible"] == 1
    assert report["outcomes"]["unreviewed"] == 7
    assert report["metrics"]["not_visible_selection_rate"] == 1.0
    assert report["metrics"]["target_recall"] is None
    assert report["diagnostics"][3]["reference_match"]["actor_id"] == "cyan"


def test_localization_mismatch_and_multiple_reference_matches_are_distinct(actor_case, tmp_path):
    labels = tmp_path / "labels.json"
    missed = _visible()
    missed["actor_bbox"] = [60, 60, 72, 80]
    ambiguous = _visible()
    ambiguous["other_actors"] = [{"actor_id": "occluded-other", "bbox": [31, 60, 43, 80]}]
    _labels(actor_case[0], labels, {0: missed, 1: ambiguous})
    report = _score(actor_case, labels, tmp_path / "evaluation.json")
    assert report["outcomes"]["localization_mismatch"] == 1
    assert report["outcomes"]["ambiguous_reference_match"] == 1
    assert report["outcomes"]["correct_target"] == 0
    assert report["diagnostics"][0]["reference_match"]["status"] == "none"
    assert report["diagnostics"][1]["reference_match"]["status"] == "ambiguous"
    assert set(report["diagnostics"][1]["reference_match"]["matched_actor_ids"]) == {
        "target",
        "occluded-other",
    }
    assert report["metrics"]["target_recall"] == 0.0
    assert report["metrics"]["visible_selection_coverage"] == 1.0
    assert report["continuity"]["eligible_adjacent_identity_pairs"] == 0


@pytest.mark.parametrize("threshold,correct", [(0.3, True), (0.5, False)])
def test_reference_iou_threshold_changes_matching_not_inference(
    actor_case, tmp_path, threshold, correct
):
    labels = tmp_path / "labels.json"
    review = _visible()
    review["actor_bbox"] = [36, 60, 48, 80]
    _labels(actor_case[0], labels, {0: review})
    report = _score(actor_case, labels, tmp_path / "evaluation.json", actor_iou_threshold=threshold)
    assert report["actor_iou_threshold"] == threshold
    diagnostic = report["diagnostics"][0]
    target = next(
        value
        for value in diagnostic["reference_match"]["references"]
        if value["actor_id"] == "target"
    )
    assert target["iou"] == pytest.approx(1 / 3)
    assert diagnostic["outcome"] == ("correct_target" if correct else "localization_mismatch")


def test_empty_reviews_leave_metrics_undefined(actor_case, tmp_path):
    labels = tmp_path / "labels.json"
    _labels(actor_case[0], labels)
    report = _score(actor_case, labels, tmp_path / "evaluation.json")
    assert report["evaluation_status"] == "no_scorable_labels"
    assert report["coverage"] == {
        "reviewed": 0,
        "unreviewed": 8,
        "visible": 0,
        "not_visible": 0,
        "uncertain": 0,
    }
    assert report["outcomes"]["unreviewed"] == 8
    for name in (
        "target_recall",
        "visible_selection_coverage",
        "not_visible_selection_rate",
        "selection_coverage",
    ):
        assert report["metrics"][name] is None
    assert report["continuity"]["eligible_adjacent_identity_pairs"] == 0
    assert report["continuity"]["longest_correct_target_interval"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "source_hash",
        "session",
        "manifest_hash",
        "sequence",
        "source_sequence",
        "timestamp",
        "source_timestamp",
        "pts",
        "rgb",
        "reordered",
        "missing_frame",
        "duplicate_frame",
        "unknown_top",
        "unknown_review",
        "invalid_domain",
        "missing_subject",
        "bad_visibility",
        "missing_target_box",
        "hidden_target_box",
        "uncertain_without_note",
        "absent_without_note",
        "float_box",
        "boolean_box",
        "outside_box",
        "empty_box",
        "reserved_other_id",
        "duplicate_other_id",
        "invalid_other_id",
        "other_box_outside",
        "unknown_other_field",
    ],
)
def test_invalid_or_foreign_reference_labels_fail_before_output(actor_case, tmp_path, mutation):
    labels = tmp_path / "labels.json"
    document = _labels(actor_case[0], labels, {1: _visible()})
    row = document["frames"][1]
    review = row["review"]
    if mutation == "source_hash":
        document["source_sha256"] = "0" * 64
    elif mutation == "session":
        document["session_id"] = "0" * 64
    elif mutation == "manifest_hash":
        document["manifest_sha256"] = "0" * 64
    elif mutation in {"sequence", "source_sequence", "timestamp", "source_timestamp"}:
        key = {"timestamp": "timestamp_ms", "source_timestamp": "source_timestamp_ms"}.get(
            mutation, mutation
        )
        row[key] += 1
    elif mutation == "pts":
        row["pts"] = 3
    elif mutation == "rgb":
        row["rgb_sha256"] = "0" * 64
    elif mutation == "reordered":
        document["frames"].reverse()
    elif mutation == "missing_frame":
        document["frames"].pop()
    elif mutation == "duplicate_frame":
        document["frames"][2] = row
    elif mutation == "unknown_top":
        document["automatic"] = True
    elif mutation == "unknown_review":
        review["track_id"] = "track-1-1"
    elif mutation == "invalid_domain":
        document["domain"] = "production_ready"
    elif mutation == "missing_subject":
        document["subject"] = None
    elif mutation == "bad_visibility":
        review["visibility"] = True
    elif mutation == "missing_target_box":
        review["actor_bbox"] = None
    elif mutation == "hidden_target_box":
        review["visibility"] = "not_visible"
    elif mutation in {"uncertain_without_note", "absent_without_note"}:
        row["review"] = _hidden(
            "uncertain" if mutation == "uncertain_without_note" else "not_visible"
        )
        row["review"]["note"] = ""
    elif mutation in {"float_box", "boolean_box", "outside_box", "empty_box"}:
        review["actor_bbox"] = {
            "float_box": [30.0, 60, 42, 80],
            "boolean_box": [True, 60, 42, 80],
            "outside_box": [30, 60, 161, 80],
            "empty_box": [30, 60, 30, 80],
        }[mutation]
    elif mutation == "reserved_other_id":
        review["other_actors"][0]["actor_id"] = "target"
    elif mutation == "duplicate_other_id":
        review["other_actors"].append({"actor_id": "cyan", "bbox": [60, 60, 72, 80]})
    elif mutation == "invalid_other_id":
        review["other_actors"][0]["actor_id"] = "cyan/../../person"
    elif mutation == "other_box_outside":
        review["other_actors"][0]["bbox"] = [150, 60, 170, 80]
    elif mutation == "unknown_other_field":
        review["other_actors"][0]["automatic"] = True
    _write(labels, document)
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError):
        _score(actor_case, labels, output)
    assert not output.exists()


def test_rehashed_selected_id_forgery_is_recomputed_before_identity_scoring(actor_case, tmp_path):
    recording, original = actor_case
    replay = tmp_path / "forged-replay"
    shutil.copytree(original, replay)
    rows = _rows(replay)
    row = rows[3]
    other = next(track for track in row["belief"]["tracks"] if track["bbox"] == RIGHT)
    assert other["track_id"] != row["actor_selection"]["actor_track_id"]
    row["actor_selection"]["actor_track_id"] = other["track_id"]
    encoded = "".join(canonical_json(value) + "\n" for value in rows).encode("utf-8")
    (replay / "tracks.jsonl").write_bytes(encoded)
    report_path = replay / "report.json"
    report = json.loads(report_path.read_text())
    report["tracks_sha256"] = report["files"]["tracks.jsonl"] = hashlib.sha256(encoded).hexdigest()
    _write(report_path, report)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {3: _visible()})
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="trajectory mismatch"):
        _score((recording, replay), labels, output)
    assert not output.exists()


def test_review_mutation_during_reinference_cannot_publish_mixed_evidence(actor_case, tmp_path):
    labels = tmp_path / "labels.json"
    document = _labels(actor_case[0], labels, {0: _visible()})

    def change_review():
        document["frames"][0]["review"]["actor_bbox"] = list(RIGHT)
        _write(labels, document)

    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="changed|differs|mutation"):
        _score(actor_case, labels, output, TwoColorDetector(during_inference=change_review))
    assert not output.exists()


@pytest.mark.parametrize("changed_input", ["source_pixels", "replay_report"])
def test_source_or_replay_mutation_during_reinference_cannot_be_scored(
    actor_case, tmp_path, changed_input
):
    original_recording, original_replay = actor_case
    recording, replay = tmp_path / "recording", tmp_path / "replay"
    shutil.copytree(original_recording, recording)
    shutil.copytree(original_replay, replay)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {0: _visible()})

    def mutate_input():
        if changed_input == "source_pixels":
            path = recording / "frames/000000.png"
            with Image.open(path) as original:
                image = original.convert("RGB")
            image.putpixel((0, 0), (94, 95, 96))
            image.save(path)
            image.close()
        else:
            path = replay / "report.json"
            value = json.loads(path.read_text())
            value["frames_processed"] += 1
            _write(path, value)

    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError):
        _score((recording, replay), labels, output, TwoColorDetector(during_inference=mutate_input))
    assert not output.exists()


def test_plain_replay_without_an_explicit_target_cannot_claim_actor_evaluation(
    actor_case, tmp_path
):
    recording, _ = actor_case
    replay = tmp_path / "plain-replay"
    track_recording(recording, replay, TwoColorDetector(), preview_max_frames=2)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {0: _visible()})
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="actor|select"):
        _score((recording, replay), labels, output)
    assert not output.exists()


def test_template_and_benchmark_outputs_are_create_only_and_outside_inputs(actor_case, tmp_path):
    recording, replay = actor_case
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {0: _visible()})
    original_labels = labels.read_bytes()
    with pytest.raises(FileExistsError):
        create_actor_label_template(recording, labels)
    with pytest.raises(ValueError):
        create_actor_label_template(recording, recording / "new-labels.json")
    for output in (recording / "evaluation.json", replay / "evaluation.json", labels):
        with pytest.raises((ValueError, FileExistsError)):
            _score(actor_case, labels, output)
    output = tmp_path / "evaluation.json"
    _score(actor_case, labels, output)
    original_report = output.read_bytes()
    with pytest.raises(FileExistsError):
        _score(actor_case, labels, output)
    assert output.read_bytes() == original_report
    assert labels.read_bytes() == original_labels


@pytest.mark.parametrize("threshold", [True, False, 0, -0.5, 1.1, float("nan"), float("inf")])
def test_invalid_identity_thresholds_are_rejected(actor_case, tmp_path, threshold):
    labels = tmp_path / "labels.json"
    _labels(actor_case[0], labels, {0: _visible()})
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError):
        _score(actor_case, labels, output, actor_iou_threshold=threshold)
    assert not output.exists()


def test_original_nonuniform_pts_are_preserved_in_actor_reviews_and_diagnostics(tmp_path):
    av = pytest.importorskip("av")
    source = tmp_path / "variable.mp4"
    with av.open(str(source), "w", format="mp4") as writer:
        stream = writer.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for pts, shade in [(140, 20), (180, 75), (300, 130), (520, 200)]:
            image = Image.new("RGB", (64, 48), (shade,) * 3)
            frame = av.VideoFrame.from_image(image)
            image.close()
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    recording, replay = tmp_path / "recording", tmp_path / "replay"
    import_recording(
        source,
        recording,
        source_uri="urn:self:actor-evaluation-variable-clock",
        permission="self_authored",
        license="CC0",
        presentation_time=True,
        start_frame=1,
        max_frames=3,
    )
    selection = create_recording_actor_selection(
        recording,
        tmp_path / "selection.json",
        source_sequence=1,
        bbox=(20, 20, 30, 40),
        reviewer="clock-only fixture",
        subject="unobserved target; all identity labels unreviewed",
    )
    track_recording(
        recording, replay, TwoColorDetector(), actor_selection=selection, preview_max_frames=2
    )
    assert verify_recording_tracking(recording, replay, TwoColorDetector())[
        "actor_selection_recomputed"
    ]
    labels = tmp_path / "labels.json"
    template = _labels(recording, labels)
    report = _score((recording, replay), labels, tmp_path / "evaluation.json")
    assert [row["source_sequence"] for row in template["frames"]] == [1, 2, 3]
    assert [row["timestamp_ms"] for row in template["frames"]] == [0, 120, 340]
    assert [row["source_timestamp_ms"] for row in template["frames"]] == [180, 300, 520]
    assert [row["pts"] for row in report["diagnostics"]] == [
        row["pts"] for row in template["frames"]
    ]
    assert report["evaluation_status"] == "no_scorable_labels"
