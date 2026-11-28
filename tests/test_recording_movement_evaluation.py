"""Independent pixel and annotation probes for movement-candidate evaluation.

The marker scene is authored independently of the estimator. Its feet rise four
pixels per frame, reach the platform at local frame 20, then remain there. Human
review values below are chosen from that geometry or deliberately changed to
probe scoring; they are never copied from the generated movement predictions.
These tests establish engineering behavior, not game-domain model accuracy.
"""

import hashlib
import json
import shutil
from fractions import Fraction

import pytest
from PIL import Image
from test_recording_movement import GeometryMarkerDetector, authored_recording

from maple_vision.catalog import canonical_json
from maple_vision.movement_hypotheses import MovementConfig
from maple_vision.recording_movement_evaluation import (
    benchmark_movement_recording,
    create_movement_label_template,
)
from maple_vision.recording_tracking import track_recording
from maple_vision.recordings import import_recording

PREDICATES = ("rope_climb_visible", "arrival_after_climb_visible")


class ObservedGeometryDetector(GeometryMarkerDetector):
    """Inference receives source pixels only, with an optional detector dropout."""

    def __init__(self, *, missing_local_frame=None, during_inference=None):
        self.missing_local_frame = missing_local_frame
        self.during_inference = during_inference
        self.seen = []

    def detect(self, frame):
        self.seen.append((frame.sequence, frame.timestamp_ms))
        assert not hasattr(frame, "review")
        assert not hasattr(frame, "labels")
        assert not hasattr(frame, "ground_truth")
        if self.during_inference is not None:
            callback, self.during_inference = self.during_inference, None
            callback()
        if frame.sequence == self.missing_local_frame:
            return ()
        return super().detect(frame)


@pytest.fixture(scope="module")
def movement_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("movement-evaluation-pixels")
    recording = authored_recording(root)
    replay = root / "replay"
    track_recording(
        recording,
        replay,
        ObservedGeometryDetector(),
        movement_config=MovementConfig(),
        preview_max_frames=2,
    )
    return recording, replay


def _write(path, document):
    path.write_bytes((json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))


def _actor_box(sequence):
    feet = 130 - 4 * min(sequence, 20)
    return [75, feet - 16, 85, feet]


def _review(sequence, climb, arrival, *, bbox=None):
    return {
        "actor_bbox": _actor_box(sequence) if bbox is None else bbox,
        "rope_climb_visible": climb,
        "arrival_after_climb_visible": arrival,
        "note": "Independent authored geometry review; ownership is not asserted.",
    }


def _labels(recording, output, reviews=None):
    document = create_movement_label_template(recording, output)
    document.update(
        domain="authored_fixture", reviewer="independent pixel review", subject="magenta figure"
    )
    for sequence, review in (reviews or {}).items():
        document["frames"][sequence]["review"] = review
    _write(output, document)
    return document


def _score(case, labels, output, detector=None, **options):
    recording, replay = case
    return benchmark_movement_recording(
        recording,
        replay,
        labels,
        output,
        ObservedGeometryDetector() if detector is None else detector,
        **options,
    )


def test_template_binds_every_imported_frame_and_preserves_source_offset(movement_case, tmp_path):
    recording, _ = movement_case
    output = tmp_path / "labels.json"
    template = create_movement_label_template(recording, output)
    manifest = json.loads((recording / "manifest.json").read_text())

    assert json.loads(output.read_text()) == template
    assert template["kind"] == "recording_movement_labels"
    assert template["schema_version"] == 1
    assert template["session_id"] == manifest["session_id"]
    assert template["source_sha256"] == manifest["source_sha256"]
    assert (
        template["manifest_sha256"]
        == hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    )
    assert template["domain"] is template["reviewer"] is template["subject"] is None
    frames = template["frames"]
    assert len(frames) == 25
    assert [row["sequence"] for row in frames] == list(range(25))
    assert [row["source_sequence"] for row in frames] == list(range(17, 42))
    assert [row["timestamp_ms"] for row in frames] == [50 * index for index in range(25)]
    assert [row["source_timestamp_ms"] for row in frames] == [50 * index for index in range(17, 42)]
    assert [row["rgb_sha256"] for row in frames] == [
        row["rgb_sha256"] for row in manifest["frames"]
    ]
    assert all(row["pts"] is None and row["review"] is None for row in frames)


def test_full_pixel_chain_scores_independently_reviewed_geometry(movement_case, tmp_path):
    recording, replay = movement_case
    labels = tmp_path / "labels.json"
    reviews = {index: _review(index, 1 <= index < 20, 20 <= index < 25) for index in range(25)}
    _labels(recording, labels, reviews)
    before = {path.name: path.read_bytes() for path in replay.iterdir()}
    detector = ObservedGeometryDetector()
    output = tmp_path / "evaluation.json"
    report = _score(movement_case, labels, output, detector)

    assert json.loads(output.read_text()) == report
    assert report["kind"] == "recording_movement_benchmark"
    assert report["declared_domain"] == "authored_fixture"
    assert report["frames"] == 25
    assert report["coverage"] == {
        "reviewed": 25,
        "unreviewed": 0,
        "reviewed_uncertain": 0,
        "reviewed_with_decidable_predicate": 25,
    }
    assert detector.seen == [(index, 50 * index) for index in range(25)]
    assert report["verification"]["trajectory_recomputed"] is True
    assert report["verification"]["movement_recomputed"] is True
    assert report["read_only"] is True
    assert report["action_execution"] == "none"
    assert report["bindings"]["labels_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert (
        report["bindings"]["tracking_report_sha256"]
        == hashlib.sha256((replay / "report.json").read_bytes()).hexdigest()
    )
    assert (
        report["bindings"]["tracks_sha256"]
        == hashlib.sha256((replay / "tracks.jsonl").read_bytes()).hexdigest()
    )
    assert "overall_accuracy" not in report
    assert report["predicates"]["rope_climb_visible"]["positives"] == 19
    assert report["predicates"]["arrival_after_climb_visible"]["positives"] == 5
    assert len(report["diagnostics"]) == 25
    for name in PREDICATES:
        metrics = report["predicates"][name]
        assert metrics["false_negatives"] == metrics["positives"] - metrics["true_positives"]
        assert metrics["negatives"] == (
            metrics["true_negatives"] + metrics["false_positives"] + metrics["abstained_negative"]
        )
        assert metrics["decision_coverage"] == pytest.approx(
            (25 - metrics["abstained_positive"] - metrics["abstained_negative"]) / 25
        )
        assert metrics["recall"] == metrics["true_positives"] / metrics["positives"]
    assert {path.name: path.read_bytes() for path in replay.iterdir()} == before


def test_sparse_reviews_keep_unreviewed_uncertain_and_unknown_distinct(movement_case, tmp_path):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    reviews = {
        0: _review(0, False, False),
        10: _review(10, True, False),
        24: _review(24, False, True),
        8: {
            "actor_bbox": None,
            "rope_climb_visible": None,
            "arrival_after_climb_visible": None,
            "note": "The selected figure is occluded; no predicate can be decided.",
        },
    }
    _labels(recording, labels, reviews)
    report = _score(movement_case, labels, tmp_path / "evaluation.json")

    assert report["coverage"] == {
        "reviewed": 4,
        "unreviewed": 21,
        "reviewed_uncertain": 1,
        "reviewed_with_decidable_predicate": 3,
    }
    for name in PREDICATES:
        metrics = report["predicates"][name]
        assert metrics["positives"] == 1
        assert metrics["negatives"] == 2
        assert metrics["true_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_positives"] == metrics["false_negatives"] == 0
        assert metrics["abstained_positive"] == 0
        assert metrics["abstained_negative"] == 1
        assert metrics["unreviewed"] == 21
        assert metrics["uncertain"] == 1
        assert metrics["decision_coverage"] == pytest.approx(2 / 3)
        assert metrics["precision"] == metrics["recall"] == 1.0
    diagnostics = report["diagnostics"]
    assert diagnostics[0]["predicted_state"] == "unknown"
    assert diagnostics[0]["outcomes"]["rope_climb_visible"] == "abstained_negative"
    assert diagnostics[8]["outcomes"]["rope_climb_visible"] == "uncertain"
    assert diagnostics[9]["outcomes"]["rope_climb_visible"] == "unreviewed"
    assert diagnostics[10]["outcomes"]["rope_climb_visible"] == "tp"
    assert diagnostics[24]["outcomes"]["arrival_after_climb_visible"] == "tp"


def test_detector_dropout_on_visible_ascent_stays_in_positive_recall_denominator(
    movement_case, tmp_path
):
    recording, _ = movement_case
    replay = tmp_path / "dropout-replay"
    track_recording(
        recording,
        replay,
        ObservedGeometryDetector(missing_local_frame=10),
        movement_config=MovementConfig(),
        preview_max_frames=2,
    )
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {10: _review(10, True, False)})
    report = _score(
        (recording, replay),
        labels,
        tmp_path / "evaluation.json",
        ObservedGeometryDetector(missing_local_frame=10),
    )

    metrics = report["predicates"]["rope_climb_visible"]
    assert metrics["positives"] == metrics["false_negatives"] == 1
    assert metrics["abstained_positive"] == 1
    assert metrics["true_positives"] == 0
    assert metrics["recall"] == 0.0
    assert metrics["precision"] is None
    assert metrics["decision_coverage"] == 0.0
    assert report["diagnostics"][10]["predicted_state"] == "unknown"
    negative = report["predicates"]["arrival_after_climb_visible"]
    assert negative["abstained_negative"] == 1
    assert negative["true_negatives"] == negative["false_positives"] == 0


def test_decided_negative_is_a_miss_while_other_predicate_can_remain_uncertain(
    movement_case, tmp_path
):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    # The drawn figure has already risen 4 px in this frame. The estimator has
    # not accumulated a climbing candidate yet; the independent positive stays.
    _labels(recording, labels, {1: _review(1, True, None)})
    report = _score(movement_case, labels, tmp_path / "evaluation.json")

    climb = report["predicates"]["rope_climb_visible"]
    assert climb["positives"] == climb["false_negatives"] == 1
    assert climb["abstained_positive"] == 0
    assert climb["decision_coverage"] == 1.0
    assert climb["recall"] == 0.0
    assert report["diagnostics"][1]["outcomes"]["rope_climb_visible"] == "fn"
    arrival = report["predicates"]["arrival_after_climb_visible"]
    assert arrival["uncertain"] == 1
    assert arrival["unreviewed"] == 24
    assert arrival["positives"] == arrival["negatives"] == 0
    assert arrival["decision_coverage"] is None
    assert report["coverage"]["reviewed_with_decidable_predicate"] == 1
    assert report["coverage"]["reviewed_uncertain"] == 0


def test_wrong_actor_candidates_are_exposed_and_cannot_earn_positive_or_negative_credit(
    movement_case, tmp_path
):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    other_figure = [250, 210, 260, 226]
    _labels(
        recording,
        labels,
        {
            10: _review(10, True, False, bbox=other_figure),
            24: _review(24, False, True, bbox=other_figure),
        },
    )
    report = _score(movement_case, labels, tmp_path / "evaluation.json")

    assert report["actor_alignment"]["mismatched"] == 2
    assert report["actor_alignment"]["wrong_actor_positive_candidates"] == 2
    for index in (10, 24):
        assert report["diagnostics"][index]["actor_match"]["status"] == "mismatched"
        assert report["diagnostics"][index]["actor_match"]["iou"] == 0.0
    for name in PREDICATES:
        metrics = report["predicates"][name]
        assert metrics["true_positives"] == metrics["true_negatives"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == metrics["abstained_positive"] == 1
        assert metrics["abstained_negative"] == 1
        assert metrics["decision_coverage"] == metrics["recall"] == 0.0


def test_reviewer_values_change_scores_without_changing_pixel_predictions(movement_case, tmp_path):
    recording, _ = movement_case
    positive_labels, negative_labels = tmp_path / "positive.json", tmp_path / "negative.json"
    _labels(
        recording,
        positive_labels,
        {10: _review(10, True, False), 24: _review(24, False, True)},
    )
    # An intentionally different reviewer judgment probes scorer independence.
    # It is not a second claim about the scene's physical ground truth.
    _labels(
        recording,
        negative_labels,
        {10: _review(10, False, False), 24: _review(24, False, False)},
    )
    positive_detector, negative_detector = ObservedGeometryDetector(), ObservedGeometryDetector()
    positive = _score(
        movement_case, positive_labels, tmp_path / "positive-score.json", positive_detector
    )
    negative = _score(
        movement_case, negative_labels, tmp_path / "negative-score.json", negative_detector
    )

    assert positive_detector.seen == negative_detector.seen
    for first, second in zip(positive["diagnostics"], negative["diagnostics"], strict=True):
        for key in ("predicted_state", "predicted_reasons", "actor_match"):
            assert first[key] == second[key]
    for name in PREDICATES:
        assert positive["predicates"][name]["true_positives"] == 1
        assert negative["predicates"][name]["false_positives"] == 1
        assert negative["predicates"][name]["true_negatives"] == 1
        assert negative["predicates"][name]["precision"] == 0.0
        assert negative["predicates"][name]["recall"] is None


@pytest.mark.parametrize("threshold,matched", [(0.3, True), (0.5, False)])
def test_actor_iou_threshold_applies_to_selected_subject(
    movement_case, tmp_path, threshold, matched
):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    shifted = _actor_box(10)
    shifted[0] += 5
    shifted[2] += 5
    _labels(recording, labels, {10: _review(10, True, False, bbox=shifted)})
    report = _score(
        movement_case, labels, tmp_path / "evaluation.json", actor_iou_threshold=threshold
    )

    assert report["actor_iou_threshold"] == threshold
    diagnostic = report["diagnostics"][10]
    assert diagnostic["actor_match"]["iou"] == pytest.approx(1 / 3)
    assert diagnostic["actor_match"]["status"] == ("matched" if matched else "mismatched")
    assert diagnostic["outcomes"]["rope_climb_visible"] == (
        "tp" if matched else "abstained_positive"
    )


def test_empty_reviews_have_undefined_metrics_instead_of_zero_accuracy(movement_case, tmp_path):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    _labels(recording, labels)
    report = _score(movement_case, labels, tmp_path / "evaluation.json")

    assert report["coverage"]["reviewed"] == 0
    for name in PREDICATES:
        metrics = report["predicates"][name]
        assert metrics["unreviewed"] == 25
        assert metrics["positives"] == metrics["negatives"] == 0
        assert metrics["precision"] is metrics["recall"] is metrics["decision_coverage"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "source_hash",
        "manifest_hash",
        "wrong_offset",
        "wrong_clock",
        "wrong_pts",
        "wrong_pixels",
        "missing_frame",
        "reordered_frames",
        "duplicate_frame",
        "unknown_field",
        "invalid_domain",
        "missing_subject",
        "integer_boolean",
        "both_true",
        "missing_actor",
        "out_of_frame_actor",
        "uncertainty_without_reason",
    ],
)
def test_malformed_or_foreign_reviews_fail_before_publishing(movement_case, tmp_path, mutation):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    document = _labels(recording, labels, {10: _review(10, True, False)})
    entry, review = document["frames"][10], document["frames"][10]["review"]
    if mutation == "source_hash":
        document["source_sha256"] = "0" * 64
    elif mutation == "manifest_hash":
        document["manifest_sha256"] = "0" * 64
    elif mutation == "wrong_offset":
        entry["source_sequence"] += 1
    elif mutation == "wrong_clock":
        entry["source_timestamp_ms"] += 1
    elif mutation == "wrong_pts":
        entry["pts"] = 300
    elif mutation == "wrong_pixels":
        entry["rgb_sha256"] = "0" * 64
    elif mutation == "missing_frame":
        document["frames"].pop()
    elif mutation == "reordered_frames":
        document["frames"].reverse()
    elif mutation == "duplicate_frame":
        document["frames"][11] = document["frames"][10]
    elif mutation == "unknown_field":
        review["automatic_prediction"] = True
    elif mutation == "invalid_domain":
        document["domain"] = "production_ready"
    elif mutation == "missing_subject":
        document["subject"] = None
    elif mutation == "integer_boolean":
        review["rope_climb_visible"] = 1
    elif mutation == "both_true":
        review["arrival_after_climb_visible"] = True
    elif mutation == "missing_actor":
        review["actor_bbox"] = None
    elif mutation == "out_of_frame_actor":
        review["actor_bbox"] = [0, 0, 321, 257]
    else:
        review.update(
            actor_bbox=None, rope_climb_visible=None, arrival_after_climb_visible=None, note=""
        )
    _write(labels, document)
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError):
        _score(movement_case, labels, output)
    assert not output.exists()


def test_labels_changed_during_reinference_cannot_publish_a_mixed_report(movement_case, tmp_path):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    document = _labels(recording, labels, {10: _review(10, True, False)})

    def mutate_labels():
        document["frames"][10]["review"]["rope_climb_visible"] = False
        _write(labels, document)

    output = tmp_path / "evaluation.json"
    detector = ObservedGeometryDetector(during_inference=mutate_labels)
    with pytest.raises(ValueError, match="changed|differs|mutation"):
        _score(movement_case, labels, output, detector)
    assert not output.exists()


@pytest.mark.parametrize("target", ["already_read_source_pixels", "replay_report"])
def test_source_or_replay_mutation_during_inference_is_rejected(movement_case, tmp_path, target):
    original_recording, original_replay = movement_case
    recording, replay = tmp_path / "recording", tmp_path / "replay"
    shutil.copytree(original_recording, recording)
    shutil.copytree(original_replay, replay)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {10: _review(10, True, False)})

    def mutate_input():
        if target == "already_read_source_pixels":
            path = recording / "frames/000000.png"
            with Image.open(path) as original:
                image = original.copy()
            image.putpixel((0, 0), (1, 2, 3))
            image.save(path)
            image.close()
        else:
            path = replay / "report.json"
            document = json.loads(path.read_text())
            document["frames_processed"] += 1
            _write(path, document)

    output = tmp_path / "evaluation.json"
    detector = ObservedGeometryDetector(during_inference=mutate_input)
    with pytest.raises(ValueError):
        _score((recording, replay), labels, output, detector)
    assert not output.exists()


def test_rehashed_forged_movement_trace_is_recomputed_before_scoring(movement_case, tmp_path):
    recording, original = movement_case
    replay = tmp_path / "forged-replay"
    shutil.copytree(original, replay)
    rows = [json.loads(line) for line in (replay / "tracks.jsonl").read_text().splitlines()]
    rows[10]["movement"]["state"] = "platform_arrival_candidate"
    encoded = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    (replay / "tracks.jsonl").write_bytes(encoded)
    report_path = replay / "report.json"
    report = json.loads(report_path.read_text())
    report["tracks_sha256"] = report["files"]["tracks.jsonl"] = hashlib.sha256(encoded).hexdigest()
    _write(report_path, report)
    labels = tmp_path / "labels.json"
    _labels(recording, labels, {10: _review(10, True, False)})
    output = tmp_path / "evaluation.json"

    with pytest.raises(ValueError, match="trajectory mismatch"):
        _score((recording, replay), labels, output)
    assert not output.exists()


def test_plain_tracking_without_movement_cannot_be_scored(movement_case, tmp_path):
    recording, _ = movement_case
    replay = tmp_path / "plain-replay"
    track_recording(recording, replay, ObservedGeometryDetector(), preview_max_frames=2)
    labels = tmp_path / "labels.json"
    _labels(recording, labels)
    output = tmp_path / "evaluation.json"

    with pytest.raises(ValueError, match="movement"):
        _score((recording, replay), labels, output)
    assert not output.exists()


def test_nonuniform_video_clock_keeps_original_pts_and_local_time_distinct(tmp_path):
    av = pytest.importorskip("av")
    source = tmp_path / "nonuniform.mp4"
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
    recording = tmp_path / "recording"
    import_recording(
        source,
        recording,
        source_uri="urn:self:movement-evaluation-nonuniform-clock",
        permission="self_authored",
        license="CC0",
        presentation_time=True,
        start_frame=1,
        max_frames=3,
    )
    replay = tmp_path / "replay"
    track_recording(
        recording,
        replay,
        ObservedGeometryDetector(),
        movement_config=MovementConfig(),
        preview_max_frames=2,
    )
    labels = tmp_path / "labels.json"
    template = _labels(recording, labels)
    report = _score((recording, replay), labels, tmp_path / "evaluation.json")
    manifest = json.loads((recording / "manifest.json").read_text())

    assert [row["source_sequence"] for row in template["frames"]] == [1, 2, 3]
    assert [row["timestamp_ms"] for row in template["frames"]] == [0, 120, 340]
    assert [row["source_timestamp_ms"] for row in template["frames"]] == [180, 300, 520]
    clock = Fraction(manifest["video_clock"]["numerator"], manifest["video_clock"]["denominator"])
    assert [row["pts"] * clock for row in template["frames"]] == [
        Fraction(9, 50),
        Fraction(3, 10),
        Fraction(13, 25),
    ]
    assert [row["pts"] for row in report["diagnostics"]] == [
        row["pts"] for row in template["frames"]
    ]


def test_templates_and_scores_are_create_only_and_cannot_modify_source(movement_case, tmp_path):
    recording, replay = movement_case
    labels = tmp_path / "labels.json"
    _labels(recording, labels)
    original = labels.read_bytes()
    with pytest.raises(FileExistsError):
        create_movement_label_template(recording, labels)
    assert labels.read_bytes() == original
    with pytest.raises(ValueError):
        create_movement_label_template(recording, recording / "labels.json")
    for unsafe in (recording / "evaluation.json", replay / "evaluation.json", labels):
        with pytest.raises((ValueError, FileExistsError)):
            _score(movement_case, labels, unsafe)
    assert labels.read_bytes() == original


@pytest.mark.parametrize("threshold", [False, True, 0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_actor_matching_thresholds_are_rejected(movement_case, tmp_path, threshold):
    recording, _ = movement_case
    labels = tmp_path / "labels.json"
    _labels(recording, labels)
    with pytest.raises(ValueError):
        _score(movement_case, labels, tmp_path / "evaluation.json", actor_iou_threshold=threshold)
