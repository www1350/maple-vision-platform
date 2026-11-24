import json
from dataclasses import replace

import pytest

from maple_vision.contracts import Detection
from maple_vision.dataset import generate_dataset
from maple_vision.evaluation import EvaluationSample, benchmark_dataset, evaluate_predictions
from maple_vision.scene import ImageFrame


def _box(x: float = 0, confidence: float = 1, class_id: str = "player") -> Detection:
    return Detection(class_id, (x, 0, x + 10, 10), confidence)


def _evaluate(truth, predictions, **kwargs):
    return evaluate_predictions([EvaluationSample(tuple(truth), tuple(predictions))], **kwargs)


def test_perfect_predictions_have_perfect_precision_recall_and_ap() -> None:
    result = _evaluate([_box()], [_box(confidence=0.9)])
    assert result["micro"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "ground_truth": 1,
        "predictions": 1,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }
    assert result["map50"] == 1
    assert result["ap_classes_with_ground_truth"] == 1
    assert result["per_class"]["monster"]["ap50"] is None


def test_duplicate_prediction_counts_as_false_positive() -> None:
    result = _evaluate([_box()], [_box(confidence=0.9), _box(confidence=0.8)])
    assert result["micro"]["tp"] == 1
    assert result["micro"]["fp"] == 1
    assert result["micro"]["precision"] == 0.5
    # A lower-ranked duplicate does not reduce an already perfect AP envelope.
    assert result["map50"] == 1


def test_confidence_ranking_not_input_order_controls_ap() -> None:
    wrong = _box(x=50, confidence=0.9)
    right = _box(confidence=0.8)
    assert _evaluate([_box()], [right, wrong])["map50"] == 0.5
    assert _evaluate([_box()], [wrong, right])["map50"] == 0.5


def test_known_101_point_interpolation_example() -> None:
    result = _evaluate(
        [_box(), _box(20)],
        [_box(confidence=0.9), _box(50, confidence=0.8), _box(20, confidence=0.7)],
    )
    # Recall 0..0.50 (51 points) has envelope precision1; remaining50 points get2/3.
    assert result["map50"] == pytest.approx((51 + 50 * 2 / 3) / 101)


def test_missing_predictions_count_false_negatives_with_zero_ap() -> None:
    result = _evaluate([_box()], [])
    assert result["micro"]["fn"] == 1
    assert result["micro"]["recall"] == 0
    assert result["micro"]["precision"] is None
    assert result["micro"]["f1"] == 0
    assert result["map50"] == 0


def test_no_ground_truth_does_not_suppress_false_positives() -> None:
    result = _evaluate([], [_box()])
    assert result["micro"]["fp"] == 1
    assert result["micro"]["precision"] == 0
    assert result["micro"]["recall"] is None
    assert result["per_class"]["player"]["ap50"] is None
    assert result["map50"] is None


def test_empty_measurement_has_no_fabricated_quality_scores() -> None:
    result = evaluate_predictions([])
    assert result["images"] == 0
    assert result["map50"] is None
    assert result["micro"]["precision"] is None
    assert result["micro"]["recall"] is None
    assert result["micro"]["f1"] is None


def test_wrong_class_produces_a_false_positive_and_a_false_negative() -> None:
    result = _evaluate([_box()], [_box(class_id="monster")])
    assert (result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]) == (0, 1, 1)


def test_unknown_predicted_class_is_counted_not_silently_dropped() -> None:
    result = _evaluate([_box()], [_box(class_id="unknown_label")])
    assert result["micro"]["fp"] == 1
    assert result["unrecognized_prediction_classes"] == ["unknown_label"]
    assert result["per_class"]["unknown_label"]["ap50"] is None


def test_prediction_cannot_match_ground_truth_in_another_image() -> None:
    result = evaluate_predictions(
        [
            EvaluationSample((_box(),), ()),
            EvaluationSample((), (_box(),)),
        ]
    )
    assert (result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]) == (0, 1, 1)


def test_one_prediction_cannot_match_two_overlapping_targets() -> None:
    result = _evaluate([_box(), _box(1)], [_box(0.5)])
    assert result["micro"]["tp"] == 1
    assert result["micro"]["fn"] == 1
    assert result["map50"] == pytest.approx(51 / 101)


def test_iou_threshold_is_inclusive_and_nonhalf_threshold_not_named_ap50() -> None:
    truth = Detection("player", (0, 0, 20, 10), 1)
    prediction = Detection("player", (0, 0, 10, 10), 1)
    assert _evaluate([truth], [prediction], iou_threshold=0.5)["micro"]["tp"] == 1
    result = _evaluate([truth], [prediction], iou_threshold=0.50001)
    assert result["micro"]["tp"] == 0
    assert result["map50"] is None and result["map_at_iou"] == 0


def test_confidence_cutoff_is_inclusive_and_reported() -> None:
    assert _evaluate([_box()], [_box(confidence=0.25)])["micro"]["tp"] == 1
    result = _evaluate([_box()], [_box(confidence=0.2499)])
    assert result["micro"]["fn"] == 1
    assert result["predictions_below_confidence_threshold"] == 1


@pytest.mark.parametrize("threshold", [-1, float("nan"), 1.01, True])
def test_invalid_thresholds_fail(threshold) -> None:
    with pytest.raises(ValueError):
        evaluate_predictions([], confidence_threshold=threshold)


def test_zero_iou_would_match_disjoint_boxes_so_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions([], iou_threshold=0)


def test_invalid_truth_class_and_mutable_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        _evaluate([_box(class_id="unregistered")], [])
    with pytest.raises(ValueError, match="immutable"):
        replace(EvaluationSample((), ()), predictions=[_box()])


class _CountingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: ImageFrame) -> tuple[Detection, ...]:
        self.calls += 1
        return ()


def test_benchmark_records_real_split_counts_warmup_and_persistent_report(tmp_path) -> None:
    data = tmp_path / "dataset"
    generate_dataset(data, sessions=3, frames_per_session=2, seed=1)
    detector = _CountingDetector()
    output = tmp_path / "reports" / "benchmark.json"
    result = benchmark_dataset(data, output, detector)
    assert detector.calls == 4  # 2 warmup + 2 measured
    assert result["metrics"]["images"] == 2
    assert result["metrics"]["micro"]["fn"] > 0
    assert result["metrics"]["micro"]["tp"] == 0
    assert result["metrics"]["map50"] == 0
    assert result["latency_ms"]["warmup_calls_excluded"] == 2
    assert result["latency_ms"]["samples"] == 2
    assert 0 <= result["latency_ms"]["p50"] <= result["latency_ms"]["p95"]
    assert len(result["dataset_manifest_sha256"]) == 64
    assert result["scope"] == "synthetic_fixture"
    assert result["model_training_overlap_verified"] is False
    saved = json.loads(output.read_text())
    assert saved == {key: value for key, value in result.items() if key != "output"}
    with pytest.raises(FileExistsError):
        benchmark_dataset(data, output, detector)
    assert detector.calls == 4


def test_default_pixel_benchmark_is_executable_without_yolo(tmp_path) -> None:
    data = tmp_path / "dataset"
    generate_dataset(data, sessions=3, frames_per_session=1, seed=1)
    result = benchmark_dataset(data, tmp_path / "pixels.json")
    assert result["detector"]["method"] == "fixed_palette_connected_components"
    assert result["metrics"]["micro"]["tp"] > 0
    assert result["metrics"]["map50"] > 0


def test_invalid_dataset_blocks_inference_and_preserves_output_absence(tmp_path) -> None:
    data = tmp_path / "dataset"
    generate_dataset(data, sessions=3, frames_per_session=1)
    (data / "data.yaml").write_text("corrupt")
    detector = _CountingDetector()
    output = tmp_path / "benchmark.json"
    with pytest.raises(ValueError):
        benchmark_dataset(data, output, detector)
    assert detector.calls == 0 and not output.exists()
