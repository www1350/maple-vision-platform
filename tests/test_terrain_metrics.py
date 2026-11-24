import json
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.contracts import Detection
from maple_vision.dataset import generate_dataset
from maple_vision.evaluation import EvaluationSample, benchmark_dataset
from maple_vision.onnx_detector import OnnxDetector
from maple_vision.terrain_metrics import evaluate_terrain_geometry
from maple_vision.vision import PixelPerception


def _box(box, confidence=1.0, class_id="platform"):
    return Detection(class_id, box, confidence)


def _evaluate(targets, predictions, size=(320, 192), **kwargs):
    return evaluate_terrain_geometry(
        [EvaluationSample(tuple(targets), tuple(predictions))], [size], **kwargs
    )


def test_half_width_box_passes_iou50_but_endpoint_errors_expose_unusable_floor():
    result = _evaluate([_box((0, 168, 320, 172))], [_box((0, 168, 160, 172), 0.9)])
    floor = result["full_width_floor"]
    assert (floor["ground_truth"], floor["tp"], floor["fn"], floor["recall"]) == (1, 1, 0, 1)
    assert floor["matched_iou"]["mean"] == 0.5
    assert floor["absolute_error_pixels"]["right"]["p95"] == 160
    assert floor["absolute_error_pixels"]["top"]["p95"] == 0
    assert floor["fp"] is None


def test_too_thick_floor_can_match_but_top_edge_error_is_reported_in_original_pixels():
    floor = _evaluate([_box((0, 168, 320, 172))], [_box((0, 166, 320, 174))])["full_width_floor"]
    assert floor["tp"] == 1
    assert floor["matched_iou"]["mean"] == 0.5
    assert floor["absolute_error_pixels"]["top"]["p95"] == 2
    assert floor["absolute_error_pixels"]["height"]["p95"] == 4


def test_platform_matching_precedes_floor_stratification_and_never_steals_a_better_match():
    result = _evaluate(
        [_box((0, 100, 320, 104)), _box((0, 100, 160, 104))],
        [_box((0, 100, 160, 104), 0.8)],
    )
    assert result["platform"]["tp"] == 1
    assert result["full_width_floor"]["tp"] == 0
    assert result["full_width_floor"]["fn"] == 1
    assert result["full_width_floor"]["absolute_error_pixels"]["top"]["p95"] is None


def test_missing_targets_remain_in_recall_and_are_not_zero_error_measurements():
    target = _box((0, 168, 320, 172))
    samples = [EvaluationSample((target,), (target,)), EvaluationSample((target,), ())]
    floor = evaluate_terrain_geometry(samples, [(320, 192)] * 2)["full_width_floor"]
    assert floor["recall"] == 0.5
    assert floor["absolute_error_pixels"]["top"] == {
        "samples": 1,
        "mean": 0,
        "p50": 0,
        "p95": 0,
        "max": 0,
    }
    assert floor["missing_examples"][0]["image_index"] == 1


def test_confidence_order_duplicate_predictions_and_class_matching():
    target = _box((0, 168, 320, 172))
    result = _evaluate(
        [target, _box((40, 80, 44, 120), class_id="ladder")],
        [
            _box((0, 169, 320, 172), 0.7),
            _box(target.bbox, 0.9),
            _box((40, 80, 44, 120), 0.2, "ladder"),
            _box(target.bbox, 0.99, "monster"),
        ],
    )
    assert result["platform"]["tp"] == result["platform"]["fp"] == 1
    assert result["full_width_floor"]["absolute_error_pixels"]["top"]["p95"] == 0
    assert result["ladder"]["recall"] == 0
    assert result["ladder"]["fp"] == 0


def test_equal_confidence_preserves_input_order():
    result = _evaluate(
        [_box((0, 168, 320, 172))], [_box((0, 169, 320, 172), 0.8), _box((0, 168, 320, 172), 0.8)]
    )
    assert result["full_width_floor"]["absolute_error_pixels"]["top"]["p95"] == 1
    assert result["full_width_floor"]["worst_matched_examples"][0]["prediction_index"] == 0


def test_original_width_and_linear_quantiles_not_fixed_fixture_dimensions():
    samples = [
        EvaluationSample((_box((0, 20, 100, 24)),), (_box((0, 20, 100 - e, 24)),))
        for e in (0, 2, 4)
    ]
    floor = evaluate_terrain_geometry(samples, [(100, 40)] * 3)["full_width_floor"]
    assert floor["ground_truth"] == 3
    assert floor["absolute_error_pixels"]["right"]["p50"] == 2
    assert floor["absolute_error_pixels"]["right"]["p95"] == pytest.approx(3.8)


def test_no_floor_is_unknown_and_narrow_platforms_do_not_imply_a_floor():
    result = _evaluate([_box((1, 168, 319, 172))], [_box((1, 168, 319, 172))])
    assert result["platform"]["recall"] == 1
    assert result["full_width_floor"]["ground_truth"] == 0
    assert result["full_width_floor"]["recall"] is None
    assert result["full_width_floor"]["matched_iou"]["samples"] == 0
    assert evaluate_terrain_geometry([], [])["full_width_floor"]["recall"] is None


@pytest.mark.parametrize(
    "sizes", [[], [(True, 192)], [(320, 0)], [(4097, 1)], [[320, 192]], [(320,)]]
)
def test_invalid_image_sizes_rejected(sizes):
    with pytest.raises(ValueError):
        evaluate_terrain_geometry([EvaluationSample((), ())], sizes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_threshold": True},
        {"confidence_threshold": -1},
        {"iou_threshold": 0},
        {"iou_threshold": float("nan")},
    ],
)
def test_invalid_matching_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        _evaluate([], [], **kwargs)


def test_out_of_frame_boxes_and_matching_work_budget_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        _evaluate([], [_box((0, 168, 321, 172))])
    with pytest.raises(ValueError, match="500"):
        _evaluate([_box((0, 168, 320, 172))] * 501, [])


def test_benchmark_terrain_metrics_share_one_inference_per_frame_and_preserve_old_report(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=3)

    class CountingDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return PixelPerception().detect(frame)

    detector = CountingDetector()
    report = benchmark_dataset(
        dataset, tmp_path / "with.json", detector, split="val", terrain_metrics=True
    )
    assert report["schema_version"] == 3
    assert (
        detector.calls
        == report["metrics"]["images"] + report["latency_ms"]["warmup_calls_excluded"]
    )
    assert (
        report["terrain_geometry"]["platform"]["tp"]
        == report["metrics"]["per_class"]["platform"]["tp"]
    )
    assert report["terrain_geometry"]["full_width_floor"]["ground_truth"] == 3
    assert report["terrain_geometry"]["full_width_floor"]["recall"] == 1
    assert (
        report["terrain_geometry"]["full_width_floor"]["absolute_error_pixels"]["top"]["p95"]
        < 0.001
    )
    stored = json.loads((tmp_path / "with.json").read_text())
    assert stored["terrain_geometry"] == report["terrain_geometry"]
    old = benchmark_dataset(dataset, tmp_path / "without.json", split="val")
    assert old["schema_version"] == 1
    assert "terrain_geometry" not in old
    assert old["metrics"] == report["metrics"]


def test_benchmark_rejects_non_boolean_geometry_option_before_data_read(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        benchmark_dataset(tmp_path / "missing", tmp_path / "out", terrain_metrics=1)
    assert not (tmp_path / "out").exists()


@pytest.fixture
def native_detector():
    pytest.importorskip("onnxruntime")
    root = Path(__file__).parents[1] / "examples" / "lifecycle-model"
    return OnnxDetector(root / "model.onnx", root / "manifest.json")


def test_geometry_report_binds_actual_model_and_native_runtime(tmp_path, native_detector):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=1)
    result = benchmark_dataset(
        dataset, tmp_path / "report.json", native_detector, split="val", terrain_metrics=True
    )
    assert result["detector_binding_checked_before_and_after"]
    identity = result["detector"]
    assert identity["loaded_model_sha256"] == native_detector.loaded_model_sha256
    assert identity["full_model_manifest"]["model_sha256"] == native_detector.loaded_model_sha256
    assert identity["full_model_manifest"]["input_width"] == 320
    assert identity["inference_runtime"]["providers"] == ["CPUExecutionProvider"]
    assert identity["inference_runtime"]["intra_op_num_threads"] == 1
    copy = native_detector.runtime_settings
    copy["intra_op_num_threads"] = 99
    assert native_detector.runtime_settings["intra_op_num_threads"] == 1


def test_geometry_benchmark_rejects_unbound_model_before_detect(
    tmp_path, native_detector, monkeypatch
):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=1)
    native_detector.manifest = replace(native_detector.manifest, model_sha256="0" * 64)

    def forbidden(frame):
        pytest.fail("unbound identity must be rejected before any inference")

    monkeypatch.setattr(native_detector, "detect", forbidden)
    with pytest.raises(ValueError, match="loaded model hash"):
        benchmark_dataset(dataset, tmp_path / "report.json", native_detector, terrain_metrics=True)
    assert not (tmp_path / "report.json").exists()


def test_geometry_benchmark_rejects_inference_settings_changed_during_measurement(
    tmp_path, native_detector, monkeypatch
):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=1)
    original = native_detector.detect

    def change(frame):
        result = original(frame)
        native_detector.confidence_threshold = 0.3
        return result

    monkeypatch.setattr(native_detector, "detect", change)
    with pytest.raises(ValueError, match="settings changed"):
        benchmark_dataset(dataset, tmp_path / "report.json", native_detector, terrain_metrics=True)
    assert not (tmp_path / "report.json").exists()


def test_geometry_benchmark_rejects_manifest_swapped_after_validation(tmp_path, monkeypatch):
    from maple_vision import evaluation

    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=1)
    validate = evaluation.validate_dataset

    def swap(path):
        result = validate(path)
        manifest = path / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    class ForbiddenDetector:
        def detect(self, frame):
            pytest.fail("changed manifest must be rejected before warmup")

    monkeypatch.setattr(evaluation, "validate_dataset", swap)
    with pytest.raises(ValueError, match="manifest changed after validation"):
        benchmark_dataset(
            dataset, tmp_path / "report.json", ForbiddenDetector(), terrain_metrics=True
        )
    assert not (tmp_path / "report.json").exists()


def test_geometry_benchmark_rejects_manifest_changed_during_inference(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sessions=3, frames_per_session=1)

    class ChangingDetector:
        def detect(self, frame):
            manifest = dataset / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            return ()

    with pytest.raises(ValueError, match="manifest changed during evaluation"):
        benchmark_dataset(
            dataset, tmp_path / "report.json", ChangingDetector(), terrain_metrics=True
        )
    assert not (tmp_path / "report.json").exists()
