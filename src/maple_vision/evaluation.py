"""Independent detection matching and recorded, domain-qualified benchmarks.

AP uses a 101-point interpolated precision envelope at one declared IoU. It is
not COCO mAP50-95: crowd/ignore/area buckets and multiple IoUs are not implemented.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Detection
from .dataset import validate_dataset
from .onnx_detector import OnnxDetector
from .scene import CLASS_NAMES, ImageFrame
from .vision import Detector, PixelPerception


@dataclass(frozen=True)
class EvaluationSample:
    ground_truth: tuple[Detection, ...]
    predictions: tuple[Detection, ...]

    def __post_init__(self) -> None:
        for values in (self.ground_truth, self.predictions):
            if not isinstance(values, tuple) or any(not isinstance(v, Detection) for v in values):
                raise ValueError("evaluation samples need immutable Detection tuples")
            if len(values) > 10_000:
                raise ValueError("evaluation sample exceeds 10000 objects")


def _threshold(value: float, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        or (positive and value == 0)
    ):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be finite in {interval}")
    return float(value)


def _iou(a: Detection, b: Detection) -> float:
    left = max(a.bbox[0], b.bbox[0])
    top = max(a.bbox[1], b.bbox[1])
    right = min(a.bbox[2], b.bbox[2])
    bottom = min(a.bbox[3], b.bbox[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
    area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
    return intersection / (area_a + area_b - intersection)


def _summary(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ground_truth": tp + fn,
        "predictions": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _ap101(matches: list[bool], ground_truth_count: int) -> float | None:
    if not ground_truth_count:
        return None
    if not matches:
        return 0.0
    true_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for rank, matched in enumerate(matches, 1):
        true_positives += int(matched)
        recalls.append(true_positives / ground_truth_count)
        precisions.append(true_positives / rank)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    result = 0.0
    for step in range(101):
        target = step / 100
        result += next(
            (
                precision
                for recall, precision in zip(recalls, precisions, strict=True)
                if recall >= target
            ),
            0.0,
        )
    return result / 101


def evaluate_predictions(
    samples: Sequence[EvaluationSample],
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Match each prediction to at most one same-image, same-class ground truth.

    Predictions are ranked globally per class by confidence, with deterministic
    input order for ties. Unknown prediction classes count as false positives.
    AP is calculated after the specified confidence cutoff, explicitly reported.
    """
    confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")
    iou_threshold = _threshold(iou_threshold, "iou_threshold", positive=True)
    if (
        not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("class_names must contain distinct nonempty strings")
    if any(not isinstance(sample, EvaluationSample) for sample in samples):
        raise ValueError("samples must contain EvaluationSample values")
    classes = list(class_names)
    known_classes = set(classes)
    extras: set[str] = set()
    filtered_count = 0
    for sample in samples:
        if any(target.class_id not in known_classes for target in sample.ground_truth):
            raise ValueError("ground truth contains a class outside the schema")
        extras.update(
            prediction.class_id
            for prediction in sample.predictions
            if prediction.class_id not in known_classes
            and prediction.confidence >= confidence_threshold
        )
        filtered_count += sum(p.confidence < confidence_threshold for p in sample.predictions)
    classes.extend(sorted(extras))
    per_class: dict[str, dict[str, int | float | None]] = {}
    ap_values: list[float] = []
    micro_tp = micro_fp = micro_fn = 0
    for class_id in classes:
        truth_by_image = {
            index: tuple(target for target in sample.ground_truth if target.class_id == class_id)
            for index, sample in enumerate(samples)
        }
        ground_truth_count = sum(len(targets) for targets in truth_by_image.values())
        predictions = [
            (prediction.confidence, image_index, prediction_index, prediction)
            for image_index, sample in enumerate(samples)
            for prediction_index, prediction in enumerate(sample.predictions)
            if prediction.class_id == class_id and prediction.confidence >= confidence_threshold
        ]
        predictions.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
        used: set[tuple[int, int]] = set()
        matches: list[bool] = []
        for _, image_index, _, prediction in predictions:
            candidates = [
                (_iou(prediction, target), index)
                for index, target in enumerate(truth_by_image[image_index])
                if (image_index, index) not in used
            ]
            # Equal-IoU ties choose the earlier GT annotation deterministically.
            best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
            matched = best is not None and best[0] >= iou_threshold
            if matched and best is not None:
                used.add((image_index, best[1]))
            matches.append(matched)
        tp = sum(matches)
        fp = len(matches) - tp
        fn = ground_truth_count - tp
        ap = _ap101(matches, ground_truth_count)
        if ap is not None:
            ap_values.append(ap)
        per_class[class_id] = {
            **_summary(tp, fp, fn),
            "ap_at_iou": ap,
            "ap50": ap if iou_threshold == 0.5 else None,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else None
    return {
        "images": len(samples),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "ap_definition": "101-point interpolated precision envelope after confidence filtering",
        "map_at_iou": mean_ap,
        "map50": mean_ap if iou_threshold == 0.5 else None,
        "map50_95": None,
        "ap_classes_with_ground_truth": len(ap_values),
        "class_count": len(classes),
        "classes": classes,
        "unrecognized_prediction_classes": sorted(extras),
        "predictions_below_confidence_threshold": filtered_count,
        "micro": _summary(micro_tp, micro_fp, micro_fn),
        "per_class": per_class,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _environment() -> dict[str, object]:
    dependencies: dict[str, str] = {}
    for package in ("maple-vision-platform", "numpy", "Pillow", "onnxruntime"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpus": os.cpu_count(),
        "dependencies": dependencies,
    }


def _detector_metadata(detector: Detector) -> dict[str, object]:
    metadata: dict[str, object] = {
        "implementation": f"{type(detector).__module__}.{type(detector).__qualname__}",
        "scope": "unspecified_custom_detector",
        "model_sha256": None,
    }
    if isinstance(detector, PixelPerception):
        metadata.update(
            {
                "scope": "synthetic_fixture",
                "method": "fixed_palette_connected_components",
                "source_id": detector.source_id,
            }
        )
    elif isinstance(detector, OnnxDetector):
        metadata.update(
            {
                "scope": detector.manifest.scope,
                "method": "onnx_cpu",
                "model_sha256": detector.manifest.model_sha256,
                "class_names": list(detector.manifest.class_names),
                "decoder_confidence_threshold": detector.confidence_threshold,
                "decoder_nms_iou_threshold": detector.iou_threshold,
                "output_format": detector.manifest.output_format,
            }
        )
    return metadata


def _ground_truth(path: Path, width: int, height: int) -> tuple[Detection, ...]:
    objects: list[Detection] = []
    # The dataset validator has already checked class IDs, finite normalized
    # values and geometry; decode separately from the renderer that made labels.
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        class_id = int(fields[0])
        cx, cy, w, h = (float(value) for value in fields[1:])
        x1 = max(0.0, (cx - w / 2) * width)
        y1 = max(0.0, (cy - h / 2) * height)
        x2 = min(float(width), (cx + w / 2) * width)
        y2 = min(float(height), (cy + h / 2) * height)
        objects.append(Detection(CLASS_NAMES[class_id], (x1, y1, x2, y2), 1.0))
    return tuple(objects)


def benchmark_dataset(
    dataset: Path,
    output: Path,
    detector: Detector | None = None,
    split: str = "test",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Validate data, benchmark one detector, and exclusively save a JSON report."""
    confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")
    iou_threshold = _threshold(iou_threshold, "iou_threshold", positive=True)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    dataset, output = Path(dataset).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"benchmark report already exists: {output}")
    if any(
        output.resolve().is_relative_to(dataset / directory) for directory in ("images", "labels")
    ):
        raise ValueError("benchmark output cannot modify dataset images or labels")
    dataset_summary = validate_dataset(dataset)
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest: Any = json.loads(manifest_bytes)
    records = [sample for sample in manifest["samples"] if sample["split"] == split]
    if not records:
        raise ValueError("selected dataset split is empty")
    active_detector = detector if detector is not None else PixelPerception()
    warmup_count = min(3, len(records))
    for record in records[:warmup_count]:
        active_detector.detect(ImageFrame.from_file(dataset / record["image"]))
    samples: list[EvaluationSample] = []
    latency_ms: list[float] = []
    for record in records:
        frame = ImageFrame.from_file(
            dataset / record["image"],
            sequence=record["sequence"],
            timestamp_ms=record["timestamp_ms"],
        )
        started = time.perf_counter_ns()
        predictions = active_detector.detect(frame)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        latency_ms.append(elapsed_ms)
        targets = _ground_truth(dataset / record["label"], frame.width, frame.height)
        samples.append(EvaluationSample(targets, predictions))
    metrics = evaluate_predictions(
        samples, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_origin": manifest["origin"],
        "split": split,
        "dataset_split_role": "held_out_test" if split == "test" else split,
        "model_training_overlap_verified": False,
        "dataset": dataset_summary,
        "detector": _detector_metadata(active_detector),
        "metrics": metrics,
        "latency_ms": {
            "samples": len(latency_ms),
            "warmup_calls_excluded": warmup_count,
            "p50": _percentile(latency_ms, 0.5),
            "p95": _percentile(latency_ms, 0.95),
            "mean": sum(latency_ms) / len(latency_ms),
            "min": min(latency_ms),
            "max": max(latency_ms),
            "timing_scope": (
                "detect(frame) only; includes detector preprocessing/decoding, excludes PNG I/O"
            ),
            "clock": "perf_counter_ns",
        },
        "environment": _environment(),
        "limitations": [
            "Synthetic fixture results do not establish accuracy on real recordings.",
            "A split named test does not prove the model never saw those images elsewhere.",
            "AP uses a declared confidence cutoff and one IoU; this is not COCO mAP50-95.",
            "Small image counts yield uncertain accuracy and latency estimates.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {**report, "output": str(output)}
