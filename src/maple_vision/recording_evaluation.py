"""Score frozen, human-reviewed frame labels without exposing them to inference."""

import hashlib
import io
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import canonical_json, strict_json
from .contracts import Detection
from .dataset import _contained, validate_dataset
from .evaluation import (
    EvaluationSample,
    _bounded_manifest_bytes,
    _detector_metadata,
    _environment,
    _iou,
    evaluate_predictions,
)
from .local_files import publish_new, read_regular
from .recordings import _load_frame, _validated_manifest
from .scene import CLASS_NAMES
from .vision import Detector, PixelPerception

MAX_FRAMES = 5000
MAX_OBJECTS = 50_000
MAX_OBJECTS_PER_FRAME = 128
MAX_LABEL_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Case:
    recording: Path
    manifest: dict[str, Any]
    label_sha256: str
    reviewer: str
    domain: str
    annotations: tuple[tuple[str, tuple[Detection, ...], tuple[bool, ...]], ...]


def _write(path: Path, value: object) -> None:
    # Reuse atomic create-only publication. Never overwrite reviewed labels.
    publish_new(
        path, (canonical_json(value) + "\n").encode(), 64 * 1024 * 1024, "recording evidence"
    )


def create_label_template(recording: Path, output: Path) -> dict[str, object]:
    if output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write the label template outside the captured recording")
    manifest = _validated_manifest(recording)
    if manifest["frame_count"] > MAX_FRAMES:
        raise ValueError("recording exceeds the scored frame budget")
    value = {
        "schema_version": 1,
        "kind": "recording_object_labels",
        "session_id": manifest["session_id"],
        "domain": None,
        "reviewer": None,
        "frames": [
            {
                "sequence": frame["sequence"],
                "rgb_sha256": frame["rgb_sha256"],
                "map_id": None,
                "objects": None,
            }
            for frame in manifest["frames"]
        ],
    }
    _write(output, value)
    return {
        "session_id": manifest["session_id"],
        "frames": len(value["frames"]),
        "output": str(output),
    }


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"invalid annotation {field}")
    return value


def _case(recording: Path, labels: Path, available_frames: int) -> Case:
    manifest = _validated_manifest(recording)
    if manifest["frame_count"] > available_frames:
        raise ValueError("total recording frame budget exceeded")
    raw = read_regular(labels, MAX_LABEL_BYTES, "frame labels")
    document = strict_json(raw.decode())
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "kind", "session_id", "domain", "reviewer", "frames"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != "recording_object_labels"
        or document["session_id"] != manifest["session_id"]
    ):
        raise ValueError("label format or recording session mismatch")
    domain = document["domain"]
    if domain not in ("authored_fixture", "game_recording"):
        raise ValueError("declare the recording domain")
    reviewer = _text(document["reviewer"], "reviewer")
    frames = document["frames"]
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("labels must cover every imported frame")
    result = []
    for entry, frame in zip(frames, manifest["frames"], strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sequence", "rgb_sha256", "map_id", "objects"}
            or type(entry["sequence"]) is not int
            or entry["sequence"] != frame["sequence"]
            or entry["rgb_sha256"] != frame["rgb_sha256"]
        ):
            raise ValueError("label frame identity differs from captured pixels")
        map_id = _text(entry["map_id"], "map ID")
        objects = entry["objects"]
        if not isinstance(objects, list) or len(objects) > MAX_OBJECTS_PER_FRAME:
            raise ValueError("each frame needs reviewed objects, with [] for reviewed empty frames")
        boxes = []
        occlusions = []
        for item in objects:
            if (
                not isinstance(item, dict)
                or set(item) != {"class_id", "bbox", "occluded"}
                or item["class_id"] not in CLASS_NAMES
                or not isinstance(item["bbox"], list)
                or type(item["occluded"]) is not bool
            ):
                raise ValueError("invalid object label")
            box = Detection(item["class_id"], tuple(item["bbox"]), 1.0)
            if box.bbox[2] > manifest["width"] or box.bbox[3] > manifest["height"]:
                raise ValueError("label exceeds recording bounds")
            boxes.append(box)
            occlusions.append(item["occluded"])
        result.append((map_id, tuple(boxes), tuple(occlusions)))
    return Case(
        recording, manifest, hashlib.sha256(raw).hexdigest(), reviewer, domain, tuple(result)
    )


def _recall_groups(
    sample: EvaluationSample,
    flags: tuple[bool, ...],
    threshold: float,
    totals: Counter[str],
    matched: Counter[str],
) -> None:
    # Match over all instances first, so visible objects cannot be counted as hits
    # on an occluded neighbor. Ties follow the input order of checked annotations.
    hits: set[int] = set()
    for pred in sorted(sample.predictions, key=lambda d: -d.confidence):
        if pred.confidence < threshold:
            continue
        candidates = [
            (_iou(pred, obj), i)
            for i, obj in enumerate(sample.ground_truth)
            if i not in hits and obj.class_id == pred.class_id
        ]
        if candidates:
            score, i = max(candidates, key=lambda x: (x[0], -x[1]))
            if score >= 0.5:
                hits.add(i)
    for i, obj in enumerate(sample.ground_truth):
        x1, y1, x2, y2 = obj.bbox
        for group in (
            "occluded" if flags[i] else "unoccluded",
            "small" if (x2 - x1) * (y2 - y1) < 32**2 else "other_size",
        ):
            totals[group] += 1
            matched[group] += int(i in hits)


def _development_split(
    dataset: Path, recording_pixels: set[tuple[int, int, str]]
) -> dict[str, Any]:
    root = dataset.resolve()
    manifest_path = root / "manifest.json"
    original = _bounded_manifest_bytes(manifest_path)
    validate_dataset(root)
    doc = strict_json(original.decode())
    count = 0
    for row in doc["samples"]:
        if row["split"] not in ("train", "val"):
            continue
        path = _contained(root, row["image"])
        image_bytes = read_regular(path, 64 * 1024 * 1024, "development image")
        if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError("development image differs from the validated manifest")
        with Image.open(io.BytesIO(image_bytes)) as image:
            key = (
                image.width,
                image.height,
                hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
            )
        if key in recording_pixels:
            raise ValueError("recording pixels overlap the declared train/val data")
        count += 1
    if original != _bounded_manifest_bytes(manifest_path):
        raise ValueError("development manifest changed during evaluation preflight")
    return {"manifest_sha256": hashlib.sha256(original).hexdigest(), "train_val_frames": count}


def benchmark_recordings(
    pairs: Sequence[tuple[Path, Path]],
    output: Path,
    detector: Detector | None = None,
    *,
    confidence: float = 0.25,
    development_datasets: Sequence[Path] = (),
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"report already exists: {output}")
    if not 1 <= len(pairs) <= 32:
        raise ValueError("a benchmark needs 1..32 labeled recording sessions")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be in [0,1]")
    cases: list[Case] = []
    sources: set[str] = set()
    frame_count = truth_count = 0
    for recording, labels in pairs:
        case = _case(recording, labels, MAX_FRAMES - frame_count)
        if case.manifest["source_sha256"] in sources:
            raise ValueError("duplicate source recording")
        if cases and case.domain != cases[0].domain:
            raise ValueError("score fixtures and game recordings in separate reports")
        sources.add(case.manifest["source_sha256"])
        cases.append(case)
        frame_count += len(case.annotations)
        truth_count += sum(len(row[1]) for row in case.annotations)
        if truth_count > MAX_OBJECTS:
            raise ValueError("ground-truth object budget exceeded")

    if len(development_datasets) > 16:
        raise ValueError("up to 16 development datasets can be checked")
    roots = [case.recording for case in cases] + list(development_datasets)
    if any(output.resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError("write benchmark results outside input bundles")
    recording_pixels = {
        (case.manifest["width"], case.manifest["height"], frame["rgb_sha256"])
        for case in cases
        for frame in case.manifest["frames"]
    }
    development = [_development_split(path, recording_pixels) for path in development_datasets]
    engine: Detector = PixelPerception() if detector is None else detector
    metadata = _detector_metadata(engine, include_runtime=True)
    samples = []
    by_map: dict[str, list[EvaluationSample]] = {}
    sessions = []
    times = []
    seen_rgb: set[str] = set()
    cross_session_duplicates: set[str] = set()
    predictions = 0
    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for case in cases:
        session_samples = []
        rgb = {frame["rgb_sha256"] for frame in case.manifest["frames"]}
        cross_session_duplicates.update(seen_rgb & rgb)
        seen_rgb.update(rgb)
        for item, (map_id, truth, flags) in zip(
            case.manifest["frames"], case.annotations, strict=True
        ):
            frame = _load_frame(case.recording, item)
            started = time.perf_counter()
            result = engine.detect(frame)
            times.append((time.perf_counter() - started) * 1000)
            if not isinstance(result, tuple) or len(result) > MAX_OBJECTS_PER_FRAME:
                raise ValueError("prediction count exceeds the per-frame budget")
            sample = EvaluationSample(truth, result)
            if any(d.bbox[2] > frame.width or d.bbox[3] > frame.height for d in result):
                raise ValueError("prediction exceeds recording bounds")
            predictions += len(result)
            if predictions > MAX_OBJECTS:
                raise ValueError("prediction budget exceeded")
            _recall_groups(sample, flags, confidence, totals, matched)
            samples.append(sample)
            session_samples.append(sample)
            by_map.setdefault(map_id, []).append(sample)
        sessions.append(
            {
                "session_id": case.manifest["session_id"],
                "source_sha256": case.manifest["source_sha256"],
                "labels_sha256": case.label_sha256,
                "reviewer": case.reviewer,
                "frames": len(session_samples),
                "truncated": case.manifest["truncated"],
                "source_uri": case.manifest["source_uri"],
                "permission": case.manifest["permission"],
                "metrics": evaluate_predictions(session_samples, confidence_threshold=confidence),
            }
        )
    if _detector_metadata(engine, include_runtime=True) != metadata:
        raise ValueError("detector identity changed during the benchmark")
    levels = [
        evaluate_predictions(samples, confidence_threshold=confidence, iou_threshold=i / 100)
        for i in range(50, 96, 5)
    ]
    measured = [float(v) for level in levels if isinstance(v := level["map_at_iou"], (float, int))]
    report = {
        "schema_version": 1,
        "kind": "recording_detection_benchmark",
        "declared_domain": cases[0].domain,
        "frames": frame_count,
        "session_count": len(cases),
        "ground_truth_objects": truth_count,
        "predicted_objects": predictions,
        "detector": metadata,
        "environment": _environment(),
        "sessions": sessions,
        "metrics_at_50": levels[0],
        "mean_ap_50_95": statistics.mean(measured) if measured else None,
        "ap_by_iou": [
            {"iou": level["iou_threshold"], "mean_ap": level["map_at_iou"]} for level in levels
        ],
        "ap_definition": "AP101 after confidence filtering, macro over labeled classes",
        "by_map": {
            k: evaluate_predictions(v, confidence_threshold=confidence)
            for k, v in sorted(by_map.items())
        },
        "recall_at_50": {
            group: {
                "matched": matched[group],
                "total": totals[group],
                "recall": matched[group] / totals[group] if totals[group] else None,
            }
            for group in ("small", "other_size", "occluded", "unoccluded")
        },
        "exact_rgb_hashes_shared_across_sessions": len(cross_session_duplicates),
        "training_overlap_verified": False,
        "declared_development_exact_rgb_checks": development,
        "action_outcomes_evaluated": False,
        "latency_ms": {
            "scope": "detector.detect including first call",
            "median": statistics.median(times),
            "p95": sorted(times)[(len(times) * 95 + 99) // 100 - 1],
        },
    }
    _write(output, report)
    return report
