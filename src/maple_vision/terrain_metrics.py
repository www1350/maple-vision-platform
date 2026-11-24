"""Visible terrain geometry measured after ordinary one-to-one detection matching.

This module never loads a simulator or reconstructs hidden terrain. A full-width
group is a stratum of labelled targets, not a new detector class. Its missing
targets remain false negatives even when the matched-box geometry is accurate.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .contracts import Detection

if TYPE_CHECKING:
    from .evaluation import EvaluationSample

_TERRAIN = ("platform", "ladder")
_ERRORS = ("left", "top", "right", "bottom", "width", "height")


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    from .evaluation import _percentile

    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.5) if values else None,
        "p95": _percentile(values, 0.95) if values else None,
        "max": max(values) if values else None,
    }


def _errors(target: Detection, prediction: Detection) -> dict[str, float]:
    t, p = target.bbox, prediction.bbox
    return dict(
        zip(
            _ERRORS,
            (
                abs(t[0] - p[0]),
                abs(t[1] - p[1]),
                abs(t[2] - p[2]),
                abs(t[3] - p[3]),
                abs((t[2] - t[0]) - (p[2] - p[0])),
                abs((t[3] - t[1]) - (p[3] - p[1])),
            ),
            strict=True,
        )
    )


def evaluate_terrain_geometry(
    samples: Sequence["EvaluationSample"],
    image_sizes: Sequence[tuple[int, int]],
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Report terrain recall and matched edge errors in original image pixels.

    Matching is confidence-ranked within each image/class, including all platform
    targets before stratifying full-width floors. Equal IoU chooses the first GT,
    matching the general evaluator. No split-specific selection or threshold fit
    occurs here. Limits protect this additional, potentially quadratic matching.
    """
    from .evaluation import EvaluationSample, _iou, _threshold

    confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")
    iou_threshold = _threshold(iou_threshold, "iou_threshold", positive=True)
    if len(samples) != len(image_sizes) or len(samples) > 100_000:
        raise ValueError("terrain samples need one image size each and at most 100000 images")
    for sample, size in zip(samples, image_sizes, strict=True):
        if not isinstance(sample, EvaluationSample):
            raise ValueError("terrain samples must be EvaluationSample values")
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or any(type(v) is not int or not 1 <= v <= 4096 for v in size)
        ):
            raise ValueError("image sizes must be (width, height) integer tuples in [1, 4096]")
        for objects in (sample.ground_truth, sample.predictions):
            terrain = [d for d in objects if d.class_id in _TERRAIN]
            if len(terrain) > 500:
                raise ValueError("terrain matching exceeds 500 objects per image")
            if any(d.bbox[2] > size[0] or d.bbox[3] > size[1] for d in terrain):
                raise ValueError("terrain boxes must fit declared original image dimensions")
    if sum(len(s.ground_truth) + len(s.predictions) for s in samples) > 1_000_000:
        raise ValueError("terrain evaluation exceeds one million input objects")

    def new_group() -> dict[str, Any]:
        return {"targets": 0, "predictions": 0, "matched": [], "missing": []}

    groups = {name: new_group() for name in (*_TERRAIN, "full_width_floor")}
    for image_index, (sample, (width, _height)) in enumerate(
        zip(samples, image_sizes, strict=True)
    ):
        for class_id in _TERRAIN:
            targets = [(i, d) for i, d in enumerate(sample.ground_truth) if d.class_id == class_id]
            predictions = sorted(
                (
                    (i, d)
                    for i, d in enumerate(sample.predictions)
                    if d.class_id == class_id and d.confidence >= confidence_threshold
                ),
                key=lambda item: (-item[1].confidence, item[0]),
            )
            matched: dict[int, tuple[int, Detection, float]] = {}
            for prediction_index, prediction in predictions:
                candidates = [
                    (_iou(prediction, target), index)
                    for index, target in targets
                    if index not in matched
                ]
                best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
                if best is not None and best[0] >= iou_threshold:
                    matched[best[1]] = prediction_index, prediction, best[0]
            group = groups[class_id]
            group["predictions"] += len(predictions)
            for target_index, target in targets:
                destinations = [group]
                if (
                    class_id == "platform"
                    and abs(target.bbox[0]) <= 1e-6
                    and abs(target.bbox[2] - width) <= 1e-6
                ):
                    destinations.append(groups["full_width_floor"])
                entry: dict[str, Any] = {
                    "image_index": image_index,
                    "target_index": target_index,
                    "target_bbox": list(target.bbox),
                }
                for destination in destinations:
                    destination["targets"] += 1
                if target_index not in matched:
                    for destination in destinations:
                        destination["missing"].append(entry)
                    continue
                prediction_index, prediction, overlap = matched[target_index]
                entry = {
                    **entry,
                    "prediction_index": prediction_index,
                    "prediction_bbox": list(prediction.bbox),
                    "confidence": prediction.confidence,
                    "iou": overlap,
                    "absolute_error_pixels": _errors(target, prediction),
                }
                for destination in destinations:
                    destination["matched"].append(entry)

    def summary(name: str) -> dict[str, Any]:
        group = groups[name]
        records = group["matched"]
        tp, count = len(records), group["targets"]
        ranked = sorted(
            records,
            key=lambda record: (
                -max(record["absolute_error_pixels"].values()),
                record["image_index"],
                record["target_index"],
            ),
        )
        return {
            "ground_truth": count,
            "tp": tp,
            "fn": count - tp,
            "recall": tp / count if count else None,
            "fp": group["predictions"] - tp if name != "full_width_floor" else None,
            "absolute_error_pixels": {
                key: _statistics([r["absolute_error_pixels"][key] for r in records])
                for key in _ERRORS
            },
            "matched_iou": _statistics([r["iou"] for r in records]),
            "worst_matched_examples": ranked[:20],
            "missing_examples": group["missing"][:20],
            "examples_are_capped_at": 20,
        }

    return {
        "schema_version": 1,
        "images": len(samples),
        "source": "visible_labels_and_model_predictions_only",
        "coordinate_space": "original_image_pixels",
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "matching": (
            "confidence-ranked one-to-one within image and class; "
            "all platforms matched before grouping"
        ),
        "full_width_floor_definition": (
            "platform left=0 and right=image_width, absolute tolerance 1e-6"
        ),
        "quantile": "linear interpolation at (n-1)*q",
        "error_population": "matched ground-truth targets only; inspect recall separately",
        "floor_false_positives": (
            "not defined for a ground-truth stratum; inspect overall platform false positives"
        ),
        "platform": summary("platform"),
        "ladder": summary("ladder"),
        "full_width_floor": summary("full_width_floor"),
        "limitations": [
            "Full width is an image-coordinate property, not proof of walkable or hidden geometry.",
            "Matched-box errors exclude misses; missing targets remain false negatives.",
            "Correlated frames and repeated maps do not establish real or unseen-map accuracy.",
        ],
    }
