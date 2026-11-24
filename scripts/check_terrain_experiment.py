"""Check supplied v0.3 benchmark numbers and identities using only the standard library.

This does not infer, rescore predictions, authenticate a preregistration, or promote
a model. Actual scenario, re-inference and soak gates are separate. Exit status:
0 = supplied quality gates pass, 1 = valid measurements fail, 2 = invalid input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 16 * 1024 * 1024
CLASSES = ("player", "monster", "loot", "portal", "npc", "platform", "ladder")
DYNAMIC_CLASSES = CLASSES[:5]
AP_DEFINITION = "101-point interpolated precision envelope after confidence filtering"
TIMING_SCOPE = "detect(frame) only; includes detector preprocessing/decoding, excludes PNG I/O"
TERRAIN_MATCHING = (
    "confidence-ranked one-to-one within image and class; all platforms matched before grouping"
)
FLOOR_DEFINITION = "platform left=0 and right=image_width, absolute tolerance 1e-6"


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _field(value: dict[str, Any], key: str, name: str) -> Any:
    if key not in value:
        raise ValueError(f"missing {name}.{key}")
    return value[key]


def _number(value: Any, name: str, *, low: float = 0, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < low or (high is not None and result > high):
        raise ValueError(f"{name} is outside its finite measurement range")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _equal(actual: Any, expected: Any, name: str) -> None:
    if type(expected) is bool:
        same = actual is expected
    else:
        same = actual == expected
    if not same:
        raise ValueError(f"{name} does not match the required report identity")


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    # Check nested keys too: Python dicts can contain values JSON files cannot.
    pending = [value]
    visited: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list, tuple)):
            if id(item) in visited:
                continue
            visited.add(id(item))
        if isinstance(item, dict):
            _object(item, "input")
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("inputs must contain only JSON values")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("input contains non-finite or invalid JSON values") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("input exceeds the 16 MiB JSON limit")
    return encoded


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _classes(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(CLASSES)
        or set(value) != set(CLASSES)
    ):
        raise ValueError(f"{name} must contain each of the seven object classes once")
    return tuple(value)


def _recall(row: dict[str, Any], name: str) -> tuple[int, int, float]:
    count = _integer(row.get("ground_truth"), f"{name}.ground_truth", minimum=1)
    tp = _integer(row.get("tp"), f"{name}.tp")
    fn = _integer(row.get("fn"), f"{name}.fn")
    if tp + fn != count:
        raise ValueError(f"{name} target counts disagree")
    recall = _number(row.get("recall"), f"{name}.recall", high=1)
    if not math.isclose(recall, tp / count, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name}.recall disagrees with target counts")
    return count, tp, recall


def _report_identity(
    report: dict[str, Any], registration: dict[str, Any], split: str, *, candidate: bool
) -> dict[str, Any]:
    name = "candidate" if candidate else "baseline"
    version = _integer(report.get("schema_version"), f"{name}.schema_version")
    if version not in ((3,) if candidate else (2, 3)):
        raise ValueError(f"{name} has an unsupported benchmark schema")
    expected_data = _object(registration.get("dataset"), "preregistration.dataset")
    evaluation = _object(registration.get("evaluation"), "preregistration.evaluation")
    count = _integer(expected_data["split_frames"].get(split), "expected split frames", minimum=1)
    _equal(report.get("split"), split, f"{name}.split")
    _equal(
        report.get("dataset_split_role"),
        "held_out_test" if split == "test" else split,
        f"{name}.dataset_split_role",
    )
    _equal(report.get("scope"), "synthetic_fixture", f"{name}.scope")
    _equal(report.get("dataset_origin"), "synthetic-fixture-v1", f"{name}.dataset_origin")
    data_hash = _sha(report.get("dataset_manifest_sha256"), f"{name}.dataset_manifest_sha256")
    _equal(data_hash, expected_data.get("manifest_sha256"), f"{name}.dataset_manifest_sha256")
    data = _object(report.get("dataset"), f"{name}.dataset")
    # Historical schema2 baseline summaries did not record this nested hash.
    # A supplied nested identity must never contradict the checked outer one;
    # schema3 explicitly requires both bindings rather than silently filling it.
    if version == 3 or "manifest_sha256" in data:
        _equal(
            _sha(data.get("manifest_sha256"), f"{name}.dataset.manifest_sha256"),
            data_hash,
            f"{name}.dataset.manifest_sha256",
        )
    _equal(data.get("validated"), True, f"{name}.dataset.validated")
    _equal(data.get("generator_version"), expected_data.get("generator"), f"{name}.generator")
    split_frames = _object(data.get("split_frames"), f"{name}.dataset.split_frames")
    _equal(_integer(split_frames.get(split), f"{name}.split_frames"), count, f"{name}.split_frames")
    metrics = _object(report.get("metrics"), f"{name}.metrics")
    _equal(_integer(metrics.get("images"), f"{name}.images"), count, f"{name}.images")
    _classes(metrics.get("classes"), f"{name}.metrics.classes")
    _equal(_integer(metrics.get("class_count"), f"{name}.class_count"), 7, f"{name}.class_count")
    _equal(metrics.get("unrecognized_prediction_classes"), [], f"{name}.unknown_classes")
    _equal(metrics.get("ap_definition"), AP_DEFINITION, f"{name}.ap_definition")
    _equal(metrics.get("map50_95"), None, f"{name}.map50_95")
    for field, expected in (
        ("confidence_threshold", evaluation.get("confidence_threshold")),
        ("iou_threshold", evaluation.get("matching_iou_threshold")),
    ):
        _equal(_number(metrics.get(field), f"{name}.{field}", high=1), expected, f"{name}.{field}")
    ap = _number(metrics.get("map50"), f"{name}.map50", high=1)
    _equal(
        _number(metrics.get("map_at_iou"), f"{name}.map_at_iou", high=1), ap, f"{name}.map_at_iou"
    )
    per_class = _object(metrics.get("per_class"), f"{name}.per_class")
    recalls: dict[str, float] = {}
    ground_truth: dict[str, int] = {}
    counts = {"tp": 0, "fp": 0, "fn": 0}
    class_ap: list[float] = []
    for category in CLASSES:
        row = _object(per_class.get(category), f"{name}.per_class.{category}")
        ground_truth[category], matched, recalls[category] = _recall(row, f"{name}.{category}")
        false_positives = _integer(row.get("fp"), f"{name}.{category}.fp")
        _equal(
            _integer(row.get("predictions"), f"{name}.{category}.predictions"),
            matched + false_positives,
            f"{name}.{category}.predictions",
        )
        counts["tp"] += matched
        counts["fp"] += false_positives
        counts["fn"] += ground_truth[category] - matched
        class_ap.append(_number(row.get("ap50"), f"{name}.{category}.ap50", high=1))
    if not math.isclose(ap, sum(class_ap) / len(class_ap), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name}.map50 disagrees with per-class AP values")
    micro = _object(metrics.get("micro"), f"{name}.micro")
    for key, expected in {
        **counts,
        "predictions": counts["tp"] + counts["fp"],
        "ground_truth": counts["tp"] + counts["fn"],
    }.items():
        _equal(_integer(micro.get(key), f"{name}.micro.{key}"), expected, f"{name}.micro.{key}")
    detector = _object(report.get("detector"), f"{name}.detector")
    _equal(
        detector.get("implementation"),
        "maple_vision.onnx_detector.OnnxDetector",
        f"{name}.detector.implementation",
    )
    _equal(detector.get("method"), "onnx_cpu", f"{name}.detector.method")
    _equal(detector.get("scope"), "synthetic_fixture", f"{name}.detector.scope")
    classes = _classes(detector.get("class_names"), f"{name}.detector.class_names")
    _equal(detector.get("output_format"), "raw_xywh", f"{name}.detector.output_format")
    model_hash = _sha(detector.get("model_sha256"), f"{name}.model_sha256")
    if not candidate:
        _equal(model_hash, registration.get("baseline_model_sha256"), "baseline.model_sha256")
    for field, expected in (
        ("decoder_confidence_threshold", evaluation.get("confidence_threshold")),
        ("decoder_nms_iou_threshold", evaluation.get("detector_nms_iou")),
    ):
        _equal(_number(detector.get(field), f"{name}.{field}", high=1), expected, f"{name}.{field}")
    if version == 3:
        _equal(
            report.get("detector_binding_checked_before_and_after"),
            True,
            f"{name}.detector_binding_checked_before_and_after",
        )
        manifest = _object(detector.get("full_model_manifest"), "candidate.full_model_manifest")
        _equal(
            _integer(manifest.get("schema_version"), "model manifest schema"),
            1,
            "model manifest schema",
        )
        _equal(
            _sha(detector.get("loaded_model_sha256"), "loaded_model_sha256"),
            model_hash,
            "loaded_model_sha256",
        )
        _equal(
            _sha(manifest.get("model_sha256"), "manifest.model_sha256"),
            model_hash,
            "manifest.model_sha256",
        )
        _equal(
            _classes(manifest.get("class_names"), "manifest.class_names"),
            classes,
            "manifest.class_names order",
        )
        _equal(manifest.get("scope"), "synthetic_fixture", "manifest.scope")
        _equal(manifest.get("output_format"), detector["output_format"], "manifest.output_format")
        size = _integer(
            registration["training"].get("image_size"), "training.image_size", minimum=1
        )
        for dimension in ("input_width", "input_height"):
            _equal(_integer(manifest.get(dimension), dimension, minimum=1), size, dimension)
        runtime = _object(detector.get("inference_runtime"), "candidate.inference_runtime")
        _equal(runtime.get("providers"), ["CPUExecutionProvider"], "inference providers")
        _equal(
            _integer(runtime.get("intra_op_num_threads"), "intra_op_num_threads", minimum=1),
            evaluation.get("onnx_intra_op_threads"),
            "intra_op_num_threads",
        )
    latency = _object(report.get("latency_ms"), f"{name}.latency_ms")
    _equal(
        _integer(latency.get("samples"), f"{name}.latency.samples", minimum=1),
        count,
        f"{name}.latency.samples",
    )
    _equal(
        _integer(latency.get("warmup_calls_excluded"), f"{name}.warmup"),
        evaluation.get("warmup_calls_excluded"),
        f"{name}.warmup",
    )
    _equal(latency.get("timing_scope"), TIMING_SCOPE, f"{name}.timing_scope")
    _equal(latency.get("clock"), "perf_counter_ns", f"{name}.latency.clock")
    p95 = _number(latency.get("p95"), f"{name}.latency.p95")
    return {
        "schema_version": version,
        "dataset_manifest_sha256": data_hash,
        "model_sha256": model_hash,
        "split": split,
        "images": count,
        "latency_samples": count,
        "warmup_calls_excluded": latency["warmup_calls_excluded"],
        "recalls": recalls,
        "ground_truth": ground_truth,
        "macro_ap50": ap,
        "detect_latency_p95_ms": p95,
    }


def _terrain(
    report: dict[str, Any], identity: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Any]:
    terrain = _object(report.get("terrain_geometry"), "candidate.terrain_geometry")
    _equal(
        _integer(terrain.get("schema_version"), "terrain.schema_version"),
        1,
        "terrain.schema_version",
    )
    for key, expected in (
        ("source", "visible_labels_and_model_predictions_only"),
        ("coordinate_space", "original_image_pixels"),
        ("matching", TERRAIN_MATCHING),
        ("full_width_floor_definition", FLOOR_DEFINITION),
        ("quantile", "linear interpolation at (n-1)*q"),
        ("error_population", "matched ground-truth targets only; inspect recall separately"),
    ):
        _equal(terrain.get(key), expected, f"terrain.{key}")
    _equal(_integer(terrain.get("images"), "terrain.images"), identity["images"], "terrain.images")
    for key, expected in (
        ("confidence_threshold", evaluation["confidence_threshold"]),
        ("iou_threshold", evaluation["matching_iou_threshold"]),
    ):
        _equal(_number(terrain.get(key), f"terrain.{key}", high=1), expected, f"terrain.{key}")
    for category in ("platform", "ladder"):
        group = _object(terrain.get(category), f"terrain.{category}")
        count, _, recall = _recall(group, f"terrain.{category}")
        _equal(count, identity["ground_truth"][category], f"terrain.{category}.ground_truth")
        _equal(recall, identity["recalls"][category], f"terrain.{category}.recall")
    floor = _object(terrain.get("full_width_floor"), "terrain.full_width_floor")
    count, matched, recall = _recall(floor, "terrain.full_width_floor")
    if count > identity["ground_truth"]["platform"]:
        raise ValueError("full-width floor population exceeds all platforms")
    platform = terrain["platform"]
    if matched > platform["tp"] or count - matched > platform["fn"]:
        raise ValueError("full-width floor stratum counts exceed all platforms")
    _equal(floor.get("fp"), None, "full-width floor FP stratum semantics")
    errors = _object(floor.get("absolute_error_pixels"), "floor.absolute_error_pixels")
    edge_values: dict[str, float | None] = {}
    for edge in ("top", "left", "right"):
        statistics = _object(errors.get(edge), f"floor.{edge}")
        samples = _integer(statistics.get("samples"), f"floor.{edge}.samples")
        _equal(samples, matched, f"floor.{edge}.samples")
        measured = _field(statistics, "p95", f"floor.{edge}")
        if matched:
            edge_values[edge] = _number(measured, f"floor.{edge}.p95")
        else:
            _equal(measured, None, f"unmatched floor.{edge}.p95")
            edge_values[edge] = None
    return {"ground_truth": count, "matched": matched, "recall": recall, "edges": edge_values}


def _gate(
    actual: float | None, required: float, operator: str, *, missing: str = ""
) -> dict[str, Any]:
    passed = actual is not None and (actual >= required if operator == ">=" else actual <= required)
    result: dict[str, Any] = {
        "actual": actual,
        "required": {"operator": operator, "value": required},
        "passed": passed,
    }
    if actual is None:
        result["reason"] = missing
    return result


def check_experiment(
    preregistration: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check report identity and frozen numeric quality gates; never rescore or promote.

    A nonempty floor population with zero matches is a valid quality failure;
    its missing edge measurements fail explicitly. Inconsistent, missing or
    non-finite required evidence raises ValueError. The supplied registration
    is trusted input, not a signature or proof of preregistration timing.
    """
    preregistration = _object(preregistration, "preregistration")
    candidate = _object(candidate, "candidate")
    registration_hash, candidate_hash = _hash(preregistration), _hash(candidate)
    _equal(
        _integer(preregistration.get("schema_version"), "preregistration.schema_version"),
        1,
        "preregistration.schema_version",
    )
    split = candidate.get("split")
    if split not in ("val", "test"):
        raise ValueError("candidate.split must be val or test")
    if split == "val" and baseline is None:
        raise ValueError("validation requires the preregistered model on the same validation split")
    if split == "test" and baseline is not None:
        raise ValueError("test gates do not accept a baseline report or evaluate the old model")
    dataset = _object(preregistration.get("dataset"), "preregistration.dataset")
    _sha(dataset.get("manifest_sha256"), "preregistration.dataset.manifest_sha256")
    _object(dataset.get("split_frames"), "preregistration.dataset.split_frames")
    _sha(preregistration.get("baseline_model_sha256"), "preregistration.baseline_model_sha256")
    _object(preregistration.get("training"), "preregistration.training")
    evaluation = _object(preregistration.get("evaluation"), "preregistration.evaluation")
    _equal(evaluation.get("split"), "val", "preregistered evaluation split")
    for key in ("confidence_threshold", "matching_iou_threshold", "detector_nms_iou"):
        _number(evaluation.get(key), f"evaluation.{key}", high=1)
    _equal(evaluation["matching_iou_threshold"], 0.5, "AP50 matching threshold")
    _integer(evaluation.get("warmup_calls_excluded"), "expected warmup")
    _integer(evaluation.get("onnx_intra_op_threads"), "expected ONNX thread count", minimum=1)
    _equal(
        _integer(evaluation.get("measurements"), "expected measurements", minimum=1),
        dataset["split_frames"].get("val"),
        "expected validation measurements",
    )
    quality = _object(preregistration.get("validation_gates"), "validation_gates")
    _equal(quality.get("dynamic_classes"), list(DYNAMIC_CLASSES), "preregistered dynamic classes")
    _equal(
        quality.get("macro_ap50_min"), "same validation baseline macro AP50", "validation AP gate"
    )
    limits = {}
    for key in (
        "platform_recall_min",
        "ladder_recall_min",
        "full_width_floor_recall_min",
        "each_dynamic_recall_min_baseline_minus",
        "micro_precision_min",
    ):
        limits[key] = _number(quality.get(key), f"validation_gates.{key}", high=1)
    for key in (
        "full_width_floor_top_edge_abs_error_p95_max_pixels",
        "full_width_floor_each_endpoint_abs_error_p95_max_pixels",
        "detect_latency_p95_max_ms",
    ):
        limits[key] = _number(quality.get(key), f"validation_gates.{key}")
    holdout = _object(preregistration.get("holdout_policy"), "holdout_policy")
    test_dynamic = _number(
        holdout.get("test_dynamic_recall_min"), "test dynamic recall gate", high=1
    )
    test_ap = _number(holdout.get("test_macro_ap50_min"), "test AP gate", high=1)
    measured = _report_identity(candidate, preregistration, split, candidate=True)
    reference = None
    baseline_hash = None
    if baseline is not None:
        baseline = _object(baseline, "baseline")
        baseline_hash = _hash(baseline)
        reference = _report_identity(baseline, preregistration, "val", candidate=False)
        _equal(
            reference["ground_truth"],
            measured["ground_truth"],
            "validation ground-truth population",
        )
    floor = _terrain(candidate, measured, evaluation)
    micro = _object(candidate["metrics"].get("micro"), "candidate.metrics.micro")
    tp = _integer(micro.get("tp"), "micro.tp")
    fp = _integer(micro.get("fp"), "micro.fp")
    _equal(
        _integer(micro.get("predictions"), "micro.predictions"), tp + fp, "micro prediction count"
    )
    precision = micro.get("precision")
    if tp + fp:
        precision = _number(precision, "micro.precision", high=1)
        if not math.isclose(precision, tp / (tp + fp), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("micro precision disagrees with prediction counts")
    else:
        _equal(precision, None, "precision without predictions")
    gates: dict[str, Any] = {}
    for category in ("platform", "ladder"):
        gates[f"{category}_recall"] = _gate(
            measured["recalls"][category], limits[f"{category}_recall_min"], ">="
        )
    gates["full_width_floor_recall"] = _gate(
        floor["recall"], limits["full_width_floor_recall_min"], ">="
    )
    for edge in ("top", "left", "right"):
        threshold = (
            limits["full_width_floor_top_edge_abs_error_p95_max_pixels"]
            if edge == "top"
            else limits["full_width_floor_each_endpoint_abs_error_p95_max_pixels"]
        )
        gates[f"full_width_floor_{edge}_error_p95_pixels"] = _gate(
            floor["edges"][edge], threshold, "<=", missing="no_matched_floor_geometry"
        )
    gates["micro_precision"] = _gate(
        precision, limits["micro_precision_min"], ">=", missing="no_predictions_for_precision"
    )
    gates["detect_latency_p95_ms"] = _gate(
        measured["detect_latency_p95_ms"], limits["detect_latency_p95_max_ms"], "<="
    )
    for category in DYNAMIC_CLASSES:
        threshold = (
            test_dynamic
            if reference is None
            else reference["recalls"][category] - limits["each_dynamic_recall_min_baseline_minus"]
        )
        gates[f"dynamic_recall_{category}"] = _gate(measured["recalls"][category], threshold, ">=")
    gates["macro_ap50"] = _gate(
        measured["macro_ap50"], test_ap if reference is None else reference["macro_ap50"], ">="
    )
    return {
        "schema_version": 1,
        "scope": "supplied_report_quality_and_identity_check",
        "split": split,
        "input_identity": {
            "hash_kind": "SHA-256 of canonical JSON, not a signature or model-file verification",
            "preregistration_sha256": registration_hash,
            "candidate_report_sha256": candidate_hash,
            "baseline_report_sha256": baseline_hash,
            "dataset_manifest_sha256": measured["dataset_manifest_sha256"],
            "candidate_model_sha256": measured["model_sha256"],
            "baseline_model_sha256": None if reference is None else reference["model_sha256"],
            "images": measured["images"],
            "latency_samples": measured["latency_samples"],
            "warmup_calls_excluded": measured["warmup_calls_excluded"],
            "confidence_threshold": evaluation["confidence_threshold"],
            "matching_iou_threshold": evaluation["matching_iou_threshold"],
            "detector_nms_iou": evaluation["detector_nms_iou"],
            "full_width_floor_ground_truth": floor["ground_truth"],
            "full_width_floor_matches": floor["matched"],
        },
        "gates": gates,
        "quality_gates_passed": all(gate["passed"] for gate in gates.values()),
        "predictions_rescored": False,
        "model_promotion_granted": False,
        "separate_gates": ["scenario_suite", "actual_reinference", "resource_soak"],
        "limitations": [
            "Only supplied report values and identity claims are checked; "
            "predictions are not rescored.",
            "Hashes identify supplied content, not authenticity, "
            "preregistration timing or test isolation.",
            "Absence of concurrent training and hardware load requires separate process evidence.",
            "Quality pass alone does not grant model promotion, real-domain or 24-hour acceptance.",
        ],
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{path.name} exceeds the 16 MiB JSON limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant
    )
    result = _object(value, path.name)
    _canonical(result)
    return result, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        registration, registration_hash = _read_json(args.preregistration)
        candidate, candidate_hash = _read_json(args.candidate)
        baseline = None
        baseline_hash = None
        if args.baseline is not None:
            if candidate.get("split") == "test":
                raise ValueError("test gates do not accept or read a baseline report")
            baseline, baseline_hash = _read_json(args.baseline)
        result = check_experiment(registration, candidate, baseline)
        result["input_file_sha256"] = {
            "preregistration": registration_hash,
            "candidate": candidate_hash,
            "baseline": baseline_hash,
        }
        encoded = (
            json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(encoded, end="")
        return 0 if result["quality_gates_passed"] else 1
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(json.dumps({"schema_version": 1, "status": "invalid_input", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
