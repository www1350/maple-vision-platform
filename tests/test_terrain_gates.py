"""Small independent report counterexamples for the preregistered numeric gates."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_terrain_experiment.py"
SPEC = importlib.util.spec_from_file_location("terrain_gates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATES)
CLASSES = ("player", "monster", "loot", "portal", "npc", "platform", "ladder")


@pytest.fixture
def preregistration():
    # Numerical requirements are deliberately written here independently of the
    # checker. These dictionaries are synthetic reports, not held-out predictions.
    return {
        "schema_version": 1,
        "baseline_model_sha256": "b" * 64,
        "dataset": {
            "manifest_sha256": "a" * 64,
            "generator": "lifecycle-v1",
            "split_frames": {"val": 192, "test": 192},
        },
        "training": {"image_size": 320},
        "evaluation": {
            "split": "val",
            "confidence_threshold": 0.25,
            "matching_iou_threshold": 0.5,
            "detector_nms_iou": 0.45,
            "onnx_intra_op_threads": 1,
            "warmup_calls_excluded": 3,
            "measurements": 192,
        },
        "validation_gates": {
            "platform_recall_min": 0.98,
            "ladder_recall_min": 0.98,
            "full_width_floor_recall_min": 0.98,
            "full_width_floor_top_edge_abs_error_p95_max_pixels": 1.0,
            "full_width_floor_each_endpoint_abs_error_p95_max_pixels": 4.0,
            "dynamic_classes": list(CLASSES[:5]),
            "each_dynamic_recall_min_baseline_minus": 0.01,
            "macro_ap50_min": "same validation baseline macro AP50",
            "micro_precision_min": 0.98,
            "detect_latency_p95_max_ms": 75.0,
        },
        "holdout_policy": {"test_dynamic_recall_min": 0.95, "test_macro_ap50_min": 0.97},
    }


def update_counts(report):
    metrics = report["metrics"]
    totals = dict.fromkeys(("tp", "fp", "fn", "ground_truth", "predictions"), 0)
    for row in metrics["per_class"].values():
        row["fn"] = row["ground_truth"] - row["tp"]
        row["predictions"] = row["tp"] + row["fp"]
        row["recall"] = row["tp"] / row["ground_truth"]
        row["ap50"] = min(row["ap50"], row["recall"])
        for key in totals:
            totals[key] += row[key]
    metrics["micro"] = {
        **totals,
        "precision": totals["tp"] / totals["predictions"] if totals["predictions"] else None,
    }
    metrics["map50"] = metrics["map_at_iou"] = (
        sum(row["ap50"] for row in metrics["per_class"].values()) / 7
    )
    if "terrain_geometry" in report:
        for category in ("platform", "ladder"):
            report["terrain_geometry"][category] = deepcopy(metrics["per_class"][category])


def floor_group(count=50, matched=50):
    return {
        "ground_truth": count,
        "tp": matched,
        "fn": count - matched,
        "recall": matched / count if count else None,
        "fp": None,
        "absolute_error_pixels": {
            edge: {"samples": matched, "p95": value if matched else None}
            for edge, value in (("top", 0.5), ("left", 2.0), ("right", 3.0))
        },
    }


def report(preregistration, *, split="val", baseline=False, schema=None):
    model_hash = ("b" if baseline else "c") * 64
    version = schema if schema is not None else (2 if baseline else 3)
    value = {
        "schema_version": version,
        "scope": "synthetic_fixture",
        "dataset_manifest_sha256": preregistration["dataset"]["manifest_sha256"],
        "dataset_origin": "synthetic-fixture-v1",
        "split": split,
        "dataset_split_role": "held_out_test" if split == "test" else split,
        "dataset": {
            "validated": True,
            "generator_version": "lifecycle-v1",
            "split_frames": {"val": 192, "test": 192},
        },
        "detector": {
            "implementation": "maple_vision.onnx_detector.OnnxDetector",
            "method": "onnx_cpu",
            "scope": "synthetic_fixture",
            "model_sha256": model_hash,
            "class_names": list(CLASSES),
            "output_format": "raw_xywh",
            "decoder_confidence_threshold": 0.25,
            "decoder_nms_iou_threshold": 0.45,
        },
        "metrics": {
            "images": 192,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "classes": list(CLASSES),
            "class_count": 7,
            "unrecognized_prediction_classes": [],
            "ap_definition": "101-point interpolated precision envelope after confidence filtering",
            "map50_95": None,
            "per_class": {
                name: {
                    "ground_truth": 1000 if name == "platform" else 100,
                    "tp": 1000 if name == "platform" else 100,
                    "fp": (200 if name == "platform" else 20)
                    if baseline
                    else (20 if name == "platform" else 2),
                    "ap50": 0.9 if baseline else 0.99,
                }
                for name in CLASSES
            },
        },
        "latency_ms": {
            "samples": 192,
            "warmup_calls_excluded": 3,
            "p95": 30.0,
            "timing_scope": (
                "detect(frame) only; includes detector preprocessing/decoding, excludes PNG I/O"
            ),
            "clock": "perf_counter_ns",
        },
    }
    if version == 3:
        value["dataset"]["manifest_sha256"] = preregistration["dataset"]["manifest_sha256"]
        value["detector_binding_checked_before_and_after"] = True
        value["detector"].update(
            {
                "loaded_model_sha256": model_hash,
                "full_model_manifest": {
                    "schema_version": 1,
                    "model_sha256": model_hash,
                    "class_names": list(CLASSES),
                    "input_width": 320,
                    "input_height": 320,
                    "output_format": "raw_xywh",
                    "scope": "synthetic_fixture",
                },
                "inference_runtime": {
                    "providers": ["CPUExecutionProvider"],
                    "intra_op_num_threads": 1,
                },
            }
        )
        value["terrain_geometry"] = {
            "schema_version": 1,
            "images": 192,
            "source": "visible_labels_and_model_predictions_only",
            "coordinate_space": "original_image_pixels",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "matching": (
                "confidence-ranked one-to-one within image and class; "
                "all platforms matched before grouping"
            ),
            "full_width_floor_definition": (
                "platform left=0 and right=image_width, absolute tolerance 1e-6"
            ),
            "quantile": "linear interpolation at (n-1)*q",
            "error_population": "matched ground-truth targets only; inspect recall separately",
            "full_width_floor": floor_group(),
        }
    update_counts(value)
    return value


def test_valid_validation_uses_new_val_baseline_and_retains_all_fourteen_gates(preregistration):
    candidate, baseline = report(preregistration), report(preregistration, baseline=True)
    originals = deepcopy((preregistration, candidate, baseline))
    result = GATES.check_experiment(preregistration, candidate, baseline)
    assert len(result["gates"]) == 14
    assert result["quality_gates_passed"] and all(g["passed"] for g in result["gates"].values())
    assert result["gates"]["dynamic_recall_player"]["required"]["value"] == 0.99
    assert result["gates"]["macro_ap50"]["required"]["value"] == pytest.approx(0.9)
    assert not result["predictions_rescored"] and not result["model_promotion_granted"]
    assert result["input_identity"]["baseline_model_sha256"] == "b" * 64
    assert (preregistration, candidate, baseline) == originals


def test_platform_aggregate_cannot_hide_a_failing_floor_stratum(preregistration):
    candidate = report(preregistration)
    candidate["metrics"]["per_class"]["platform"]["tp"] = 998
    update_counts(candidate)
    candidate["terrain_geometry"]["full_width_floor"] = floor_group(50, 48)
    result = GATES.check_experiment(
        preregistration, candidate, report(preregistration, baseline=True)
    )
    assert not result["quality_gates_passed"]
    assert result["gates"]["platform_recall"]["passed"]
    assert result["gates"]["full_width_floor_recall"]["actual"] == 0.96
    assert not result["gates"]["full_width_floor_recall"]["passed"]
    assert result["gates"]["full_width_floor_top_error_p95_pixels"]["passed"]


def test_zero_matched_floors_is_valid_quality_failure_not_corrupt_input(preregistration):
    candidate = report(preregistration)
    candidate["metrics"]["per_class"]["platform"]["tp"] = 990
    update_counts(candidate)
    candidate["terrain_geometry"]["full_width_floor"] = floor_group(10, 0)
    result = GATES.check_experiment(
        preregistration, candidate, report(preregistration, baseline=True)
    )
    assert not result["quality_gates_passed"]
    assert result["gates"]["full_width_floor_recall"]["actual"] == 0
    for edge in ("top", "left", "right"):
        gate = result["gates"][f"full_width_floor_{edge}_error_p95_pixels"]
        assert gate["actual"] is None and not gate["passed"]
        assert gate["reason"] == "no_matched_floor_geometry"


def test_empty_floor_population_is_missing_required_evidence(preregistration):
    candidate = report(preregistration)
    candidate["terrain_geometry"]["full_width_floor"] = floor_group(0, 0)
    with pytest.raises(ValueError, match="ground_truth"):
        GATES.check_experiment(preregistration, candidate, report(preregistration, baseline=True))


def test_each_endpoint_is_gated_separately_and_thresholds_are_inclusive(preregistration):
    candidate = report(preregistration)
    edges = candidate["terrain_geometry"]["full_width_floor"]["absolute_error_pixels"]
    edges["top"]["p95"], edges["left"]["p95"], edges["right"]["p95"] = 1.0, 0.0, 4.01
    result = GATES.check_experiment(
        preregistration, candidate, report(preregistration, baseline=True)
    )
    assert result["gates"]["full_width_floor_top_error_p95_pixels"]["passed"]
    assert result["gates"]["full_width_floor_left_error_p95_pixels"]["passed"]
    assert not result["gates"]["full_width_floor_right_error_p95_pixels"]["passed"]
    edges["right"]["p95"] = 4.0
    candidate["latency_ms"]["p95"] = 75.0
    assert GATES.check_experiment(
        preregistration, candidate, report(preregistration, baseline=True)
    )["quality_gates_passed"]
    candidate["latency_ms"]["p95"] = 75.01
    result = GATES.check_experiment(
        preregistration, candidate, report(preregistration, baseline=True)
    )
    assert not result["gates"]["detect_latency_p95_ms"]["passed"]


@pytest.mark.parametrize(
    "field,value", [("samples", None), ("samples", 49), ("p95", None), ("p95", float("nan"))]
)
def test_existing_floor_matches_require_finite_errors_and_exact_sample_count(
    preregistration, field, value
):
    candidate = report(preregistration)
    candidate["terrain_geometry"]["full_width_floor"]["absolute_error_pixels"]["top"][field] = value
    with pytest.raises(ValueError):
        GATES.check_experiment(preregistration, candidate, report(preregistration, baseline=True))


@pytest.mark.parametrize(
    "path,value",
    [
        (("dataset_manifest_sha256",), "d" * 64),
        (("split",), "train"),
        (("schema_version",), 2),
        (("metrics", "confidence_threshold"), 0.1),
        (("terrain_geometry", "iou_threshold"), 0.4),
        (("terrain_geometry", "source"), "simulator_hidden_geometry"),
        (("detector", "method"), "fixed_palette_connected_components"),
        (("detector", "loaded_model_sha256"), "d" * 64),
        (("detector", "full_model_manifest", "input_width"), 640),
        (("detector", "inference_runtime", "providers"), ["CUDAExecutionProvider"]),
        (("detector", "inference_runtime", "intra_op_num_threads"), 2),
        (("detector_binding_checked_before_and_after",), False),
        (("latency_ms", "samples"), None),
        (("latency_ms", "samples"), 191),
        (("latency_ms", "warmup_calls_excluded"), 2),
        (("latency_ms", "p95"), None),
        (("latency_ms", "p95"), float("inf")),
    ],
)
def test_incompatible_report_identity_or_unknown_measurements_fail(preregistration, path, value):
    candidate = report(preregistration)
    parent = candidate
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        GATES.check_experiment(preregistration, candidate, report(preregistration, baseline=True))


def test_validation_cannot_omit_or_substitute_the_frozen_baseline(preregistration):
    candidate = report(preregistration)
    with pytest.raises(ValueError, match="requires"):
        GATES.check_experiment(preregistration, candidate)
    baseline = report(preregistration, baseline=True)
    baseline["detector"]["model_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="baseline.model_sha256"):
        GATES.check_experiment(preregistration, candidate, baseline)


@pytest.mark.parametrize("changed_report", ["candidate", "baseline"])
def test_nested_dataset_hash_cannot_contradict_matching_top_level_identity(
    preregistration, changed_report
):
    candidate, baseline = report(preregistration), report(preregistration, baseline=True)
    assert GATES.check_experiment(preregistration, candidate, baseline)["quality_gates_passed"]
    changed = candidate if changed_report == "candidate" else baseline
    changed["dataset"]["manifest_sha256"] = ("d" if changed_report == "candidate" else "e") * 64
    assert changed["dataset_manifest_sha256"] == preregistration["dataset"]["manifest_sha256"]
    with pytest.raises(ValueError, match=rf"{changed_report}\.dataset\.manifest_sha256"):
        GATES.check_experiment(preregistration, candidate, baseline)


@pytest.mark.parametrize("changed_report", ["candidate", "baseline"])
def test_schema3_requires_the_nested_dataset_hash_even_if_top_level_matches(
    preregistration, changed_report
):
    candidate = report(preregistration)
    baseline = report(preregistration, baseline=True, schema=3)
    changed = candidate if changed_report == "candidate" else baseline
    del changed["dataset"]["manifest_sha256"]
    with pytest.raises(ValueError, match=rf"{changed_report}\.dataset\.manifest_sha256"):
        GATES.check_experiment(preregistration, candidate, baseline)


def test_original_schema2_baseline_can_omit_nested_hash_but_accepts_only_matching_one(
    preregistration,
):
    candidate, baseline = report(preregistration), report(preregistration, baseline=True)
    assert "manifest_sha256" not in baseline["dataset"]
    assert GATES.check_experiment(preregistration, candidate, baseline)["quality_gates_passed"]
    baseline["dataset"]["manifest_sha256"] = preregistration["dataset"]["manifest_sha256"]
    assert GATES.check_experiment(preregistration, candidate, baseline)["quality_gates_passed"]


def test_val_relative_dynamic_gate_and_test_absolute_gates_are_different(preregistration):
    candidate = report(preregistration)
    candidate["metrics"]["per_class"]["player"]["tp"] = 98
    update_counts(candidate)
    val = GATES.check_experiment(preregistration, candidate, report(preregistration, baseline=True))
    assert not val["gates"]["dynamic_recall_player"]["passed"]
    candidate["split"], candidate["dataset_split_role"] = "test", "held_out_test"
    test = GATES.check_experiment(preregistration, candidate)
    assert test["quality_gates_passed"]
    assert test["gates"]["dynamic_recall_player"]["required"]["value"] == 0.95
    assert test["gates"]["macro_ap50"]["required"]["value"] == 0.97
    assert test["input_identity"]["baseline_model_sha256"] is None
    with pytest.raises(ValueError, match="do not accept"):
        GATES.check_experiment(preregistration, candidate, report(preregistration, baseline=True))
    candidate["metrics"]["per_class"]["player"]["tp"] = 94
    update_counts(candidate)
    assert not GATES.check_experiment(preregistration, candidate)["gates"]["dynamic_recall_player"][
        "passed"
    ]
    for row in candidate["metrics"]["per_class"].values():
        row["tp"], row["ap50"] = row["ground_truth"], 0.969
    update_counts(candidate)
    assert not GATES.check_experiment(preregistration, candidate)["gates"]["macro_ap50"]["passed"]


def test_in_memory_manifest_tuples_and_schema3_baseline_are_supported(preregistration):
    candidate = report(preregistration)
    baseline = report(preregistration, baseline=True, schema=3)
    first = GATES.check_experiment(preregistration, candidate, baseline)
    candidate["detector"]["full_model_manifest"]["class_names"] = CLASSES
    second = GATES.check_experiment(preregistration, candidate, baseline)
    assert first == second
    candidate["output"] = "caller-supplied-return-path.json"
    third = GATES.check_experiment(preregistration, candidate, baseline)
    assert (
        third["input_identity"]["candidate_report_sha256"]
        != first["input_identity"]["candidate_report_sha256"]
    )


def cli_files(tmp_path, preregistration, candidate=None):
    values = {
        "preregistration": preregistration,
        "candidate": candidate or report(preregistration),
        "baseline": report(preregistration, baseline=True),
    }
    args = []
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        args.extend((f"--{name}", str(path)))
    output = tmp_path / "result.json"
    return args + ["--output", str(output)], output


def invoke(args):
    # Isolated/no-site mode excludes the checkout and installed site packages:
    # the checker must work without importing maple_vision/numpy.
    return subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_standalone_cli_passes_and_exclusively_preserves_existing_output(tmp_path, preregistration):
    args, output = cli_files(tmp_path, preregistration)
    result = invoke(args)
    assert result.returncode == 0, result.stdout + result.stderr
    stored = output.read_bytes()
    body = json.loads(stored)
    assert body["quality_gates_passed"]
    assert (
        body["input_file_sha256"]["candidate"]
        == hashlib.sha256((tmp_path / "candidate.json").read_bytes()).hexdigest()
    )
    assert invoke(args).returncode == 2
    assert output.read_bytes() == stored


def test_standalone_cli_quality_failure_is_one_and_still_writes_evidence(tmp_path, preregistration):
    candidate = report(preregistration)
    candidate["metrics"]["per_class"]["platform"]["tp"] = 990
    update_counts(candidate)
    candidate["terrain_geometry"]["full_width_floor"] = floor_group(10, 0)
    args, output = cli_files(tmp_path, preregistration, candidate)
    result = invoke(args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not json.loads(output.read_text())["quality_gates_passed"]


@pytest.mark.parametrize("raw", ['{"x":{"a":1,"a":2}}', '{"x":NaN}', '{"x":1e999}'])
def test_cli_rejects_duplicate_keys_and_nonfinite_json_before_writing(
    tmp_path, preregistration, raw
):
    args, output = cli_files(tmp_path, preregistration)
    (tmp_path / "candidate.json").write_text(raw, encoding="utf-8")
    result = invoke(args)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "invalid_input"
    assert not output.exists()


def test_cli_limits_each_json_input_to_sixteen_mib(tmp_path, preregistration):
    args, output = cli_files(tmp_path, preregistration)
    (tmp_path / "candidate.json").write_bytes(b" " * (16 * 1024 * 1024 + 1))
    result = invoke(args)
    assert result.returncode == 2 and "16 MiB" in result.stdout
    assert not output.exists()


def test_test_mode_rejects_baseline_option_before_reading_that_file(tmp_path, preregistration):
    args, output = cli_files(tmp_path, preregistration, report(preregistration, split="test"))
    (tmp_path / "baseline.json").unlink()
    result = invoke(args)
    assert result.returncode == 2 and "do not accept or read a baseline" in result.stdout
    assert not output.exists()
