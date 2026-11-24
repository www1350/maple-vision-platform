"""Bounded, local structural audits of the supported axis-aligned YOLO DFL head.

This measures candidate geometry and regression-target support, not learned
accuracy or the final task-aligned positive assignment. Heavy imports happen
only under the same offline runtime used by training.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import platform
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .catalog import canonical_json, strict_json
from .dataset import _contained, validate_dataset
from .scene import CLASS_NAMES
from .training import _image_size, _integer, _local_file, _offline_runtime, _validate_model_config

TERRAIN_CLASSES = ("platform", "ladder")
MAX_AUDIT_BOXES = 50_000
MAX_UNIQUE_BOXES = 4096
MAX_ANCHORS = 100_000
TARGET_MARGIN = 0.01


def _preflight_model_config(path: Path, image_size: int) -> dict[str, Any]:
    """Bound construction/forward shapes before importing native model libraries.

    Deliberately support only the serial topology used by the two local fixtures.
    The training wrapper has a broader allowlist; that does not make every one of
    its configurations suitable for a small architecture audit.
    """
    _validate_model_config(path)
    config: Any = importlib.import_module("yaml").safe_load(path.read_text("utf-8"))
    if (
        image_size > 640
        or config["nc"] != len(CLASS_NAMES)
        or config.get("width_multiple", 1.0) > 1
        or config.get("depth_multiple", 1.0) > 1
        or config.get("end2end", False)
        or config.get("reg_max", 16) < 2
        or len(config["head"]) != 1
        or len(config["backbone"]) > 15
    ):
        raise ValueError("audit preflight rejects class, size or plain Detect DFL topology budget")
    # DetectionModel also probes256px during construction for smaller audits.
    allocation_size = max(256, image_size)
    cumulative_stride = 1
    grids: list[int] = []
    channels: list[int] = []
    feature_elements = 0
    for reference, repeats, module, arguments in config["backbone"]:
        if reference != -1 or module not in {"Conv", "C2f", "SPPF"}:
            raise ValueError("audit preflight supports only serial Conv/C2f/SPPF layers")
        if (
            not arguments
            or type(arguments[0]) is not int
            or not 8 <= arguments[0] <= 128
            or arguments[0] % 8
        ):
            raise ValueError("audit preflight channel budget is 8..128 in multiples of eight")
        if module == "Conv":
            if (
                repeats != 1
                or len(arguments) != 3
                or type(arguments[1]) is not int
                or arguments[1] not in (1, 3, 5)
                or type(arguments[2]) is not int
                or arguments[2] not in (1, 2)
            ):
                raise ValueError("audit preflight supports bounded Conv kernels and strides")
            cumulative_stride *= arguments[2]
        elif module == "C2f":
            if repeats > 2 or len(arguments) != 2 or type(arguments[1]) is not bool:
                raise ValueError("audit preflight C2f repetition/arguments exceed the contract")
        elif (
            repeats != 1
            or len(arguments) != 2
            or type(arguments[1]) is not int
            or arguments[1] not in (3, 5)
        ):
            raise ValueError("audit preflight SPPF arguments exceed the contract")
        if cumulative_stride > 64 or image_size % cumulative_stride:
            raise ValueError("audit preflight requires integral feature grids with stride <=64")
        grid = image_size // cumulative_stride
        elements = arguments[0] * (allocation_size // cumulative_stride) ** 2
        feature_elements += elements
        if elements > 4 * 1024 * 1024 or feature_elements > 8 * 1024 * 1024:
            raise ValueError("audit preflight feature allocation budget exceeded")
        grids.append(grid)
        channels.append(arguments[0])
    references, repeats, module, arguments = config["head"][0]
    if (
        module != "Detect"
        or repeats != 1
        or arguments != ["nc"]
        or not isinstance(references, list)
        or not 1 <= len(references) <= 4
        or len(set(references)) != len(references)
        or any(index < 0 or index >= len(grids) for index in references)
    ):
        raise ValueError(
            "audit preflight requires one bounded Detect with explicit layer references"
        )
    strides = [image_size // grids[index] for index in references]
    if strides != sorted(set(strides)) or strides[0] < 2:
        raise ValueError("audit preflight needs distinct ascending detection strides >=2")
    anchor_count = sum(grids[index] ** 2 for index in references)
    allocation_anchors = sum((allocation_size // stride) ** 2 for stride in strides)
    regression_width = max(16, channels[references[0]] // 4, 4 * config.get("reg_max", 16))
    classification_width = max(channels[references[0]], config["nc"])
    allocation_bound = (
        feature_elements + 4 * (regression_width + classification_width) * allocation_anchors
    )
    if anchor_count > MAX_ANCHORS or allocation_bound > 32 * 1024 * 1024:
        raise ValueError("audit preflight head/anchor allocation budget exceeded")
    return {
        "topology": "serial Conv/C2f/SPPF plus one ordinary Detect",
        "layer_count": len(config["backbone"]) + 1,
        "maximum_declared_channels": max(channels),
        "maximum_input_size": 640,
        "allocation_probe_size": allocation_size,
        "conservative_feature_elements": feature_elements,
        "conservative_feature_and_head_elements": allocation_bound,
        "predicted_strides": strides,
        "predicted_anchor_count": anchor_count,
        "scope": "allocation shape guard; not a process memory or wall-clock guarantee",
    }


def assess_box_capacity(
    box: Sequence[float],
    anchor_points: ArrayLike,
    anchor_strides: ArrayLike,
    candidate_mask: ArrayLike,
    reg_max: int,
) -> dict[str, Any]:
    """Check every L/T/R/B distance for independently supplied candidate anchors.

    Coordinates are input pixels; distances are divided by each anchor's stride.
    Closed DFL support [0, R-1] includes limiting values of finite-logit softmax.
    The supported training target clamp is [0, R-1-0.01]. A point outside the
    original box has a negative target even if tiny-box expansion made it eligible.
    This pure numerical helper does not generate or infer the candidate mask.
    """
    reg_max = _integer(reg_max, "reg_max", 2, 32)
    arrays = [np.asarray(value) for value in (box, anchor_points, anchor_strides)]
    if any(value.dtype.kind not in "iuf" for value in arrays):
        raise ValueError("box, anchors and strides must contain real numbers")
    bounds, points, strides = [value.astype(np.float64) for value in arrays]
    candidates = np.asarray(candidate_mask)
    if (
        bounds.shape != (4,)
        or points.ndim != 2
        or points.shape[1] != 2
        or strides.shape != (len(points),)
        or candidates.shape != (len(points),)
        or candidates.dtype != np.bool_
        or len(points) > MAX_ANCHORS
        or any(not np.isfinite(value).all() for value in (bounds, points, strides))
        or np.any(strides <= 0)
        or bounds[2] <= bounds[0]
        or bounds[3] <= bounds[1]
    ):
        raise ValueError("invalid box or bounded anchor geometry")
    distances = np.concatenate((points - bounds[:2], bounds[2:] - points), axis=1)
    distances /= strides[:, None]
    support = candidates & ((distances >= 0) & (distances <= reg_max - 1)).all(axis=1)
    unclipped = candidates & ((distances >= 0) & (distances <= reg_max - 1 - TARGET_MARGIN)).all(
        axis=1
    )
    interior = candidates & ((distances > 0) & (distances < reg_max - 1)).all(axis=1)
    negative = candidates & (distances < 0).any(axis=1)
    overflow = candidates & (distances > reg_max - 1).any(axis=1)

    def counts(mask: Any) -> dict[str, Any]:
        valid_indices = np.flatnonzero(mask & unclipped)
        witness = None
        if len(valid_indices):
            index = int(valid_indices[0])
            witness = {
                "anchor_xy": points[index].tolist(),
                "stride": float(strides[index]),
                "ltrb_stride_units": distances[index].tolist(),
            }
        return {
            "eligible_anchors": int(np.count_nonzero(mask & candidates)),
            "support_closure_anchors": int(np.count_nonzero(mask & support)),
            "training_unclipped_anchors": int(np.count_nonzero(mask & unclipped)),
            "strict_support_interior_anchors": int(np.count_nonzero(mask & interior)),
            "negative_side_anchors": int(np.count_nonzero(mask & negative)),
            "overrange_side_anchors": int(np.count_nonzero(mask & overflow)),
            "witness": witness,
        }

    result = counts(np.ones(len(points), dtype=bool))
    result["by_stride"] = {
        str(float(stride)): {
            **counts(strides == stride),
            "per_side_support_max_pixels": float((reg_max - 1) * stride),
            "per_side_training_max_pixels": float((reg_max - 1 - TARGET_MARGIN) * stride),
            "total_width_support_max_pixels": float(2 * (reg_max - 1) * stride),
        }
        for stride in np.unique(strides)
    }
    return result


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_dataset(dataset: Path, split: str) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset / "manifest.json"
    if manifest_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("dataset manifest exceeds audit size limit")
    manifest_bytes = manifest_path.read_bytes()
    validate_dataset(dataset)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("dataset manifest changed during audit validation")
    manifest: Any = strict_json(manifest_bytes.decode("utf-8"))
    sessions = {record["id"]: record for record in manifest["sessions"]}
    classes = manifest["classes"]
    if not set(TERRAIN_CLASSES) <= set(classes):
        raise ValueError("audit requires the platform and ladder class schema")
    records: list[dict[str, Any]] = []
    selected_images = 0
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        selected_images += 1
        label_path = _contained(dataset, sample["label"])
        raw = label_path.read_bytes()
        if _sha(raw) != sample["label_sha256"]:
            raise ValueError("selected labels changed after dataset validation")
        for row_index, row in enumerate(raw.decode("utf-8").splitlines()):
            parts = row.split()
            name = classes[int(parts[0])]
            if name not in TERRAIN_CLASSES:
                continue
            records.append(
                {
                    "image": sample["image"],
                    "session_id": sample["session_id"],
                    "row": row_index,
                    "class_name": name,
                    "map_id": sample.get("map_id", sessions[sample["session_id"]]["map_id"]),
                    "phase": sample.get("phase", "legacy_unlabeled"),
                    "width": sample["width"],
                    "height": sample["height"],
                    "normalized_xywh": [float(value) for value in parts[1:]],
                }
            )
            if len(records) > MAX_AUDIT_BOXES:
                raise ValueError("terrain audit exceeds bounded ground-truth count")
    if not selected_images or not records:
        raise ValueError("selected split has no terrain ground truth")
    manifest["audit_selected_images"] = selected_images
    return manifest_bytes, manifest, records


def _inspect_model(
    torch: Any, ultralytics: Any, config: Path, image_size: int, warmup: int, iterations: int
) -> tuple[Any, Any, Any, dict[str, Any]]:
    head_module = importlib.import_module("ultralytics.nn.modules.head")
    block_module = importlib.import_module("ultralytics.nn.modules.block")
    tal = importlib.import_module("ultralytics.utils.tal")
    loss = importlib.import_module("ultralytics.utils.loss")
    model = ultralytics.YOLO(str(config), task="detect", verbose=False).model.cpu().eval()
    head = model.model[-1]
    if (
        type(head) is not head_module.Detect
        or head.end2end
        or head.xyxy
        or type(head.dfl) is not block_module.DFL
        or type(head.reg_max) is not int
        or not 2 <= head.reg_max <= 32
        or sum(parameter.numel() for parameter in model.parameters()) > 5_000_000
    ):
        raise ValueError("unsupported model: audit needs a bounded plain Detect DFL head")
    expected_bins = torch.arange(head.reg_max, dtype=torch.float32)
    bins = head.dfl.conv.weight.detach().cpu().flatten()
    if bins.shape != expected_bins.shape or not torch.equal(bins, expected_bins):
        raise ValueError("unsupported DFL projection weights")
    criterion = model.init_criterion()
    if (
        type(criterion) is not loss.v8DetectionLoss
        or type(criterion.assigner) is not tal.TaskAlignedAssigner
        or type(criterion.bbox_loss.dfl_loss) is not loss.DFLoss
        or criterion.reg_max != head.reg_max
        or not torch.equal(criterion.proj.cpu(), expected_bins)
    ):
        raise ValueError("unsupported training loss or assigner")
    # Execute both real target-clamping paths instead of assuming their epsilon.
    target = torch.tensor([[-1.0, 0.5, float(head.reg_max - 1), float(head.reg_max + 1)]])
    expected_target = target.clamp(0, head.reg_max - 1 - TARGET_MARGIN)
    criterion.bbox_loss.dfl_loss(torch.zeros((4, head.reg_max)), target)
    actual_target = tal.bbox2dist(
        torch.zeros((1, 2)),
        torch.tensor([[1.0, -0.5, float(head.reg_max - 1), float(head.reg_max + 1)]]),
        head.reg_max - 1,
    )
    if not torch.equal(target, expected_target) or not torch.equal(actual_target, expected_target):
        raise ValueError("unsupported DFL target clamp behavior")
    distances = torch.tensor([[[1.0], [2.0], [5.0], [6.0]]])
    decoded = head.decode_bboxes(distances, torch.tensor([[[3.0], [4.0]]]))
    if not torch.equal(decoded, torch.tensor([[[5.0], [6.0], [6.0], [8.0]]])):
        raise ValueError("unsupported bounding-box decode behavior")
    inputs = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    timings: list[float] = []
    with torch.inference_mode():
        output = model(inputs)
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], dict):
            raise ValueError("unsupported raw detection output")
        predictions, raw = output
        if set(raw) != {"boxes", "scores", "feats"}:
            raise ValueError("unsupported raw detection fields")
        features = raw["feats"]
        strides = [float(value) for value in head.stride.tolist()]
        if (
            not 1 <= len(strides) <= 4
            or len(features) != len(strides)
            or strides != sorted(set(strides))
            or any(value not in (2, 4, 8, 16, 32, 64) for value in strides)
            or any(
                list(feature.shape[2:]) != [image_size / stride, image_size / stride]
                for feature, stride in zip(features, strides, strict=True)
            )
        ):
            raise ValueError("unsupported feature grid or actual head strides")
        anchors, anchor_strides = tal.make_anchors(features, head.stride, 0.5)
        if len(anchors) > MAX_ANCHORS or list(predictions.shape) != [1, 4 + head.nc, len(anchors)]:
            raise ValueError("unsupported or excessive model output shape")
        if not torch.equal(head.anchors.T, anchors) or not torch.equal(
            head.strides.T, anchor_strides
        ):
            raise ValueError("inference anchors disagree with the audited training grid")
        if iterations:
            for _ in range(warmup):
                model(inputs)
            for _ in range(iterations):
                started = time.perf_counter_ns()
                model(inputs)
                timings.append((time.perf_counter_ns() - started) / 1_000_000)
    source_modules = (head_module, block_module, tal, loss)
    report = {
        "head_type": f"{type(head).__module__}.{type(head).__name__}",
        "reg_max": head.reg_max,
        "dfl_bins": bins.tolist(),
        "strides": strides,
        "feature_shapes": [list(feature.shape) for feature in features],
        "output_shape": list(predictions.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "assigner": {
            "type": type(criterion.assigner).__name__,
            "topk": criterion.assigner.topk,
            "stride_val": float(criterion.assigner.stride_val),
            "candidate_stage_only": True,
        },
        "target_interval_stride_units": [0, head.reg_max - 1 - TARGET_MARGIN],
        "support_closure_stride_units": [0, head.reg_max - 1],
        "cpu_forward": {
            "performed": bool(iterations),
            "warmup_iterations": warmup if iterations else 0,
            "structural_probe_forwards_excluded": 1,
            "timed_iterations": iterations,
            "batch": 1,
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "includes": "eager model backbone, head and DFL decode; no image I/O or NMS",
            "latency_ms": (
                {
                    "p50": float(np.percentile(timings, 50)),
                    "p95": float(np.percentile(timings, 95)),
                    "min": min(timings),
                    "max": max(timings),
                    "samples": timings,
                }
                if timings
                else None
            ),
        },
        "primary_local_sources": [
            {
                "module": module.__name__,
                "sha256": _sha(Path(inspect.getfile(module)).read_bytes()),
            }
            for module in source_modules
        ],
    }
    return anchors * anchor_strides, anchor_strides[:, 0], criterion.assigner, report


def _transform_records(records: list[dict[str, Any]], image_size: int) -> list[dict[str, Any]]:
    augment = importlib.import_module("ultralytics.data.augment")
    instance = importlib.import_module("ultralytics.utils.instance")
    transform = augment.LetterBox(
        new_shape=(image_size, image_size), auto=False, scale_fill=False, scaleup=True, center=True
    )
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["width"], record["height"])].append(record)
    if len(groups) > 32:
        raise ValueError("audit exceeds source-shape budget")
    transforms = []
    for (width, height), group in groups.items():
        if width * height > 4 * 1024 * 1024:
            raise ValueError("audit source image shape exceeds budget")
        parameters = transform.get_params({"img": np.zeros((height, width, 3), dtype=np.uint8)})
        instances = instance.Instances(
            bboxes=np.asarray([r["normalized_xywh"] for r in group], dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        )
        instances.convert_bbox("xyxy")
        instances.denormalize(width, height)
        source_boxes = instances.bboxes.copy()
        instances.scale(*parameters["ratio"])
        instances.add_padding(parameters["left"], parameters["top"])
        for record, source, box in zip(group, source_boxes, instances.bboxes, strict=True):
            record["source_box"] = source.tolist()
            record["input_box"] = box.tolist()
        transforms.append(
            {
                "source_size": [width, height],
                "input_size": [image_size, image_size],
                "ratio": list(parameters["ratio"]),
                "padding_left_top": [parameters["left"], parameters["top"]],
            }
        )
    return transforms


def _coverage(
    records: list[dict[str, Any]], capacities: dict[tuple[float, ...], Any]
) -> dict[str, Any]:
    metrics = {
        "ground_truth_boxes": len(records),
        "images": len({record["image"] for record in records}),
        "with_candidate_anchor": 0,
        "with_support_closure_anchor": 0,
        "with_training_unclipped_anchor": 0,
        "with_strict_support_interior_anchor": 0,
    }
    fields = {
        "with_candidate_anchor": "eligible_anchors",
        "with_support_closure_anchor": "support_closure_anchors",
        "with_training_unclipped_anchor": "training_unclipped_anchors",
        "with_strict_support_interior_anchor": "strict_support_interior_anchors",
    }
    for record in records:
        capacity = capacities[tuple(record["input_box"])]
        for key, field in fields.items():
            metrics[key] += int(capacity[field] > 0)
    return {
        **metrics,
        "unclipped_target_coverage": (
            metrics["with_training_unclipped_anchor"] / len(records) if records else None
        ),
    }


def _audit_geometry(
    torch: Any,
    records: list[dict[str, Any]],
    points: Any,
    strides: Any,
    assigner: Any,
    reg_max: int,
) -> dict[str, Any]:
    counts = Counter(tuple(record["input_box"]) for record in records)
    boxes = list(counts)
    if len(boxes) > MAX_UNIQUE_BOXES:
        raise ValueError("audit exceeds unique geometry budget")
    points_np, strides_np = points.numpy(), strides.numpy()
    capacities: dict[tuple[float, ...], Any] = {}
    chunk_size = max(1, min(32, 1_000_000 // len(points)))
    calls = 0
    for offset in range(0, len(boxes), chunk_size):
        subset = boxes[offset : offset + chunk_size]
        targets = torch.tensor([subset], dtype=torch.float32)
        eligible = assigner.select_candidates_in_gts(
            points, targets, torch.ones((1, len(subset), 1), dtype=torch.bool)
        )
        if eligible.dtype != torch.bool or list(eligible.shape) != [1, len(subset), len(points)]:
            raise ValueError("unsupported assigner candidate mask")
        calls += 1
        for box, mask in zip(subset, eligible[0].numpy(), strict=True):
            capacities[box] = assess_box_capacity(box, points_np, strides_np, mask, reg_max)
    full_width = [
        record
        for record in records
        if record["class_name"] == "platform"
        and abs(record["source_box"][0]) < 0.001
        and abs(record["source_box"][2] - record["width"]) < 0.001
        and record["source_box"][3] - record["source_box"][1] <= 8.001
    ]
    groups: dict[str, Any] = {}
    for field in ("class_name", "map_id", "phase"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record[field]].append(record)
        groups[field] = {
            name: _coverage(group, capacities) for name, group in sorted(grouped.items())
        }
    return {
        "overall": _coverage(records, capacities),
        "groups": groups,
        "full_width_thin_platforms": _coverage(full_width, capacities),
        "full_width_thin_definition": "source x1=0, x2=image width within .001px, height<=8.001px",
        "unique_geometry_count": len(boxes),
        "actual_assigner_calls": calls,
        "maximum_candidate_pairs_per_call": chunk_size * len(points),
        "unique_geometries": [
            {"input_box": list(box), "occurrences": counts[box], "capacity": capacities[box]}
            for box in boxes
        ],
        "uncovered_examples": [
            {key: record[key] for key in ("image", "class_name", "row", "map_id", "input_box")}
            for record in records
            if not capacities[tuple(record["input_box"])]["training_unclipped_anchors"]
        ][:20],
        "missing_terrain_classes": sorted(set(TERRAIN_CLASSES) - set(groups["class_name"])),
    }


def _publish_report(output: Path, report: dict[str, Any]) -> None:
    """Link a completed local file exclusively; never replace an existing report."""
    encoded = (canonical_json(report) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".terrain-audit-", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(encoded)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit_terrain_architecture(
    dataset: Path,
    model_config: Path,
    output: Path,
    *,
    split: str = "val",
    image_size: int = 320,
    warmup: int = 3,
    iterations: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit local random weights and train/val labels; never inspect test targets.

    Set iterations=0 to omit latency measurement while another job uses the CPU.
    Dataset integrity validation still checks the complete dataset, including its
    held-out files; only the requested train/val labels enter geometry analysis.
    Use a separate CLI process because the offline dependency guard is process-wide.
    """
    if split not in ("train", "val"):
        raise ValueError("terrain architecture selection permits only train or val")
    image_size = _image_size(image_size)
    if image_size > 640:
        raise ValueError("audit preflight image_size must be <=640")
    warmup = _integer(warmup, "warmup", 1, 20)
    iterations = _integer(iterations, "iterations", 0, 100)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    dataset, output = Path(dataset).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit report already exists: {output}")
    if output.resolve().is_relative_to(dataset):
        raise ValueError("audit report cannot be written inside the source dataset")
    config = _local_file(model_config, {".yaml", ".yml"}, "model_config")
    _preflight_model_config(config, image_size)
    config_bytes = config.read_bytes()
    manifest_bytes, manifest, records = _read_dataset(dataset, split)
    with tempfile.TemporaryDirectory(prefix="maple-terrain-audit-") as temporary:
        staging = Path(temporary)
        copied_config = staging / "model.yaml"
        copied_config.write_bytes(config_bytes)
        preflight = _preflight_model_config(copied_config, image_size)
        with _offline_runtime(staging) as (torch, ultralytics), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            points, strides, assigner, architecture = _inspect_model(
                torch, ultralytics, copied_config, image_size, warmup, iterations
            )
            if architecture["output_shape"][1] != len(manifest["classes"]) + 4:
                raise ValueError("model class count disagrees with dataset schema")
            transforms = _transform_records(records, image_size)
            geometry = _audit_geometry(
                torch, records, points, strides, assigner, architecture["reg_max"]
            )
            runtime = {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            }
        if copied_config.read_bytes() != config_bytes:
            raise ValueError("copied model configuration changed during audit")
    if (dataset / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("dataset manifest changed during audit")
    report = {
        "schema_version": 1,
        "scope": "synthetic_fixture_structural_terrain_audit_not_accuracy",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_manifest_sha256": _sha(manifest_bytes),
        "dataset_schema_version": manifest["schema_version"],
        "dataset_origin": manifest["origin"],
        "split": split,
        "selected_images": manifest["audit_selected_images"],
        "terrain_classes": list(TERRAIN_CLASSES),
        "test_targets_analyzed": False,
        "complete_dataset_integrity_checked": True,
        "model_config_sha256": _sha(config_bytes),
        "model_initialization": "local_random_weights_no_pretraining",
        "seed": seed,
        "image_size": image_size,
        "preflight": preflight,
        "transforms": transforms,
        "architecture": architecture,
        "geometry": geometry,
        "environment": runtime,
        "all_terrain_gt_have_unclipped_target_anchor": (
            geometry["overall"]["unclipped_target_coverage"] == 1.0
            and not geometry["missing_terrain_classes"]
        ),
        "training_success_evaluated": False,
        "limitations": [
            "Candidate eligibility precedes quality-ranked top-k and competing-GT assignment; "
            "it does not prove any trained model selects that anchor.",
            "DFL support includes limiting endpoints; a zero edge is a training-valid target "
            "approached by finite-logit softmax, not guaranteed exact coordinates.",
            "Observed train/val geometry coverage does not cover unseen maps, assets "
            "or resolutions.",
            "Square centered letterbox with no geometric augmentation is audited; changing "
            "preprocessing or augmentation requires a new audit.",
            "Random-weight CPU forward latency excludes loading, image preprocessing and NMS; "
            "shared-host load affects measurements and does not predict training duration.",
        ],
    }
    _publish_report(output, report)
    return {**report, "output": str(output)}
