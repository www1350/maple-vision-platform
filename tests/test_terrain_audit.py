import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import maple_vision.terrain_audit as audit
from maple_vision.dataset import generate_dataset
from maple_vision.onnx_detector import ModelManifest, OnnxDetector
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.terrain_audit import assess_box_capacity, audit_terrain_architecture
from maple_vision.training import _validate_model_config, export_yolo, train_yolo

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "configs" / "terrain-yolo.yaml"
BASELINE = ROOT / "configs" / "tiny-yolo.yaml"
NATIVE = os.environ.get("MAPLE_TRAINING_SMOKE") == "1" and all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "ultralytics", "onnx", "onnxruntime")
)


def test_total_width_capacity_does_not_create_a_positive_anchor_row():
    # A32px grid has rows208,240,272. The thin-box expansion
    # [232,236] -> [230,238] contains none, despite width448>320.
    points = np.asarray([(16 + 32 * x, 16 + 32 * y) for y in range(10) for x in range(10)])
    eligible = (points[:, 0] > 0) & (points[:, 0] < 320)
    eligible &= (points[:, 1] > 230) & (points[:, 1] < 238)
    assert 2 * (8 - 1) * 32 == 448 > 320
    result = assess_box_capacity((0, 232, 320, 236), points, np.full(100, 32), eligible, 8)
    assert result["eligible_anchors"] == 0
    assert result["support_closure_anchors"] == 0
    assert result["training_unclipped_anchors"] == 0


def test_total_width_capacity_is_insufficient_when_one_side_is_out_of_range():
    # Width100 fits112px. From x4 the right side is96px, above56px per-side.
    # From x52 both52px and48px fit the individual bounds.
    result = assess_box_capacity((0, 0, 100, 4), [[4, 2], [52, 2]], [8, 8], [True, True], 8)
    assert result["eligible_anchors"] == 2
    assert result["overrange_side_anchors"] == 1
    assert result["training_unclipped_anchors"] == 1
    assert result["witness"] == {
        "anchor_xy": [52.0, 2.0],
        "stride": 8.0,
        "ltrb_stride_units": [6.5, 0.25, 6.0, 0.25],
    }


def test_expanded_candidate_outside_original_box_needs_a_negative_edge():
    # Expanding2px height around y11 to8px includes y8. The original
    # top target is -2px, which nonnegative DFL cannot represent.
    result = assess_box_capacity((0, 10, 100, 12), [[50, 8]], [8], [True], 16)
    assert result["eligible_anchors"] == 1
    assert result["negative_side_anchors"] == 1
    assert result["support_closure_anchors"] == 0
    assert result["training_unclipped_anchors"] == 0


def test_support_endpoint_and_training_clamp_are_distinguished():
    result = assess_box_capacity((0, 0, 8, 2), [[6.995, 1]], [1], [True], 8)
    assert result["support_closure_anchors"] == 1
    assert result["strict_support_interior_anchors"] == 1
    assert result["training_unclipped_anchors"] == 0
    assert result["by_stride"]["1.0"]["per_side_training_max_pixels"] == 6.99


def test_zero_edge_target_is_training_valid_but_not_strict_softmax_interior():
    result = assess_box_capacity((0, 0, 4, 4), [[0, 2]], [1], [True], 8)
    assert result["training_unclipped_anchors"] == 1
    assert result["support_closure_anchors"] == 1
    assert result["strict_support_interior_anchors"] == 0


def test_stride16_reg16_has_ten_unclipped_floor_anchors_reg12_has_two():
    points = [[8 + 16 * x, 232] for x in range(20)]
    candidate = assess_box_capacity((0, 232, 320, 236), points, [16] * 20, [True] * 20, 16)
    smaller = assess_box_capacity((0, 232, 320, 236), points, [16] * 20, [True] * 20, 12)
    assert candidate["eligible_anchors"] == 20
    assert candidate["training_unclipped_anchors"] == 10
    assert candidate["strict_support_interior_anchors"] == 0
    assert candidate["witness"]["anchor_xy"] == [88.0, 232.0]
    assert smaller["training_unclipped_anchors"] == 2
    assert smaller["witness"]["anchor_xy"] == [152.0, 232.0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"reg_max": 1},
        {"reg_max": True},
        {"box": (0, 0, float("nan"), 2)},
        {"box": (0, 0, 0, 2)},
        {"box": (False, False, True, True)},
        {"anchor_points": [[1, float("inf")]]},
        {"anchor_points": [1, 1]},
        {"anchor_strides": [0]},
        {"anchor_strides": [-1]},
        {"candidate_mask": [1]},
        {"candidate_mask": [True, False]},
    ],
)
def test_invalid_geometry_is_rejected(overrides):
    arguments = {
        "box": (0, 0, 2, 2),
        "anchor_points": [[1, 1]],
        "anchor_strides": [1],
        "candidate_mask": [True],
        "reg_max": 8,
    }
    with pytest.raises(ValueError):
        assess_box_capacity(**(arguments | overrides))


def test_candidate_configuration_is_packaged_identically_and_allowlisted():
    resource = ROOT / "src" / "maple_vision" / "resources" / "terrain-yolo.yaml"
    assert CANDIDATE.read_bytes() == resource.read_bytes()
    _validate_model_config(CANDIDATE)
    config = yaml.safe_load(CANDIDATE.read_text())
    assert config["reg_max"] == 16
    assert config["head"] == [[[2, 5, 6], 1, "Detect", ["nc"]]]
    assert audit._preflight_model_config(BASELINE, 320)["predicted_strides"] == [4, 8]
    assert audit._preflight_model_config(CANDIDATE, 320)["predicted_strides"] == [4, 8, 16]


@pytest.mark.parametrize(
    "arguments",
    [
        {"split": "test"},
        {"split": "unknown"},
        {"image_size": 65},
        {"image_size": 1280},
        {"warmup": 0},
        {"warmup": 21},
        {"iterations": -1},
        {"iterations": True},
        {"iterations": 101},
        {"seed": -1},
    ],
)
def test_arguments_are_rejected_before_loading_data_or_native_dependencies(tmp_path, arguments):
    output = tmp_path / "audit.json"
    with pytest.raises(ValueError):
        audit_terrain_architecture(tmp_path / "missing", CANDIDATE, output, **arguments)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "classes",
        "channels",
        "upsample",
        "concat",
        "depth",
        "width",
        "repeats",
        "stride",
        "layers",
        "end2end",
        "no_dfl",
        "construction_allocation",
    ],
)
def test_preflight_rejects_oversized_or_unknown_topology_before_native_import(
    tmp_path, monkeypatch, mutation
):
    config = yaml.safe_load(CANDIDATE.read_text())
    if mutation == "classes":
        config["nc"] = 1000
    elif mutation == "channels":
        config["backbone"][0][3][0] = 2048
    elif mutation == "upsample":
        config["backbone"].append([-1, 1, "nn.Upsample", [None, 32, "nearest"]])
    elif mutation == "concat":
        config["backbone"].append([[-1, -2], 1, "Concat", [1]])
    elif mutation in ("depth", "width"):
        config[f"{mutation}_multiple"] = 2.0
    elif mutation == "repeats":
        config["backbone"][2][1] = 3
    elif mutation == "stride":
        config["backbone"][0][3][2] = 4
    elif mutation == "layers":
        config["backbone"].extend([[-1, 1, "C2f", [64, True]]] * 9)
    elif mutation == "end2end":
        config["end2end"] = True
    elif mutation == "no_dfl":
        config["reg_max"] = 1
    else:
        # Requested64px must not hide the constructor's256px allocation probe.
        config["backbone"] = [[-1, 1, "Conv", [128, 3, 2]]] + [[-1, 1, "C2f", [128, True]]] * 14
        config["head"] = [[[14], 1, "Detect", ["nc"]]]
    path = tmp_path / "oversized.yaml"
    path.write_text(yaml.safe_dump(config))

    def unexpected(*args, **kwargs):
        pytest.fail("preflight must reject before dataset loading or native construction")

    monkeypatch.setattr(audit, "_read_dataset", unexpected)
    monkeypatch.setattr(audit, "_offline_runtime", unexpected)
    with pytest.raises(ValueError, match="audit preflight"):
        audit_terrain_architecture(
            tmp_path / "missing",
            path,
            tmp_path / "output.json",
            image_size=64 if mutation == "construction_allocation" else 320,
        )
    assert not (tmp_path / "output.json").exists()


def test_existing_output_and_dataset_destination_are_rejected(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", "utf-8")
    with pytest.raises(FileExistsError):
        audit_terrain_architecture(tmp_path / "missing", CANDIDATE, output)
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="inside the source dataset"):
        audit_terrain_architecture(
            tmp_path / "dataset", CANDIDATE, tmp_path / "dataset" / "new.json"
        )


@pytest.fixture
def small_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=6, frames_per_session=1, seed=707)
    return dataset


def test_selected_label_corruption_fails_before_native_import(small_dataset, tmp_path, monkeypatch):
    manifest = json.loads((small_dataset / "manifest.json").read_text())
    sample = next(sample for sample in manifest["samples"] if sample["split"] == "val")
    (small_dataset / sample["label"]).write_text("corrupt", "utf-8")

    def unexpected(*args, **kwargs):
        pytest.fail("native dependencies must not run before dataset validation")

    monkeypatch.setattr(audit, "_offline_runtime", unexpected)
    with pytest.raises(ValueError, match="checksum"):
        audit_terrain_architecture(small_dataset, CANDIDATE, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_manifest_change_after_validation_is_rejected_before_following_paths(
    small_dataset, tmp_path, monkeypatch
):
    original = audit.validate_dataset

    def changed(path):
        result = original(path)
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["samples"][0]["label"] = "../outside.txt"
        (path / "manifest.json").write_text(json.dumps(manifest))
        return result

    monkeypatch.setattr(audit, "validate_dataset", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        audit_terrain_architecture(small_dataset, CANDIDATE, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_report_publication_is_exclusive_and_removes_only_its_temporary_file(tmp_path):
    output = tmp_path / "result.json"
    audit._publish_report(output, {"complete": True})
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit._publish_report(output, {"replace": True})
    assert output.read_bytes() == first
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(not NATIVE, reason="opt-in native terrain model architecture audit")
@pytest.mark.parametrize("candidate", [False, True])
def test_native_actual_head_dfl_anchor_coverage_and_offline_restoration(
    small_dataset, tmp_path, candidate
):
    manifest = json.loads((small_dataset / "manifest.json").read_text())
    records = [row for row in manifest["samples"] if row["split"] == "val"]
    sessions = {row["id"]: row for row in manifest["sessions"]}
    assert len(records) == 2
    assert {sessions[row["session_id"]]["map_id"] for row in records} == {"meadow", "cavern"}
    for row in records:
        floors = []
        for line in (small_dataset / row["label"]).read_text().splitlines():
            category, cx, _cy, width, height = map(float, line.split())
            if (
                manifest["classes"][int(category)] == "platform"
                and abs(cx - width / 2) < 1e-12
                and abs(cx + width / 2 - 1) < 1e-12
                and 0 < height * row["height"] <= 8.001
            ):
                floors.append(line)
        assert len(floors) == 1, "each independently inspected val label has exactly one floor"
    image_loader = Image.open
    original_offline = os.environ.get("YOLO_OFFLINE")
    path = tmp_path / "audit.json"
    report = audit_terrain_architecture(
        small_dataset,
        CANDIDATE if candidate else BASELINE,
        path,
        image_size=320,
        warmup=1,
        iterations=2 if candidate else 0,
    )
    assert Image.open is image_loader
    assert os.environ.get("YOLO_OFFLINE") == original_offline
    assert report["architecture"]["strides"] == ([4.0, 8.0, 16.0] if candidate else [4.0, 8.0])
    assert report["architecture"]["dfl_bins"] == list(range(16 if candidate else 8))
    timing = report["architecture"]["cpu_forward"]
    if candidate:
        assert timing["performed"] and timing["warmup_iterations"] == 1
        assert len(timing["latency_ms"]["samples"]) == 2
        assert timing["latency_ms"]["p50"] > 0
    else:
        assert timing["latency_ms"] is None
    assert report["test_targets_analyzed"] is False
    assert report["training_success_evaluated"] is False
    floor = report["geometry"]["full_width_thin_platforms"]
    assert report["selected_images"] == 2
    assert floor["ground_truth_boxes"] == 2
    assert floor["with_training_unclipped_anchor"] == (2 if candidate else 0)
    assert floor["with_strict_support_interior_anchor"] == 0
    assert report["all_terrain_gt_have_unclipped_target_anchor"] is candidate
    assert report["geometry"]["actual_assigner_calls"] <= 2
    written = json.loads(path.read_text())
    assert written == {key: value for key, value in report.items() if key != "output"}


@pytest.mark.skipif(not NATIVE, reason="opt-in real candidate CPU training/export smoke")
def test_native_candidate_one_epoch_training_export_and_ort(small_dataset, tmp_path):
    image_loader = Image.open
    result = train_yolo(
        small_dataset,
        tmp_path / "training",
        CANDIDATE,
        epochs=1,
        image_size=160,
        batch=2,
        seed=42,
    )
    exported = export_yolo(Path(result["weights"]), tmp_path / "export", image_size=160)
    manifest = ModelManifest.load(Path(exported["manifest"]))
    detector = OnnxDetector(Path(exported["model"]), Path(exported["manifest"]))
    observations = detector.detect(SceneRenderer().render(Simulator(seed=42).snapshot()))
    assert all(item.class_id in manifest.class_names for item in observations)
    assert exported["smoke_inference"] == "passed"
    assert Image.open is image_loader
    assert result["pretrained"] is False
    # A one-epoch smoke proves the technical chain; no minimum accuracy is asserted.


@pytest.mark.skipif(not NATIVE, reason="opt-in actual unsupported native head rejection")
def test_native_no_dfl_head_is_rejected_without_output(small_dataset, tmp_path):
    config = yaml.safe_load(CANDIDATE.read_text())
    config["reg_max"] = 1
    path = tmp_path / "no-dfl.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="plain Detect DFL"):
        audit_terrain_architecture(small_dataset, path, tmp_path / "report.json", iterations=0)
    assert not (tmp_path / "report.json").exists()
