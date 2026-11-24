import importlib.util
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.onnx_detector import ModelManifest, OnnxDetector
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.training import (
    _offline_runtime,
    _training_yaml,
    _validate_model_config,
    export_yolo,
    train_yolo,
)

CONFIG = Path(__file__).parents[1] / "configs" / "tiny-yolo.yaml"


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 201},
        {"image_size": 0},
        {"image_size": 63},
        {"image_size": 161},
        {"image_size": 2048},
        {"batch": 0},
        {"batch": True},
        {"batch": 65},
        {"batch": 64, "image_size": 1280},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
    ],
)
def test_training_arguments_rejected_before_creating_output(tmp_path, overrides):
    output = tmp_path / "training"
    with pytest.raises(ValueError):
        train_yolo(tmp_path / "dataset", output, CONFIG, **overrides)
    assert not output.exists()


def test_local_model_required_no_download_by_name(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        train_yolo(tmp_path / "data", tmp_path / "output", Path("yolo26n.yaml"))
    assert not (tmp_path / "output").exists()


def test_fixture_model_config_passes_local_allowlist():
    _validate_model_config(CONFIG)


@pytest.mark.parametrize(
    "modification",
    [
        {"activation": "__import__('os').system('anything')"},
        {"head": [[[-1], 1, "TorchVision", ["resnet50"]]]},
        {"backbone": [[-1, 1, "Conv", ["remote-module"]]]},
        {"width_multiple": float("inf")},
        {"reg_max": True},
        {"nc": 0},
        {"end2end": "false"},
        {"head": [[[-1], 1, "Conv", [16, 3, 1]]]},
        {"backbone": [[99, 1, "Conv", [16, 3, 1]]]},
    ],
)
def test_unsupported_config_rejected_before_ultralytics(tmp_path, modification):
    config = yaml.safe_load(CONFIG.read_text())
    config.update(modification)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        _validate_model_config(path)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset"
    generate_dataset(path, sessions=6, frames_per_session=1, seed=7)
    return path


def test_only_manifest_copies_used_without_mutating_source_dataset(dataset, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    data, names = _training_yaml(dataset.resolve(), stage)
    config = json.loads(data.read_text())
    assert set(config) == {"path", "train", "val", "test", "names"}
    assert len(names) == 7
    manifest = json.loads((dataset / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        paths = (stage / f"{split}.txt").read_text().splitlines()
        assert paths
        assert paths == [
            str((stage / "input_snapshot" / item["image"]).resolve())
            for item in manifest["samples"]
            if item["split"] == split
        ]


def test_existing_training_output_never_overwritten(dataset, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        train_yolo(dataset, output, CONFIG, epochs=1, image_size=160)
    assert sentinel.read_text() == "keep"


def test_tampered_dataset_rejected_before_training(dataset, tmp_path):
    next((dataset / "images").rglob("*.png")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        train_yolo(dataset, tmp_path / "output", CONFIG)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 65},
        {"image_size": True},
        {"scope": "official_game"},
    ],
)
def test_export_validates_parameters_without_loading_weights(tmp_path, kwargs):
    with pytest.raises(ValueError):
        export_yolo(tmp_path / "missing.pt", tmp_path / "export", **kwargs)
    assert not (tmp_path / "export").exists()


def test_export_never_downloads_missing_weights(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        export_yolo(Path("yolo26n.pt"), tmp_path / "export")


def test_missing_heavy_dependencies_restore_environment(tmp_path, monkeypatch):
    import maple_vision.training as training

    original = training.importlib.import_module
    before = os.environ.get("YOLO_OFFLINE")

    def missing(name):
        if name == "torch":
            raise ImportError("not installed")
        return original(name)

    monkeypatch.setattr(training.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="training"):
        with _offline_runtime(tmp_path):
            pytest.fail("runtime should not initialize without dependencies")
    assert os.environ.get("YOLO_OFFLINE") == before


def test_partial_dependency_import_restores_pillow_loader(tmp_path, monkeypatch):
    import maple_vision.training as training

    original_import = training.importlib.import_module
    original_open = Image.open

    def dependency_loader(*args, **kwargs):
        raise AssertionError("third-party Pillow loader must not escape its import")

    def partial_import(name):
        if name == "torch":
            return SimpleNamespace()
        if name == "ultralytics":
            Image.open = dependency_loader
            raise ImportError("failure after an import-time global patch")
        return original_import(name)

    monkeypatch.setattr(training.importlib, "import_module", partial_import)
    with pytest.raises(ValueError, match="training"):
        with _offline_runtime(tmp_path):
            pytest.fail("partial import must not enter the runtime")
    assert Image.open is original_open


def test_pillow_import_patch_is_suppressed_during_runtime_and_restored_on_error(
    tmp_path, monkeypatch
):
    import maple_vision.training as training

    original_open = Image.open
    original_import = training.importlib.import_module
    torch = SimpleNamespace(get_num_threads=lambda: 2, set_num_threads=lambda value: None)
    modules = {
        "torch": torch,
        "ultralytics": SimpleNamespace(),
        "ultralytics.utils": SimpleNamespace(SETTINGS={}, LOGGER=logging.getLogger("pillow-test")),
        "ultralytics.utils.torch_utils": SimpleNamespace(),
        "ultralytics.nn.tasks": SimpleNamespace(),
        "ultralytics.utils.checks": SimpleNamespace(),
        "ultralytics.utils.events": SimpleNamespace(),
    }

    def dependency_loader(*args, **kwargs):
        raise AssertionError("automatic image plugin loading must stay disabled")

    def import_with_patch(name):
        if name == "ultralytics":
            Image.open = dependency_loader
        return modules[name] if name in modules else original_import(name)

    monkeypatch.setattr(training.importlib, "import_module", import_with_patch)
    with pytest.raises(RuntimeError, match="operation failed"):
        with _offline_runtime(tmp_path):
            assert Image.open is original_open
            # Even a later dependency side effect must not persist after an error.
            Image.open = dependency_loader
            raise RuntimeError("operation failed")
    assert Image.open is original_open


def _training_smoke_enabled():
    return os.environ.get("MAPLE_TRAINING_SMOKE") == "1" and all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "ultralytics", "onnx", "onnxruntime")
    )


@pytest.mark.skipif(not _training_smoke_enabled(), reason="opt-in real CPU training/export smoke")
def test_real_cpu_training_export_and_adapter_inference(dataset, tmp_path, capsys):
    original_image_open = Image.open
    result = train_yolo(
        dataset, tmp_path / "training", CONFIG, epochs=1, image_size=160, batch=2, seed=3
    )
    assert Image.open is original_image_open
    assert Path(result["weights"]).is_file()
    assert result["scope"] == "synthetic_fixture"
    assert result["pretrained"] is False
    assert validate_dataset(dataset)["validated"] is True
    assert validate_dataset(tmp_path / "training" / "input_snapshot")["validated"] is True
    exported = export_yolo(Path(result["weights"]), tmp_path / "export", image_size=160)
    assert Image.open is original_image_open
    manifest = ModelManifest.load(Path(exported["manifest"]))
    assert manifest.output_format == "raw_xywh"
    assert manifest.input_width == 160
    assert len(manifest.class_names) == 7
    detector = OnnxDetector(Path(exported["model"]), Path(exported["manifest"]))
    frame = SceneRenderer().render(Simulator(seed=3).snapshot())
    detections = detector.detect(frame)
    assert all(item.class_id in manifest.class_names for item in detections)
    assert exported["smoke_inference"] == "passed"
    # The wrapper keeps programmatic/CLI stdout available for a machine-readable result.
    assert capsys.readouterr().out == ""
