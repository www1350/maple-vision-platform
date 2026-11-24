"""Training must report and retain the inputs actually consumed by the model."""

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import maple_vision.training as training
from maple_vision.asset_registry import import_asset
from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.sprite_pack import load_sprite_pack

CONFIG = Path(__file__).parents[1] / "configs" / "tiny-yolo.yaml"


def test_custom_sprite_training_snapshot_is_independent_of_source_registry(tmp_path):
    image = tmp_path / "authored.png"
    Image.new("RGBA", (9, 15), (70, 190, 120, 240)).save(image)
    registry = tmp_path / "registry"
    asset = import_asset(
        registry,
        image,
        entity_id="training-fixture-player",
        game_version="fixture-v1",
        source_uri="urn:maple-vision:training-snapshot",
        license="project-authored-test",
        permission="self_authored",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic_fixture",
                "classes": {"player": asset.asset_id},
            }
        )
    )
    pack = load_sprite_pack(registry, mapping)
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=6, frames_per_session=1, seed=7, sprite_pack=pack)
    stage = tmp_path / "stage"
    stage.mkdir()
    training._training_yaml(dataset, stage)
    snapshot = stage / "input_snapshot"
    copied = snapshot / pack.entries[0].source_file
    assert copied.read_bytes() == image.read_bytes()
    assert (snapshot / "manifest.json").read_bytes() == (dataset / "manifest.json").read_bytes()
    shutil.rmtree(dataset)
    shutil.rmtree(registry)
    image.unlink()
    assert validate_dataset(snapshot)["validated"] is True


def test_model_config_replacement_during_snapshot_is_rejected_before_native_runtime(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=3, frames_per_session=1, seed=7)
    config = tmp_path / "model.yaml"
    shutil.copyfile(CONFIG, config)
    original = training._training_yaml

    def replace_after_initial_validation(*args):
        result = original(*args)
        config.write_text(CONFIG.read_text() + "\nactivation: unsupported_external_code\n")
        return result

    def native_must_not_run(*args):
        pytest.fail("changed unsupported model config reached native initialization")

    monkeypatch.setattr(training, "_training_yaml", replace_after_initial_validation)
    monkeypatch.setattr(training, "_offline_runtime", native_must_not_run)
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="safe keys"):
        training.train_yolo(dataset, output, config, epochs=1)
    assert not output.exists()


def test_report_binds_copied_config_and_dataset_even_if_source_files_change(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=3, frames_per_session=1, seed=7)
    replacement = tmp_path / "replacement"
    generate_dataset(replacement, sessions=3, frames_per_session=2, seed=9)
    config = tmp_path / "model.yaml"
    config.write_bytes(CONFIG.read_bytes())
    expected_config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    original = training._training_yaml

    def replace_dataset_before_copy(*args):
        shutil.rmtree(dataset)
        replacement.rename(dataset)
        return original(*args)

    class FakeModel:
        def __init__(self, path, **kwargs):
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_config_hash

        def train(self, **kwargs):
            config.write_text(CONFIG.read_text() + "\n# source changed after native loading\n")
            weights = Path(kwargs["project"]) / "run" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"wrapper contract fixture; not a trained model")
            return SimpleNamespace(results_dict={})

    @contextmanager
    def fake_runtime(directory):
        yield SimpleNamespace(manual_seed=lambda seed: None), SimpleNamespace(YOLO=FakeModel)

    monkeypatch.setattr(training, "_training_yaml", replace_dataset_before_copy)
    monkeypatch.setattr(training, "_offline_runtime", fake_runtime)
    monkeypatch.setattr(training, "_versions", lambda: {"contract_fixture": "1"})
    report = training.train_yolo(dataset, tmp_path / "training", config, epochs=1)
    snapshot = tmp_path / "training" / "input_snapshot"
    assert report["dataset"]["frames"] == 6
    assert (
        report["dataset_manifest_sha256"]
        == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    )
    assert report["model_config_sha256"] == expected_config_hash
    assert report["model_config_sha256"] != hashlib.sha256(config.read_bytes()).hexdigest()


@pytest.mark.parametrize("modified_input", ["model.yaml", "input_snapshot/manifest.json"])
def test_native_mutation_of_captured_training_inputs_rejects_publication(
    tmp_path, monkeypatch, modified_input
):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=3, frames_per_session=1, seed=7)

    class MutatingModel:
        def __init__(self, path, **kwargs):
            self.stage = Path(path).parent

        def train(self, **kwargs):
            target = self.stage / modified_input
            # Whitespace keeps both YAML and JSON semantically valid, so final
            # validation alone cannot establish that the input identity stayed fixed.
            target.write_bytes(target.read_bytes() + b"\n")
            weights = self.stage / "run" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"wrapper mutation fixture")
            return SimpleNamespace(results_dict={})

    @contextmanager
    def fake_runtime(directory):
        yield SimpleNamespace(manual_seed=lambda seed: None), SimpleNamespace(YOLO=MutatingModel)

    monkeypatch.setattr(training, "_offline_runtime", fake_runtime)
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="changed during training"):
        training.train_yolo(dataset, output, CONFIG, epochs=1)
    assert not output.exists()
