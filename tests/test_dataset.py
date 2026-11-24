import hashlib
import json

import pytest

from maple_vision.dataset import generate_dataset, validate_dataset


@pytest.fixture
def dataset(tmp_path):
    output = tmp_path / "dataset"
    generate_dataset(output, sessions=3, frames_per_session=2, seed=123)
    return output


def test_dataset_is_session_disjoint_and_reproducible(dataset, tmp_path) -> None:
    summary = validate_dataset(dataset)
    assert summary["frames"] == 6 and summary["sessions"] == 3
    assert summary["split_frames"] == {"train": 2, "val": 2, "test": 2}
    assert summary["missing_classes"] == {"train": [], "val": [], "test": []}
    assert all(counts["loot"] > 0 for counts in summary["class_counts"].values())
    second = tmp_path / "repeat"
    generate_dataset(second, sessions=3, frames_per_session=2, seed=123)
    first_manifest = json.loads((dataset / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["origin"] == "synthetic-fixture-v1"
    session_splits = {}
    for sample in first_manifest["samples"]:
        session_splits.setdefault(sample["session_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in session_splits.values())


def test_generation_refuses_overwrite(dataset) -> None:
    original = (dataset / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_dataset(dataset)
    assert (dataset / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("sessions,frames", [(2, 2), (True, 2), (3, 0), (1000, 1000)])
def test_dataset_limits_are_enforced(tmp_path, sessions, frames) -> None:
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "bad", sessions=sessions, frames_per_session=frames)
    assert not (tmp_path / "bad").exists()


def test_corrupted_image_is_detected(dataset) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text())
    image = dataset / manifest["samples"][0]["image"]
    image.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(dataset)


def test_out_of_bounds_annotation_fails_even_after_hash_recomputed(dataset) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    sample = manifest["samples"][0]
    label = dataset / sample["label"]
    label.write_text("0 0.99 0.5 0.5 0.2\n")
    sample["objects"] = 1
    sample["label_sha256"] = hashlib.sha256(label.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside image"):
        validate_dataset(dataset)


def test_session_split_tampering_is_rejected(dataset) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    sample = manifest["samples"][0]
    sample["split"] = "test" if sample["split"] != "test" else "train"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(dataset)


def test_traversal_in_manifest_is_rejected(dataset) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["image"] = "../outside.png"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        validate_dataset(dataset)


def test_duplicate_image_hash_across_splits_is_rejected(dataset) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    first, second = manifest["samples"][0], manifest["samples"][2]
    assert first["split"] != second["split"]
    (dataset / second["image"]).write_bytes((dataset / first["image"]).read_bytes())
    second["image_sha256"] = first["image_sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical image"):
        validate_dataset(dataset)


def test_unlisted_training_images_are_rejected(dataset) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text())
    image = dataset / manifest["samples"][0]["image"]
    (image.parent / "unlisted.png").write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match="unlisted"):
        validate_dataset(dataset)


def test_training_configuration_cannot_redirect_to_other_data(dataset) -> None:
    (dataset / "data.yaml").write_text('{"train": "../external"}')
    with pytest.raises(ValueError, match="configuration"):
        validate_dataset(dataset)


def test_training_redirect_still_fails_after_rehash(dataset) -> None:
    yaml = dataset / "data.yaml"
    yaml.write_text('{"train": "../external"}')
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["data_yaml_sha256"] = hashlib.sha256(yaml.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration disagrees"):
        validate_dataset(dataset)


def test_default_sized_dataset_has_all_fixture_maps_and_classes_in_each_split(tmp_path) -> None:
    summary = generate_dataset(
        tmp_path / "nine_sessions", sessions=9, frames_per_session=2, seed=42
    )
    assert summary["split_sessions"] == {"train": 5, "val": 2, "test": 2}
    assert summary["missing_maps"] == {"train": [], "val": [], "test": []}
    assert summary["missing_classes"] == {"train": [], "val": [], "test": []}
