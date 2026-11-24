import json
import os

import pytest
from PIL import Image

from maple_vision.asset_registry import import_asset, list_assets, verify_asset


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "sprite.png"
    Image.new("RGBA", (24, 32), (120, 20, 80, 255)).save(path)
    return path


def add(registry, image, **overrides):
    metadata = dict(
        entity_id="fixture.player",
        game_version="fixture-v1",
        source_uri="urn:self:fixture",
        license="CC0-1.0",
        permission="self_authored",
    )
    metadata.update(overrides)
    return import_asset(registry, image, **metadata)


def test_asset_import_deduplicates_content_and_preserves_metadata(tmp_path, image):
    registry = tmp_path / "registry"
    first = add(registry, image)
    assert add(registry, image) == first
    assert verify_asset(registry, first.asset_id) == first
    assert (first.width, first.height, first.media_type) == (24, 32, "image/png")
    second = add(registry, image, source_uri="urn:other:fixture")
    assert first.asset_id != second.asset_id
    assert first.sha256 == second.sha256
    assert len(list((registry / "objects").iterdir())) == 1
    assert set(list_assets(registry)) == {first, second}
    assert image.read_bytes() == (registry / first.relative_path).read_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"license": ""},
        {"permission": "unknown"},
        {"entity_id": ""},
        {"game_version": ""},
        {"source_uri": "https://secret@example.org"},
    ],
)
def test_invalid_provenance_leaves_no_registry(tmp_path, image, overrides):
    registry = tmp_path / "registry"
    with pytest.raises(ValueError):
        add(registry, image, **overrides)
    assert not registry.exists()


def test_corrupt_image_rejected_before_registry_creation(tmp_path):
    image = tmp_path / "bad.png"
    image.write_text("not an image")
    with pytest.raises(ValueError, match="decoded"):
        add(tmp_path / "registry", image)
    assert not (tmp_path / "registry").exists()


def test_truncated_image_rejected(tmp_path, image):
    image.write_bytes(image.read_bytes()[:40])
    with pytest.raises(ValueError):
        add(tmp_path / "registry", image)


def test_tampered_object_is_rejected_on_verify_and_reimport(tmp_path, image):
    registry = tmp_path / "registry"
    entry = add(registry, image)
    (registry / entry.relative_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        verify_asset(registry, entry.asset_id)
    with pytest.raises(ValueError, match="immutable"):
        add(registry, image)


def test_tampered_metadata_is_rejected(tmp_path, image):
    registry = tmp_path / "registry"
    entry = add(registry, image)
    metadata_path = registry / "records" / f"{entry.asset_id}.json"
    value = json.loads(metadata_path.read_text())
    value["relative_path"] = "../../private.png"
    metadata_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="SHA256"):
        verify_asset(registry, entry.asset_id)


@pytest.mark.parametrize("identifier", ["../secret", "f" * 63, "F" * 64, True])
def test_asset_id_cannot_escape_registry(tmp_path, identifier):
    with pytest.raises(ValueError, match="SHA256"):
        verify_asset(tmp_path / "registry", identifier)


def test_symlink_object_is_rejected(tmp_path, image):
    if os.name == "nt":
        pytest.skip("symlink creation is not available to unprivileged Windows CI users")
    registry = tmp_path / "registry"
    entry = add(registry, image)
    target = registry / entry.relative_path
    target.unlink()
    target.symlink_to(image)
    with pytest.raises(ValueError, match="symlink"):
        verify_asset(registry, entry.asset_id)


def test_registry_cannot_be_redirected_by_subdirectory_symlink(tmp_path, image):
    if os.name == "nt":
        pytest.skip("symlink creation is not available to unprivileged Windows CI users")
    registry = tmp_path / "registry"
    registry.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (registry / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        add(registry, image)
    assert list(outside.iterdir()) == []


def test_unsupported_image_format_rejected(tmp_path):
    image = tmp_path / "sprite.gif"
    Image.new("RGB", (3, 3)).save(image)
    with pytest.raises(ValueError, match="format"):
        add(tmp_path / "registry", image)
