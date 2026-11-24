import hashlib
import json
from dataclasses import asdict, replace

import pytest
from PIL import Image, ImageDraw

from maple_vision.asset_registry import import_asset
from maple_vision.catalog import canonical_json
from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import MonsterState
from maple_vision.simulation import Simulator
from maple_vision.sprite_pack import load_sprite_pack
from maple_vision.vision import PixelPerception


def _pack(tmp_path, *, alpha=255, padding=True, scope="synthetic_fixture"):
    image = Image.new("RGBA", (14, 22), (0, 0, 0, 0))
    box = (4, 3, 9, 18) if padding else (0, 0, 13, 21)
    ImageDraw.Draw(image).rectangle(box, fill=(20, 80, 200, alpha))
    source = tmp_path / "sprite.png"
    image.save(source)
    registry = tmp_path / "registry"
    asset = import_asset(
        registry,
        source,
        entity_id="authored-test-player",
        game_version="fixture-v1",
        source_uri="urn:maple-vision:test-sprite",
        license="project-authored-test",
        permission="self_authored",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": scope,
                "classes": {"player": asset.asset_id},
            }
        )
    )
    return load_sprite_pack(registry, mapping), registry, mapping


def test_registry_pack_loads_exact_source_and_provenance(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path)
    sprite = pack.get("player")
    assert sprite is not None
    assert sprite.asset.entity_id == "authored-test-player"
    assert sprite.asset.source_uri == "urn:maple-vision:test-sprite"
    assert sprite.asset.permission == "self_authored"
    assert sprite.asset.sha256 == hashlib.sha256(sprite.encoded).hexdigest()
    assert pack.get("monster") is None
    assert len(pack.manifest()["pack_sha256"]) == 64


def test_transparent_padding_does_not_paint_or_expand_labels(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path)
    snapshot = Simulator().snapshot()
    sample = SceneRenderer(sprite_pack=pack).render_sample(snapshot)
    player = next(d for d in sample.annotations if d.class_id == "player")
    assert player.bbox == (28, 149, 34, 165)
    without_player = replace(snapshot, player=replace(snapshot.player, alive=False))
    background = SceneRenderer().render(without_player).to_image()
    rendered = sample.frame.to_image()
    assert rendered.getpixel((24, 146)) == background.getpixel((24, 146))
    assert rendered.getpixel((30, 155)) == (20, 80, 200)
    assert not any(d.class_id == "player" for d in PixelPerception().detect(sample.frame))


def test_transparent_holes_reveal_underlying_object_and_keep_its_label(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path)
    snapshot = Simulator().snapshot()
    snapshot = replace(snapshot, monsters=(MonsterState("behind", 24, 146, 14, 22, 36, 36),))
    sample = SceneRenderer(sprite_pack=pack).render_sample(snapshot)
    monster = next(d for d in sample.annotations if d.class_id == "monster")
    assert monster.bbox == (24, 146, 38, 168)
    assert sample.frame.to_image().getpixel((24, 146)) == (238, 82, 113)


def test_fully_opaque_registered_sprite_occludes_underlying_label(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path, padding=False)
    snapshot = replace(
        Simulator().snapshot(), monsters=(MonsterState("behind", 24, 146, 14, 22, 36, 36),)
    )
    sample = SceneRenderer(sprite_pack=pack).render_sample(snapshot)
    assert not any(d.class_id == "monster" for d in sample.annotations)
    assert next(d for d in sample.annotations if d.class_id == "player").bbox == (24, 146, 38, 168)


def test_partial_alpha_blends_pixels_and_preserves_both_visible_instances(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path, alpha=128, padding=False)
    snapshot = replace(
        Simulator().snapshot(), monsters=(MonsterState("behind", 24, 146, 14, 22, 36, 36),)
    )
    sample = SceneRenderer(sprite_pack=pack).render_sample(snapshot)
    assert {d.class_id for d in sample.annotations} >= {"player", "monster"}
    expected = tuple(
        (a * 128 + b * 127 + 127) // 255 for a, b in zip((20, 80, 200), (238, 82, 113), strict=True)
    )
    assert sample.frame.to_image().getpixel((30, 155)) == expected


def test_registry_tampering_is_rejected_before_rendering(tmp_path) -> None:
    pack, registry, mapping = _pack(tmp_path)
    sprite = pack.entries[0]
    (registry / sprite.asset.relative_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_sprite_pack(registry, mapping)


def test_wrong_metadata_provenance_hash_is_rejected(tmp_path) -> None:
    pack, registry, mapping = _pack(tmp_path)
    path = registry / "records" / f"{pack.entries[0].asset.asset_id}.json"
    metadata = json.loads(path.read_text())
    metadata["license"] = "changed without updating identity"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA256"):
        load_sprite_pack(registry, mapping)


def test_unknown_mapping_class_is_rejected(tmp_path) -> None:
    pack, registry, mapping = _pack(tmp_path)
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic_fixture",
                "classes": {"unknown": pack.entries[0].asset.asset_id},
            }
        )
    )
    with pytest.raises(ValueError, match="class"):
        load_sprite_pack(registry, mapping)


def test_invalid_scope_type_is_a_validation_error(tmp_path) -> None:
    pack, registry, mapping = _pack(tmp_path)
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": ["authorized_assets"],
                "classes": {"player": pack.entries[0].asset.asset_id},
            }
        )
    )
    with pytest.raises(ValueError, match="schema"):
        load_sprite_pack(registry, mapping)


def test_fully_transparent_sprite_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="visible pixels"):
        _pack(tmp_path, alpha=0)


def test_default_renderer_remains_byte_identical_to_pre_pack_fixture() -> None:
    snapshot = Simulator(seed=0).snapshot()
    plain = SceneRenderer().render_sample(snapshot)
    explicit_none = SceneRenderer(sprite_pack=None).render_sample(snapshot)
    assert plain == explicit_none
    assert hashlib.sha256(plain.frame.pixels).hexdigest() == (
        "16f1bc035350bb6d86646dd2c7a242b11f25838ea71be7cf77aed1fcb7447ac8"
    )
    assert hashlib.sha256(
        canonical_json([asdict(d) for d in plain.annotations]).encode()
    ).hexdigest() == ("76ee241a4f95b3dea39331313d7f45ca92c779c3e16da93cd3a89f69b31c90af")


def test_sprite_dataset_binds_copied_sources_hashes_and_scope(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path, scope="authorized_assets")
    output = tmp_path / "dataset"
    report = generate_dataset(output, sessions=3, frames_per_session=1, sprite_pack=pack)
    assert report["origin"] == "synthetic-fixture-v1"
    assert report["visual_origin"] == "synthetic_geometry_with_authorized_assets"
    assert report["sprite_pack"]["scope"] == "authorized_assets"
    assert report["sprite_pack"]["pack_sha256"] == pack.manifest()["pack_sha256"]
    sprite = pack.entries[0]
    assert (output / sprite.source_file).read_bytes() == sprite.encoded
    assert validate_dataset(output)["validated"] is True
    (output / sprite.source_file).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(output)


def test_wrong_pack_rgba_hash_rejected_even_after_manifest_hash_recalculation(tmp_path) -> None:
    pack, _, _ = _pack(tmp_path)
    output = tmp_path / "dataset"
    generate_dataset(output, sessions=3, frames_per_session=1, sprite_pack=pack)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    metadata = manifest["sprite_pack"]
    metadata["classes"]["player"]["rgba_sha256"] = "0" * 64
    metadata.pop("pack_sha256")
    metadata["pack_sha256"] = hashlib.sha256(canonical_json(metadata).encode()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(output)


def test_default_dataset_manifest_is_unchanged_by_explicit_none(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate_dataset(first, sessions=3, frames_per_session=1, seed=7)
    generate_dataset(second, sessions=3, frames_per_session=1, seed=7, sprite_pack=None)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert "sprite_pack" not in json.loads((first / "manifest.json").read_text())
    assert not (first / "sprite_objects").exists()
