"""Animation timing, alpha geometry and exported-source integrity counterexamples."""

import hashlib
import json
import os
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import maple_vision.animation_pack as animation
from maple_vision.animation_pack import (
    AnimationPack,
    AnimationPlacement,
    AnimationSource,
    compose_animation,
    export_animation_pack,
    load_animation_pack,
    verify_animation_pack,
)
from maple_vision.asset_registry import import_asset
from maple_vision.catalog import canonical_json


def add(
    registry,
    source,
    *,
    color="red",
    bbox=(2, 3, 5, 8),
    entity="fixture.player",
    version="fixture-v1",
):
    image = Image.new("RGBA", (8, 10))
    if bbox is not None:
        ImageDraw.Draw(image).rectangle(bbox, fill=color)
    image.save(source)
    return import_asset(
        registry,
        source,
        entity_id=entity,
        game_version=version,
        source_uri="urn:self:animation-test-fixture",
        license="CC0-1.0",
        permission="self_authored",
    )


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rehash(manifest):
    payload = {key: value for key, value in manifest.items() if key != "pack_sha256"}
    manifest["pack_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@pytest.fixture
def assets(tmp_path):
    registry = tmp_path / "registry"
    first = add(registry, tmp_path / "first.png")
    second = add(registry, tmp_path / "second.png", color="blue", bbox=(1, 1, 6, 7))
    mapping = tmp_path / "animations.json"
    frames = [
        {"asset_id": first.asset_id, "duration_ms": 80, "anchor": [4, 10]},
        {"asset_id": second.asset_id, "duration_ms": 120, "anchor": [4, 10]},
    ]
    data = {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "clips": [
            {
                "clip_id": "player.idle",
                "entity_id": "fixture.player",
                "game_version": "fixture-v1",
                "class_id": "player",
                "action_state": "idle",
                "playback": "loop",
                "frames": frames,
            },
            {
                "clip_id": "player.attack",
                "entity_id": "fixture.player",
                "game_version": "fixture-v1",
                "class_id": "player",
                "action_state": "attack",
                "playback": "once",
                "frames": [dict(frames[0], duration_ms=40), dict(frames[1], duration_ms=60)],
            },
        ],
    }
    write(mapping, data)
    return registry, mapping, data, first, second


@pytest.fixture
def pack(assets):
    registry, mapping, *_ = assets
    return load_animation_pack(registry, mapping)


@pytest.mark.parametrize(
    ("time", "index", "cycle", "within"),
    [
        (0, 0, 0, 0),
        (79, 0, 0, 79),
        (80, 1, 0, 0),
        (199, 1, 0, 119),
        (200, 0, 1, 0),
        (279, 0, 1, 79),
        (1_000_080, 1, 5000, 0),
    ],
)
def test_integer_loop_timeline_has_half_open_frame_boundaries(pack, time, index, cycle, within):
    sample = pack.get("player.idle").sample(time)
    assert (sample.frame_index, sample.cycle_index, sample.frame_elapsed_ms, sample.finished) == (
        index,
        cycle,
        within,
        False,
    )


@pytest.mark.parametrize(
    ("time", "index", "finished", "within"),
    [
        (0, 0, False, 0),
        (39, 0, False, 39),
        (40, 1, False, 0),
        (99, 1, False, 59),
        (100, 1, True, 59),
        (2**53 - 1, 1, True, 59),
    ],
)
def test_once_timeline_holds_last_frame_after_completion(pack, time, index, finished, within):
    sample = pack.get("player.attack").sample(time)
    assert (sample.frame_index, sample.finished, sample.frame_elapsed_ms, sample.cycle_index) == (
        index,
        finished,
        within,
        0,
    )


@pytest.mark.parametrize("time", [True, -1, 0.5, float("nan"), 2**53])
def test_timeline_rejects_noninteger_or_unbounded_time(pack, time):
    with pytest.raises(ValueError):
        pack.get("player.idle").sample(time)


def test_versioned_action_resolution_does_not_fallback_to_another_version(pack):
    assert pack.resolve("fixture.player", "fixture-v1", "attack").clip_id == "player.attack"
    assert pack.resolve("fixture.player", "fixture-v2", "attack") is None
    assert pack.resolve("fixture.player", "fixture-v1", "hurt") is None
    with pytest.raises(FrozenInstanceError):
        pack.clips[0].playback = "loop"
    manifest = pack.manifest()
    manifest["clips"][0]["frames"][0]["anchor"][0] = 999
    assert pack.clips[0].frames[0].anchor == (4, 10)


def test_transparent_source_bbox_anchor_and_canvas_clipping(pack):
    source = pack.get("player.idle").frames[0].source
    assert source.visible_bbox == (2, 3, 6, 9)
    result = compose_animation(pack, (AnimationPlacement("player.idle", 0, (1, 5)),), size=(12, 12))
    assert result.placements[0].canvas_bounds == (-3, -5, 5, 5)
    assert result.placements[0].visible_bbox == result.visible_bbox == (0, 0, 3, 4)
    assert result.image().getpixel((2, 3)) == (255, 0, 0, 255)
    assert result.image().getpixel((3, 3)) == (0, 0, 0, 0)
    invisible = compose_animation(
        pack, (AnimationPlacement("player.idle", 0, (-100, -100)),), size=(12, 12)
    )
    assert invisible.visible_bbox is None and invisible.placements[0].visible_bbox is None


def test_mirroring_mirrors_pixel_edge_anchor_in_addition_to_pixels(pack):
    clip = pack.get("player.idle")
    clip = replace(clip, frames=(replace(clip.frames[0], anchor=(1, 10)),))
    one = AnimationPack("synthetic_fixture", (clip,))
    normal = compose_animation(one, (AnimationPlacement(clip.clip_id, 0, (4, 10)),), size=(12, 12))
    mirrored = compose_animation(
        one, (AnimationPlacement(clip.clip_id, 0, (4, 10), flip_x=True),), size=(12, 12)
    )
    assert normal.placements[0].canvas_bounds == (3, 0, 11, 10)
    assert mirrored.placements[0].canvas_bounds == (-3, 0, 5, 10)
    assert normal.visible_bbox == (5, 3, 9, 9)
    assert mirrored.visible_bbox == (0, 3, 3, 9)


def test_layer_order_is_stable_and_composition_is_repeatable(pack, tmp_path):
    placements = (
        AnimationPlacement("player.idle", 80, (8, 10), layer=0),
        AnimationPlacement("player.idle", 0, (8, 10), layer=1),
    )
    first = compose_animation(pack, placements, size=(16, 16))
    second = compose_animation(pack, placements, size=(16, 16))
    assert first.rgba == second.rgba and first.png_bytes() == second.png_bytes()
    assert first.image().getpixel((6, 3)) == (255, 0, 0, 255)
    image = first.image()
    image.putpixel((6, 3), (0, 0, 0, 0))
    assert first.image().getpixel((6, 3)) == (255, 0, 0, 255)
    output = tmp_path / "composite.png"
    first.save_png(output)
    with Image.open(output) as image:
        assert image.tobytes() == first.rgba
    with pytest.raises(FileExistsError):
        first.save_png(output)


def test_export_is_self_contained_after_all_original_paths_are_deleted(pack, assets, tmp_path):
    registry, mapping, *_ = assets
    placement = (AnimationPlacement("player.attack", 100, (8, 12)),)
    before = compose_animation(pack, placement, size=(16, 16)).png_bytes()
    shutil.rmtree(registry)
    mapping.unlink()
    (tmp_path / "first.png").unlink()
    (tmp_path / "second.png").unlink()
    exported = tmp_path / "exported"
    manifest = export_animation_pack(pack, exported)
    actual = verify_animation_pack(exported)
    assert actual.manifest() == manifest == pack.manifest()
    assert compose_animation(actual, placement, size=(16, 16)).png_bytes() == before
    assert len(list((exported / "animation_objects").iterdir())) == 2
    with pytest.raises(ValueError, match="already exists"):
        export_animation_pack(pack, exported)


@pytest.mark.parametrize("failure", ["source", "manifest"])
def test_export_io_failure_rolls_back_owned_files_and_allows_retry(
    pack, tmp_path, monkeypatch, failure
):
    output = tmp_path / "export"
    original = Path.open
    source = pack.sources[-1].source_file

    def fail_one(path, mode="r", *args, **kwargs):
        if mode == "xb" and (
            failure == "source"
            and path == output / source
            or failure == "manifest"
            and path == output / "animation-pack.json"
        ):
            raise OSError("review injected output failure")
        return original(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_one)
        with pytest.raises(OSError, match="injected"):
            export_animation_pack(pack, output)
    assert not output.exists()
    export_animation_pack(pack, output)
    assert verify_animation_pack(output).manifest() == pack.manifest()


@pytest.mark.parametrize(
    "change",
    [
        "schema-bool",
        "schema-float",
        "empty",
        "playback",
        "playback-list",
        "duration-zero",
        "duration-bool",
        "duration-budget",
        "negative-anchor",
        "anchor-x",
        "anchor-y",
        "anchor-float",
        "unknown-state",
        "duplicate-id",
        "duplicate-binding",
        "unknown-field",
    ],
)
def test_invalid_mappings_fail_before_rendering(assets, change):
    registry, mapping, data, *_ = assets
    frame = data["clips"][0]["frames"][0]
    if change.startswith("schema"):
        data["schema_version"] = True if change.endswith("bool") else 1.0
    elif change == "empty":
        data["clips"] = []
    elif change == "playback":
        data["clips"][0]["playback"] = "random"
    elif change == "playback-list":
        data["clips"][0]["playback"] = []
    elif change == "duration-zero":
        frame["duration_ms"] = 0
    elif change == "duration-bool":
        frame["duration_ms"] = True
    elif change == "duration-budget":
        frame["duration_ms"] = 60_000
    elif change == "negative-anchor":
        frame["anchor"] = [-1, 0]
    elif change == "anchor-x":
        frame["anchor"] = [9, 10]
    elif change == "anchor-y":
        frame["anchor"] = [4, 11]
    elif change == "anchor-float":
        frame["anchor"] = [4.0, 10]
    elif change == "unknown-state":
        data["clips"][0]["action_state"] = "unknown"
    elif change == "duplicate-id":
        data["clips"][1]["clip_id"] = data["clips"][0]["clip_id"]
    elif change == "duplicate-binding":
        data["clips"][1]["action_state"] = "idle"
    else:
        frame["fallback"] = True
    write(mapping, data)
    with pytest.raises(ValueError):
        load_animation_pack(registry, mapping)


@pytest.mark.parametrize("wrong", ["entity", "version", "blank"])
def test_registered_source_is_still_rejected_when_wrong_for_the_clip(assets, tmp_path, wrong):
    registry, mapping, data, *_ = assets
    other = add(
        registry,
        tmp_path / "other.png",
        entity="fixture.other" if wrong == "entity" else "fixture.player",
        version="fixture-v2" if wrong == "version" else "fixture-v1",
        bbox=None if wrong == "blank" else (2, 3, 5, 8),
    )
    data["clips"][0]["frames"][0]["asset_id"] = other.asset_id
    write(mapping, data)
    with pytest.raises(ValueError):
        load_animation_pack(registry, mapping)


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "rgba",
        "bbox",
        "bbox-float",
        "source-content",
        "unlisted",
        "source-metadata",
        "missing-source",
    ],
)
def test_export_verification_rejects_tampering_even_with_rehashed_manifest(pack, tmp_path, change):
    output = tmp_path / "export"
    manifest = export_animation_pack(pack, output)
    identifier = next(iter(manifest["sources"]))
    source = manifest["sources"][identifier]
    if change == "path":
        source["source_file"] = "../private.png"
    elif change == "rgba":
        source["rgba_sha256"] = "0" * 64
    elif change == "bbox":
        source["visible_bbox"] = [0, 0, 8, 10]
    elif change == "bbox-float":
        source["visible_bbox"] = [float(value) for value in source["visible_bbox"]]
    elif change == "source-content":
        (output / source["source_file"]).write_bytes(b"forged")
    elif change == "unlisted":
        (output / "animation_objects" / "extra.png").write_bytes(b"unlisted")
    elif change == "source-metadata":
        source["asset"]["source_uri"] = "urn:forged:source"
    else:
        (output / source["source_file"]).unlink()
    rehash(manifest)
    write(output / "animation-pack.json", manifest)
    with pytest.raises(ValueError):
        verify_animation_pack(output)


def test_source_symlink_is_rejected_for_registry_and_export(pack, assets, tmp_path):
    if os.name == "nt":
        pytest.skip("symlink creation is not available to unprivileged Windows CI users")
    registry, mapping, _, first, _ = assets
    target = registry / first.relative_path
    external = tmp_path / "external.png"
    target.rename(external)
    target.symlink_to(external)
    with pytest.raises(ValueError, match="nonsymlink"):
        load_animation_pack(registry, mapping)
    output = tmp_path / "export"
    manifest = export_animation_pack(pack, output)
    source = next(iter(manifest["sources"].values()))
    target = output / source["source_file"]
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="nonsymlink"):
        verify_animation_pack(output)


def test_decoded_pixel_budget_is_checked_before_opening_any_image(assets, monkeypatch):
    registry, mapping, *_ = assets
    monkeypatch.setattr(animation, "MAX_PACK_PIXELS", 100)

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate pixel budget must precede image decoding")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ValueError, match="decoded-pixel budget"):
        load_animation_pack(registry, mapping)


def test_source_byte_budget_precedes_hashing_or_image_decoding(assets, monkeypatch):
    registry, mapping, _, first, _ = assets
    with (registry / first.relative_path).open("wb") as stream:
        stream.truncate(animation.MAX_SOURCE_BYTES + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("source byte budget must precede unbounded reads and decoding")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        load_animation_pack(registry, mapping)


def test_manifest_count_limit_precedes_source_access(assets, monkeypatch):
    registry, mapping, data, *_ = assets
    data["clips"] = [data["clips"][0]] * (animation.MAX_CLIPS + 1)
    write(mapping, data)

    def forbidden(*args, **kwargs):
        raise AssertionError("clip count must precede registry traversal")

    monkeypatch.setattr(animation, "_directory", forbidden)
    with pytest.raises(ValueError, match="count budget"):
        load_animation_pack(registry, mapping)


def test_canvas_and_pixel_work_budgets_precede_image_allocation(pack, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("canvas and work budgets must precede image allocation")

    monkeypatch.setattr(Image, "new", forbidden)
    placements = (AnimationPlacement("player.idle", 0, (0, 0)),)
    with pytest.raises(ValueError, match="pixel budget"):
        compose_animation(pack, placements, size=(4096, 4096))
    monkeypatch.setattr(animation, "MAX_COMPOSITION_WORK_PIXELS", 100)
    with pytest.raises(ValueError, match="pixel-work budget"):
        compose_animation(pack, placements, size=(10, 10))


def test_encoded_pixels_are_reverified_for_direct_source_construction(assets):
    registry, _, _, first, _ = assets
    original = (registry / first.relative_path).read_bytes()
    with pytest.raises(ValueError, match="SHA256"):
        AnimationSource(first, original + b"changed")
    source = AnimationSource(first, original)
    mutable = source.image()
    mutable.putpixel((2, 3), (0, 0, 0, 0))
    assert source.image().getpixel((2, 3)) == (255, 0, 0, 255)


def test_repository_self_drawn_fixture_verifies_without_an_import_registry():
    fixture = Path(__file__).resolve().parents[1] / "examples" / "animations" / "fixture"
    pack = verify_animation_pack(fixture)
    assert pack.scope == "synthetic_fixture" and len(pack.clips) == 3 and len(pack.sources) == 4
    assert {clip.action_state for clip in pack.clips} == {"idle", "move", "attack"}
    assert all(source.asset.permission == "self_authored" for source in pack.sources)
    assert all(source.asset.game_version == "synthetic-animation-v1" for source in pack.sources)
