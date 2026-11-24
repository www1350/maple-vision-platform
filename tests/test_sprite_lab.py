import hashlib
import json
import shutil

import pytest
from PIL import Image

from maple_vision.asset_registry import import_asset
from maple_vision.lab import RunConfig, run_episode, verify_run
from maple_vision.scene import ImageFrame, SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.sprite_pack import load_sprite_pack


@pytest.fixture
def sprite_run(tmp_path):
    registry = tmp_path / "registry"
    source = tmp_path / "player.png"
    Image.new("RGBA", (14, 22), (20, 80, 200, 255)).save(source)
    entry = import_asset(
        registry,
        source,
        entity_id="custom-player",
        game_version="fixture-v1",
        source_uri="urn:maple-vision:custom-player-test",
        license="self-authored-test",
        permission="self_authored",
    )
    mapping = tmp_path / "sprite-map.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic_fixture",
                "classes": {"player": entry.asset_id},
            }
        )
    )
    pack = load_sprite_pack(registry, mapping)
    output = tmp_path / "run"
    summary = run_episode(output, RunConfig(max_steps=30, record_stride=1), sprite_pack=pack)
    return output, summary, pack, registry, source


def test_custom_art_does_not_silently_fall_back_to_ground_truth(sprite_run) -> None:
    output, summary, pack, _, _ = sprite_run
    assert summary["status"] == "stopped"
    assert summary["reason"] == "unknown_observation"
    assert summary["independent_goal_check"] is False
    assert summary["sprite_pack"] == pack.manifest()
    assert summary["detector"] == "pixel-fixture"
    assert set(summary["actions"]) == {"idle"}
    frame = ImageFrame.from_file(output / "frames" / "000000.png")
    assert frame.to_image().getpixel((30, 155)) == (20, 80, 200)
    assert frame.pixels != SceneRenderer().render(Simulator().snapshot()).pixels
    first = json.loads((output / "trace.jsonl").read_text().splitlines()[0])
    assert not any(d["class_id"] == "player" for d in first["observation"]["detections"])


def test_custom_art_replays_without_original_registry_or_source(sprite_run) -> None:
    output, summary, pack, registry, source = sprite_run
    shutil.rmtree(registry)
    source.unlink()
    replay = verify_run(output)
    assert replay["verified"] is True
    assert replay["steps_replayed"] == summary["steps"]
    assert replay["perception_reproduced"] is False
    assert replay["independent_goal_check"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert "sprite-pack.json" in manifest["files"]
    assert pack.entries[0].source_file in manifest["files"]


def test_default_bundle_remains_compatible_and_uses_original_pixels(tmp_path) -> None:
    output = tmp_path / "default"
    summary = run_episode(output, RunConfig(max_steps=3, record_stride=1))
    assert "sprite_pack" not in summary
    assert not (output / "sprite-pack.json").exists()
    assert not (output / "sprite_objects").exists()
    frame = ImageFrame.from_file(output / "frames" / "000000.png")
    assert frame.pixels == SceneRenderer().render(Simulator().snapshot()).pixels
    replay = verify_run(output)
    assert replay["verified"] is True and replay["steps_replayed"] == 3


def test_corrupted_embedded_asset_is_rejected(sprite_run) -> None:
    output, _, pack, _, _ = sprite_run
    path = output / pack.entries[0].source_file
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(output)


def test_asset_corruption_is_detected_after_outer_hash_is_recomputed(sprite_run) -> None:
    output, _, pack, _, _ = sprite_run
    relative = pack.entries[0].source_file
    path = output / relative
    path.write_bytes(b"tampered")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sprite source checksum"):
        verify_run(output)


def test_run_manifest_must_cover_each_embedded_source(sprite_run) -> None:
    output, _, pack, _, _ = sprite_run
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop(pack.entries[0].source_file)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits sprite source"):
        verify_run(output)


def test_unmanifested_sprite_metadata_is_rejected(sprite_run) -> None:
    output, _, _, _, _ = sprite_run
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("sprite-pack.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unmanifested sprite"):
        verify_run(output)


def test_pack_summary_must_match_embedded_provenance(sprite_run) -> None:
    output, _, _, _, _ = sprite_run
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sprite_pack"]["scope"] = "authorized_assets"
    summary_path.write_text(json.dumps(summary))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sprite metadata disagrees"):
        verify_run(output)
