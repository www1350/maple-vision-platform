"""Independent animation integration review: authored pixels, no training or game data."""

import copy
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import maple_vision.lab as lab_module
import maple_vision.scene_animation as scene_module
from maple_vision.animation_pack import (
    AnimationClip,
    AnimationFrame,
    AnimationPack,
    AnimationSource,
)
from maple_vision.asset_registry import import_asset
from maple_vision.catalog import canonical_json
from maple_vision.cli import main
from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.evaluation import benchmark_dataset
from maple_vision.lab import RunConfig, run_episode, verify_perception, verify_run
from maple_vision.map_pack import MapDefinition, MapPack
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.scene import PALETTE, SceneRenderer
from maple_vision.scene_animation import (
    SceneAnimation,
    export_scene_animation,
    load_scene_animation,
)
from maple_vision.sim_types import NPC, Ladder, MonsterState, Platform, Portal, SimAction
from maple_vision.simulation import Simulator
from maple_vision.training import _training_yaml


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _scene(root, images, anchors, durations, class_id="player"):
    """Build actual registered sources; expected pixels below are independently specified."""
    root.mkdir(parents=True)
    frames = []
    for index, (image, anchor, duration) in enumerate(zip(images, anchors, durations, strict=True)):
        source = root / f"authored-{index}.png"
        image.save(source)
        entry = import_asset(
            root / "registry",
            source,
            entity_id=f"review.{class_id}",
            game_version="review-v1",
            source_uri="urn:independent-review:self-authored-pixels",
            license="CC0-1.0",
            permission="self_authored",
        )
        frames.append(AnimationFrame(AnimationSource(entry, source.read_bytes()), duration, anchor))
    clip = AnimationClip(
        f"review.{class_id}.fixed",
        f"review.{class_id}",
        "review-v1",
        class_id,
        "idle",
        "loop",
        tuple(frames),
    )
    return SceneAnimation(AnimationPack("synthetic_fixture", (clip,)), ((class_id, clip.clip_id),))


def _palette_scene(root):
    first = Image.new("RGBA", (14, 22), (*PALETTE["player"], 255))
    second = first.copy()
    # Different pixels, same detectable player bounds: animation affects evidence.
    second.putpixel((0, 0), (0, 0, 0, 0))
    second.putpixel((13, 21), (0, 0, 0, 0))
    return _scene(root, (first, second), ((7, 22), (7, 22)), (30, 70))


def _snapshot(**changes):
    base = Simulator(seed=123).snapshot()
    base = replace(
        base,
        sequence=17,
        timestamp_ms=0,
        player=replace(base.player, x=40, y=40),
        monsters=(),
        drops=(),
        platforms=(),
        ladders=(),
        portals=(),
        npcs=(),
        dialogue=None,
    )
    return replace(base, **changes)


def _boxes(sample, class_id):
    return tuple(
        detection.bbox for detection in sample.annotations if detection.class_id == class_id
    )


def _rehash_scene(document):
    pack = document["content"]["pack"]
    pack.pop("pack_sha256")
    pack["pack_sha256"] = _sha(canonical_json(pack).encode())
    document["content_sha256"] = _sha(canonical_json(document["content"]).encode())


def _rehash_run(root):
    manifest = _read(root / "manifest.json")
    manifest["files"] = {name: _sha((root / name).read_bytes()) for name in manifest["files"]}
    _write(root / "manifest.json", manifest)


def _maps():
    maps = []
    for index, (identifier, destination) in enumerate(
        (("review_field", "review_cave"), ("review_cave", "review_field"))
    ):
        maps.append(
            MapDefinition(
                id=identifier,
                theme="meadow" if index == 0 else "cavern",
                marker=(9 + index * 25, 22 + index * 17, 33 + index * 23),
                spawn_x=30,
                platforms=(Platform(0, 320, 168), Platform(80, 150, 105)),
                portals=(Portal("exit", 285, 132, 20, 36, destination),),
                npcs=(NPC("guide", 65, 144, 14, 24),),
                ladders=(Ladder(100, 105, 10, 63),),
                monster_x=(135, 210),
            )
        )
    return MapPack(
        "review_maps",
        "1.0.0",
        "urn:independent-review:maps",
        "self_authored",
        "CC0-1.0",
        tuple(maps),
        ("review_field", "review_cave"),
    )


def test_native_size_foot_anchor_uses_absolute_time_and_actual_alpha(tmp_path):
    first = Image.new("RGBA", (8, 10), (0, 0, 0, 0))
    ImageDraw.Draw(first).rectangle((2, 3, 5, 8), fill=(*PALETTE["player"], 255))
    second = Image.new("RGBA", (8, 10), (0, 0, 0, 0))
    ImageDraw.Draw(second).rectangle((1, 1, 3, 6), fill=(*PALETTE["player"], 255))
    animation = _scene(tmp_path / "art", (first, second), ((1, 9), (6, 8)), (30, 70))
    renderer = SceneRenderer(animation=animation)
    # Physical body is 40,40..54,62. Native images attach to its foot (47,62).
    expected = {
        0: (48, 56, 52, 62),
        29: (48, 56, 52, 62),
        30: (42, 55, 45, 61),
        99: (42, 55, 45, 61),
        100: (48, 56, 52, 62),
    }
    for time, box in expected.items():
        snapshot = _snapshot(timestamp_ms=time)
        sample = renderer.render_sample(snapshot)
        assert _boxes(sample, "player") == (box,)
        assert sample.frame.to_image().getpixel((box[0], box[1])) == PALETTE["player"]
        # Sequence identity and unrelated events do not reset the absolute clip clock.
        changed = replace(snapshot, sequence=999, events=("attack_hit",))
        assert renderer.render(changed).pixels == sample.frame.pixels
    assert (
        renderer.render(_snapshot(timestamp_ms=30)).pixels
        != renderer.render(_snapshot(timestamp_ms=29)).pixels
    )


@pytest.mark.parametrize("alpha", [0, 128, 255])
def test_unregistered_monster_is_occluded_by_actual_animation_alpha(tmp_path, alpha):
    image = Image.new("RGBA", (28, 30), (*PALETTE["player"], 255))
    # Placement is (33,32). Window corresponds to world x42..46,y44..52.
    ImageDraw.Draw(image).rectangle((9, 12, 12, 19), fill=(*PALETTE["player"], alpha))
    animation = _scene(tmp_path / "art", (image,), ((14, 30),), (100,))
    snapshot = _snapshot(monsters=(MonsterState("test", 42, 44, 8, 8, 10, 10),))
    sample = SceneRenderer(animation=animation).render_sample(snapshot)
    # No SpritePack is involved; the animation must activate the real-alpha mask path.
    assert _boxes(sample, "monster") == (() if alpha == 255 else ((42, 44, 46, 52),))
    expected = tuple(
        (foreground * alpha + background * (255 - alpha) + 127) // 255
        for foreground, background in zip(PALETTE["player"], PALETTE["monster"], strict=True)
    )
    assert sample.frame.to_image().getpixel((42, 44)) == expected


def test_animated_monster_labels_are_clipped_by_hud_and_dialogue(tmp_path):
    image = Image.new("RGBA", (20, 20), (*PALETTE["monster"], 255))
    animation = _scene(tmp_path / "art", (image,), ((10, 20),), (100,), "monster")
    snapshot = _snapshot(monsters=(MonsterState("test", 60, 25, 10, 10, 10, 10),))
    sample = SceneRenderer(animation=animation).render_sample(snapshot)
    assert _boxes(sample, "monster") == ((55, 27, 75, 35),)
    # At this location the panel covers every native source pixel.
    blocked = replace(
        snapshot, monsters=(MonsterState("test", 105, 50, 10, 10, 10, 10),), dialogue="review"
    )
    assert _boxes(SceneRenderer(animation=animation).render_sample(blocked), "monster") == ()


def test_none_animation_preserves_frozen_v05_pixel_bytes_and_labels():
    simulator = Simulator(seed=731)
    for action in (SimAction("move", 1), SimAction("jump", 1), SimAction("idle")):
        simulator.step(action)
    snapshot = simulator.snapshot()
    rendered = SceneRenderer(style_seed=11, animation=None).render_sample(snapshot)
    # Independently measured on the unmodified v0.5 tree before running this review.
    assert (
        _sha(rendered.frame.pixels)
        == "ad103a81bb24b484992530bec7c250619f9686bf1bb0d392b95b064a01bae103"
    )
    assert rendered == SceneRenderer(style_seed=11).render_sample(snapshot)
    assert (rendered.frame.sequence, rendered.frame.timestamp_ms) == (3, 150)


@pytest.mark.parametrize("structured", [False, True])
def test_run_animation_is_self_contained_and_reinfer_replays_every_acquired_frame(
    tmp_path, structured
):
    animation = _palette_scene(tmp_path / "original-art")
    output = tmp_path / "run"
    summary = run_episode(
        output,
        RunConfig(seed=7, max_steps=7, record_stride=1),
        animation=animation,
        runtime=RuntimeConfig() if structured else None,
    )
    manifest = _read(output / "manifest.json")
    assert manifest["schema_version"] == (2 if structured else 1)
    assert summary["animation"] == animation.manifest()
    assert "scene-animation.json" in manifest["files"]
    assert all(source.source_file in manifest["files"] for source in animation.pack.sources)
    trace = [
        _read_line
        for _read_line in map(json.loads, (output / "trace.jsonl").read_text().splitlines())
    ]
    assert all("animation" not in row["observation"] for row in trace)
    before = {name: (output / name).read_bytes() for name in manifest["files"]}
    shutil.rmtree(tmp_path / "original-art")
    assert verify_run(output)["verified"] is True
    result = verify_perception(output)
    assert result["perception_reproduced"] is True
    assert result["frames_reinferred"] == summary["steps"] + 1
    if structured:
        assert result["belief_frames_recomputed"] == summary["steps"] + 1
    assert {name: (output / name).read_bytes() for name in manifest["files"]} == before


def test_unrecognizable_animation_pauses_instead_of_reading_simulator_truth(tmp_path):
    image = Image.new("RGBA", (14, 22), (14, 31, 77, 255))
    scene = _scene(tmp_path / "art", (image,), ((7, 22),), (100,))
    output = tmp_path / "run"
    result = run_episode(output, RunConfig(max_steps=40), animation=scene, runtime=RuntimeConfig())
    assert result["status"] == "stopped"
    assert result["independent_goal_check"] is False
    assert set(result["actions"]) == {"idle"}
    assert verify_perception(output)["perception_reproduced"] is True


@pytest.fixture
def mutable_run(tmp_path):
    scene = _palette_scene(tmp_path / "art")
    output = tmp_path / "run"
    run_episode(
        output,
        RunConfig(seed=7, max_steps=4, record_stride=1),
        animation=scene,
        runtime=RuntimeConfig(),
    )
    return output


@pytest.mark.parametrize(
    "change",
    [
        "summary",
        "sidecar",
        "pack-pixels",
        "timeline-all-copies",
        "omitted-source",
        "missing-summary",
    ],
)
def test_run_metadata_and_rendering_reject_contradictions_after_rehash(mutable_run, change):
    output = mutable_run
    summary = _read(output / "summary.json")
    scene = _read(output / "scene-animation.json")
    if change == "summary":
        summary["animation"]["content"]["bindings"] = {}
        _rehash_scene(summary["animation"])
        _write(output / "summary.json", summary)
    elif change == "missing-summary":
        summary.pop("animation")
        _write(output / "summary.json", summary)
    elif change == "omitted-source":
        manifest = _read(output / "manifest.json")
        filename = next(name for name in manifest["files"] if name.startswith("animation_objects/"))
        del manifest["files"][filename]
        _write(output / "manifest.json", manifest)
    else:
        if change == "sidecar":
            scene["content"]["bindings"]["player"] = "missing.clip"
        elif change == "pack-pixels":
            source = next(iter(scene["content"]["pack"]["sources"].values()))
            source["rgba_sha256"] = "0" * 64
        else:
            scene["content"]["pack"]["clips"][0]["frames"][0]["duration_ms"] = 60
        _rehash_scene(scene)
        _write(output / "scene-animation.json", scene)
        if change != "sidecar":
            summary["animation"] = copy.deepcopy(scene)
            _write(output / "summary.json", summary)
    _rehash_run(output)
    with pytest.raises(ValueError):
        verify_run(output)


@pytest.mark.parametrize("kind", ["manifest", "source"])
def test_run_animation_budgets_apply_before_unbounded_read_or_decoder(
    mutable_run, monkeypatch, kind
):
    original = Path.read_bytes

    def bounded_sources_only(path):
        if path.name == "scene-animation.json" or path.parent.name == "animation_objects":
            pytest.fail("animation evidence used unbounded read_bytes")
        return original(path)

    def no_decode(*args, **kwargs):
        pytest.fail("over-budget animation reached its decoder")

    monkeypatch.setattr(Path, "read_bytes", bounded_sources_only)
    monkeypatch.setattr(scene_module, "verify_animation_pack", no_decode)
    monkeypatch.setattr(
        lab_module, "MAX_SCENE_ANIMATION_BYTES" if kind == "manifest" else "MAX_SOURCE_BYTES", 16
    )
    with pytest.raises(ValueError, match="budget"):
        verify_run(mutable_run)


def test_scene_manifest_budget_applies_before_pack_reconstruction(tmp_path, monkeypatch):
    scene = _palette_scene(tmp_path / "art")
    output = tmp_path / "scene"
    export_scene_animation(scene, output)

    def no_decode(*args, **kwargs):
        pytest.fail("oversized scene metadata reached pack reconstruction")

    monkeypatch.setattr(scene_module, "MAX_SCENE_ANIMATION_BYTES", 16)
    monkeypatch.setattr(scene_module, "verify_animation_pack", no_decode)
    with pytest.raises(ValueError, match="budget"):
        load_scene_animation(output)


@pytest.fixture(scope="module")
def animated_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent-animation-dataset")
    scene = _palette_scene(root / "art")
    output = root / "dataset"
    generate_dataset(
        output, sessions=3, frames_per_session=12, seed=103, profile="lifecycle", animation=scene
    )
    shutil.rmtree(root / "art")
    return output


@pytest.fixture
def mutable_dataset(animated_dataset, tmp_path):
    return Path(shutil.copytree(animated_dataset, tmp_path / "dataset"))


def test_animation_dataset_schema4_replays_pixels_labels_and_sources(animated_dataset):
    manifest = _read(animated_dataset / "manifest.json")
    report = validate_dataset(animated_dataset)
    assert report["schema_version"] == manifest["schema_version"] == 4
    assert report["frames"] == 36
    assert manifest["animation_origin"] == "synthetic_geometry_with_versioned_animation"
    assert report["visual_agent_success_evaluated"] is False
    assert report["trajectory_verification"] == "actions_pixels_labels_and_phase_facts"
    assert report["animation"] == load_scene_animation(animated_dataset).manifest()


def test_schema4_external_maps_are_preserved_in_restore_evaluation_and_training(tmp_path):
    animation = _palette_scene(tmp_path / "art")
    maps = _maps()
    dataset = tmp_path / "dataset"
    generate_dataset(
        dataset,
        sessions=3,
        frames_per_session=18,
        seed=29,
        profile="lifecycle",
        map_pack=maps,
        animation=animation,
    )
    shutil.rmtree(tmp_path / "art")
    report = validate_dataset(dataset)
    assert report["schema_version"] == 4 and report["frames"] == 54
    assert MapPack.from_manifest(report["map_pack"]) == maps
    evaluation = benchmark_dataset(dataset, tmp_path / "benchmark.json", split="val")
    assert evaluation["dataset"]["schema_version"] == 4
    assert set(evaluation["grouped_metrics"]["by_map"]) == set(maps.map_ids)
    stage = tmp_path / "stage"
    stage.mkdir()
    _training_yaml(dataset, stage)
    snapshot = stage / "input_snapshot"
    assert (snapshot / "scene-animation.json").read_bytes() == (
        dataset / "scene-animation.json"
    ).read_bytes()
    assert (snapshot / "manifest.json").read_bytes() == (dataset / "manifest.json").read_bytes()
    shutil.rmtree(dataset)
    copied = validate_dataset(snapshot)
    assert copied["schema_version"] == 4
    assert copied["animation"] == animation.manifest()
    assert MapPack.from_manifest(copied["map_pack"]) == maps


@pytest.mark.parametrize(
    "change",
    ["origin", "version", "sidecar", "manifest-binding", "timeline-all-copies", "pixels", "labels"],
)
def test_schema4_mutations_cannot_hide_behind_repaired_hashes(mutable_dataset, change):
    dataset = mutable_dataset
    manifest = _read(dataset / "manifest.json")
    scene = _read(dataset / "scene-animation.json")
    if change == "origin":
        manifest["animation_origin"] = "real_game_validated"
    elif change == "version":
        manifest["schema_version"] = 2
    elif change == "manifest-binding":
        manifest["animation"]["content"]["bindings"] = {}
        _rehash_scene(manifest["animation"])
    elif change in {"sidecar", "timeline-all-copies"}:
        if change == "sidecar":
            scene["content"]["bindings"]["player"] = "missing.clip"
        else:
            scene["content"]["pack"]["clips"][0]["frames"][0]["duration_ms"] = 60
        _rehash_scene(scene)
        _write(dataset / "scene-animation.json", scene)
        if change == "timeline-all-copies":
            manifest["animation"] = scene
    elif change == "pixels":
        sample = manifest["samples"][0]
        path = dataset / sample["image"]
        with Image.open(path) as source:
            image = source.copy()
        image.putpixel((0, 0), (1, 2, 3))
        image.save(path)
        sample["image_sha256"] = _sha(path.read_bytes())
    else:
        sample = manifest["samples"][0]
        path = dataset / sample["label"]
        rows = path.read_text().splitlines()
        row = next(index for index, text in enumerate(rows) if text.startswith("0 "))
        columns = rows[row].split()
        columns[1] = str(float(columns[1]) + 0.001)
        rows[row] = " ".join(columns)
        path.write_text("\n".join(rows) + "\n")
        sample["label_sha256"] = _sha(path.read_bytes())
    _write(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError):
        validate_dataset(dataset)


def test_legacy_generation_rejects_animation_before_creating_output(tmp_path):
    animation = _palette_scene(tmp_path / "art")
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="lifecycle"):
        generate_dataset(output, sessions=3, frames_per_session=1, animation=animation)
    assert not output.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink permission is environment-dependent")
@pytest.mark.parametrize("name", ["scene-animation.json", "animation_objects"])
def test_legacy_dataset_rejects_even_dangling_animation_evidence(tmp_path, name):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, sessions=3, frames_per_session=1, seed=19)
    (dataset / name).symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="animation"):
        validate_dataset(dataset)


def test_cli_scene_run_and_schema4_dataset_roundtrip(tmp_path, capsys):
    animation = _palette_scene(tmp_path / "art")
    scene = tmp_path / "scene"
    export_scene_animation(animation, scene)
    run = tmp_path / "run"
    assert (
        main(
            [
                "lab",
                "run",
                "--output",
                str(run),
                "--max-steps",
                "4",
                "--runtime",
                "structured",
                "--animation-scene",
                str(scene),
            ]
        )
        == 1
    )
    run_report = json.loads(capsys.readouterr().out)
    assert run_report["animation"] == animation.manifest()
    assert run_report["status"] == "failed"
    assert run_report["reason"] == "step_budget_exhausted"
    assert main(["lab", "reinfer", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["structured_runtime_reproduced"] is True
    dataset = tmp_path / "dataset"
    assert (
        main(
            [
                "dataset",
                "create",
                str(dataset),
                "--sessions",
                "3",
                "--frames",
                "12",
                "--seed",
                "17",
                "--profile",
                "lifecycle",
                "--animation-scene",
                str(scene),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == 4
    assert main(["dataset", "check", str(dataset)]) == 0
    assert json.loads(capsys.readouterr().out)["validated"] is True
