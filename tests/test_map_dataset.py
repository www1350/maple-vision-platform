"""External map identity, action replay, and dataset portability contracts."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

import maple_vision.dataset as factory
from maple_vision.asset_registry import import_asset
from maple_vision.dataset import LIFECYCLE_PHASES, generate_dataset, validate_dataset
from maple_vision.evaluation import benchmark_dataset
from maple_vision.map_pack import MapPack, load_map_pack
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator
from maple_vision.sprite_pack import load_sprite_pack
from maple_vision.training import _training_yaml

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/maps/three-map-fixture.json"


def _read(path):
    return json.loads((path / "manifest.json").read_text("utf-8"))


def _write(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest), "utf-8")


@pytest.fixture(scope="module")
def external_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("external-map-dataset")
    source = root / "authored.json"
    shutil.copyfile(EXAMPLE, source)
    pack = load_map_pack(source)
    source.unlink()
    dataset = root / "dataset"
    generate_dataset(
        dataset,
        sessions=3,
        frames_per_session=27,
        seed=123,
        profile="lifecycle",
        map_pack=pack,
    )
    return dataset


@pytest.fixture
def mutable_external(external_dataset, tmp_path):
    return Path(shutil.copytree(external_dataset, tmp_path / "copy"))


def test_three_maps_all_jobs_replay_six_phases_without_source_file(external_dataset):
    manifest = _read(external_dataset)
    pack = MapPack.from_manifest(manifest["map_pack"])
    assert manifest["schema_version"] == 3
    assert pack.map_ids == ("grove", "quarry", "ruins")
    assert not (external_dataset.parent / "authored.json").exists()
    for session in manifest["sessions"]:
        sim = Simulator(session["seed"], session["job"], pack.entry_map, map_pack=pack)
        transitions = []
        for action in session["trajectory"]["actions"]:
            previous = sim.snapshot().map_id
            snapshot = sim.step(SimAction(**action)).snapshot
            if "map_changed" in snapshot.events:
                transitions.append((previous, snapshot.map_id))
        assert transitions == [("grove", "quarry"), ("quarry", "ruins"), ("ruins", "grove")]
        assert sim.snapshot().player.alive
        assert all(
            not world["drops"] and not any(monster["hp"] for monster in world["monsters"])
            for world in sim.save()["maps"].values()
        )
        records = [x for x in manifest["samples"] if x["session_id"] == session["id"]]
        assert {(x["map_id"], x["phase"]) for x in records} == {
            (map_id, phase) for map_id in pack.map_ids for phase in LIFECYCLE_PHASES
        }
    result = validate_dataset(external_dataset)
    assert result["frames"] == 81
    assert result["map_pack"] == pack.manifest()
    assert result["visual_agent_success_evaluated"] is False
    assert result["missing_maps"] == {"train": [], "val": [], "test": []}


def test_external_dataset_is_byte_reproducible(external_dataset, tmp_path):
    other = tmp_path / "other"
    generate_dataset(
        other,
        sessions=3,
        frames_per_session=27,
        seed=123,
        profile="lifecycle",
        map_pack=load_map_pack(EXAMPLE),
    )
    assert {
        p.relative_to(external_dataset): p.read_bytes()
        for p in external_dataset.rglob("*")
        if p.is_file()
    } == {p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "change",
    ["missing_pack", "stale_hash", "rehashed_theme", "rotated_route", "unknown_map", "downgrade"],
)
def test_identity_and_semantic_tampering_rejected(mutable_external, change):
    manifest = _read(mutable_external)
    if change == "missing_pack":
        del manifest["map_pack"]
    elif change == "stale_hash":
        manifest["map_pack"]["content"]["maps"][0]["theme"] = "cavern"
    elif change == "rehashed_theme":
        content = manifest["map_pack"]["content"]
        content["maps"][0]["theme"] = "cavern"
        manifest["map_pack"] = MapPack.from_dict(content).manifest()
    elif change == "rotated_route":
        content = manifest["map_pack"]["content"]
        content["route"] = ["quarry", "ruins", "grove"]
        manifest["map_pack"] = MapPack.from_dict(content).manifest()
    elif change == "unknown_map":
        manifest["samples"][0]["map_id"] = "unlisted"
    elif change == "downgrade":
        manifest["schema_version"] = 2
    _write(mutable_external, manifest)
    with pytest.raises(ValueError):
        validate_dataset(mutable_external)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"profile": "legacy", "frames_per_session": 18}, "lifecycle"),
        ({"profile": "lifecycle", "frames_per_session": 17}, "at least 18"),
    ],
)
def test_map_phase_budget_checked_before_output_creation(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_dataset(
            tmp_path / "invalid", sessions=3, map_pack=load_map_pack(EXAMPLE), **kwargs
        )
    assert list(tmp_path.iterdir()) == []


def test_map_type_checked_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match="verified MapPack"):
        generate_dataset(tmp_path / "invalid", map_pack={})
    assert list(tmp_path.iterdir()) == []


def test_incomplete_external_trajectory_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "LIFECYCLE_MAX_STEPS", 2)
    with pytest.raises(ValueError, match="step budget"):
        generate_dataset(
            tmp_path / "incomplete",
            sessions=3,
            frames_per_session=18,
            profile="lifecycle",
            map_pack=load_map_pack(EXAMPLE),
        )
    assert list(tmp_path.iterdir()) == []


def test_evaluation_groups_follow_embedded_map_ids(external_dataset, tmp_path):
    # Fixture baseline only: no learned candidate is scored or promoted here.
    report = benchmark_dataset(external_dataset, tmp_path / "pixel-val.json", split="val")
    assert report["dataset"]["schema_version"] == 3
    groups = report["grouped_metrics"]
    assert set(groups["by_map"]) == {"grove", "quarry", "ruins"}
    assert set(groups["by_map_phase"]) == {"grove", "quarry", "ruins"}
    assert sum(value["images"] for value in groups["by_map"].values()) == 27
    for metric in ("tp", "fp", "fn"):
        assert (
            sum(value["micro"][metric] for value in groups["by_map"].values())
            == (report["metrics"]["micro"][metric])
        )


def test_training_private_snapshot_keeps_embedded_pack(mutable_external, tmp_path):
    stage = tmp_path / "training-stage"
    stage.mkdir()
    before = (mutable_external / "manifest.json").read_bytes()
    _, names = _training_yaml(mutable_external.resolve(), stage)
    shutil.rmtree(mutable_external)
    snapshot = stage / "input_snapshot"
    assert (snapshot / "manifest.json").read_bytes() == before
    assert len(names) == 7
    result = validate_dataset(snapshot)
    assert result["schema_version"] == 3 and result["frames"] == 81


def test_external_map_and_sprite_sources_are_independently_portable(tmp_path):
    original = tmp_path / "authored.png"
    Image.new("RGBA", (8, 8), (40, 80, 210, 255)).save(original)
    registry = tmp_path / "registry"
    asset = import_asset(
        registry,
        original,
        entity_id="player",
        game_version="fixture-v1",
        source_uri="urn:test:authored-player",
        license="CC0-1.0",
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
        ),
        "utf-8",
    )
    dataset = tmp_path / "combined"
    generate_dataset(
        dataset,
        sessions=3,
        frames_per_session=18,
        profile="lifecycle",
        map_pack=load_map_pack(EXAMPLE),
        sprite_pack=load_sprite_pack(registry, mapping),
    )
    original.unlink()
    mapping.unlink()
    shutil.rmtree(registry)
    result = validate_dataset(dataset)
    assert result["schema_version"] == 3
    assert result["map_pack"]["content"]["id"] == "three-map-fixture"
    assert "sprite_pack" in result
