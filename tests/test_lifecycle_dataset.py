import hashlib
import json
import shutil
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest
from PIL import Image

import maple_vision.dataset as factory
from maple_vision.asset_registry import import_asset
from maple_vision.contracts import Detection
from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.evaluation import benchmark_dataset
from maple_vision.scene import ImageFrame, SceneRenderer
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator
from maple_vision.sprite_pack import load_sprite_pack


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory):
    path = tmp_path_factory.mktemp("lifecycle-source") / "dataset"
    generate_dataset(path, sessions=3, frames_per_session=24, seed=123, profile="lifecycle")
    return path


@pytest.fixture
def mutable_dataset(lifecycle, tmp_path):
    return Path(shutil.copytree(lifecycle, tmp_path / "copy"))


def _read(path):
    return json.loads((path / "manifest.json").read_text("utf-8"))


def _write(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest), "utf-8")


def test_actual_two_map_trajectories_cover_spawn_combat_loot_clear_dialogue_and_transition(
    lifecycle,
) -> None:
    manifest = _read(lifecycle)
    for session in manifest["sessions"]:
        simulator = Simulator(session["seed"], session["job"], session["map_id"])
        selected = {
            sample["sequence"]: sample
            for sample in manifest["samples"]
            if sample["session_id"] == session["id"]
        }
        previous_map = None
        seen = set()
        transitions = []
        spawn_positions = []
        high_platforms = set()
        for sequence in range(len(session["trajectory"]["actions"]) + 1):
            snapshot = simulator.snapshot()
            if "map_changed" in snapshot.events:
                assert previous_map != snapshot.map_id
                transitions.append((previous_map, snapshot.map_id))
            if sequence in selected:
                sample = selected[sequence]
                assert sample["map_id"] == snapshot.map_id
                assert sample["timestamp_ms"] == sequence * 50
                phase = sample["phase"]
                seen.add((snapshot.map_id, phase))
                if phase == "spawn":
                    assert snapshot.player.x < 40
                    assert len(snapshot.monsters) == 3 and not snapshot.drops
                    spawn_positions.append(snapshot.player.x)
                elif phase == "combat":
                    assert snapshot.monsters
                elif phase == "loot":
                    assert not snapshot.monsters and snapshot.drops
                elif phase == "cleared":
                    assert not snapshot.monsters and not snapshot.drops
                    assert snapshot.dialogue is None
                elif phase == "dialogue":
                    assert snapshot.dialogue is not None
                elif phase == "transition":
                    assert "map_changed" in snapshot.events
                else:
                    pytest.fail(f"unrecognized lifecycle phase: {phase}")
                if "high_platform" in sample["coverage"]["tags"]:
                    assert snapshot.player.grounded
                    assert snapshot.player.y + snapshot.player.h < 168
                    high_platforms.add(snapshot.map_id)
            previous_map = snapshot.map_id
            if sequence < len(session["trajectory"]["actions"]):
                simulator.step(SimAction(**session["trajectory"]["actions"][sequence]))
        assert seen == {
            (map_id, phase)
            for map_id in ("meadow", "cavern")
            for phase in ("spawn", "combat", "loot", "cleared", "dialogue", "transition")
        }
        assert transitions == [("meadow", "cavern"), ("cavern", "meadow")]
        assert 24.0 in spawn_positions and 35.0 in spawn_positions
        assert high_platforms == {"meadow", "cavern"}


def test_coverage_summary_is_recounted_from_actual_samples_and_labels(lifecycle) -> None:
    manifest = _read(lifecycle)
    summary = validate_dataset(lifecycle)
    assert summary["schema_version"] == 2
    assert summary["visual_agent_success_evaluated"] is False
    assert summary["coverage"] == manifest["coverage"]
    assert summary["missing_classes"] == {"train": [], "val": [], "test": []}
    assert summary["missing_maps"] == {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        records = [sample for sample in manifest["samples"] if sample["split"] == split]
        reported = summary["coverage"]["by_split"][split]
        assert reported["frames"] == 24
        assert reported["phase_counts"] == Counter(sample["phase"] for sample in records)
        assert reported["player_position_bins"]["left_edge"] >= 2
        assert reported["player_position_bins"]["right_edge"] >= 2
        assert reported["tag_counts"]["high_platform"] >= 2
        class_counts = Counter()
        for sample in records:
            x = sample["coverage"]["player_x"]
            if x < 40:
                assert sample["coverage"]["position_bin"] == "left_edge"
            if x >= 280:
                assert sample["coverage"]["position_bin"] == "right_edge"
            for line in (lifecycle / sample["label"]).read_text().splitlines():
                class_counts[manifest["classes"][int(line.split()[0])]] += 1
        assert reported["class_counts"] == class_counts
        assert sum(group["frames"] for group in reported["maps"].values()) == 24
    sessions_by_split = {
        split: {sample["session_id"] for sample in manifest["samples"] if sample["split"] == split}
        for split in ("train", "val", "test")
    }
    assert all(
        not sessions_by_split[a] & sessions_by_split[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    )


def test_lifecycle_is_byte_reproducible(lifecycle, tmp_path) -> None:
    second = tmp_path / "again"
    generate_dataset(second, sessions=3, frames_per_session=24, seed=123, profile="lifecycle")
    assert (second / "manifest.json").read_bytes() == (lifecycle / "manifest.json").read_bytes()
    assert (second / "data.yaml").read_bytes() == (lifecycle / "data.yaml").read_bytes()


def test_legacy_release_dataset_retains_its_exact_manifest_hash(tmp_path) -> None:
    result = generate_dataset(
        tmp_path / "legacy", sessions=18, frames_per_session=16, seed=20261124
    )
    assert result["manifest_sha256"] == (
        "ecc0ff6c4e9bbebc9915f0313bb3b227a99ecc321051ee2ae2e8a0eff096ad67"
    )
    assert result["schema_version"] == 1
    assert "coverage" not in result and "profile" not in result


@pytest.mark.parametrize("profile,frames", [("unknown", 24), (True, 24), ("lifecycle", 11)])
def test_profile_and_required_sample_budget_are_validated(tmp_path, profile, frames) -> None:
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "bad", sessions=3, frames_per_session=frames, profile=profile)
    assert not (tmp_path / "bad").exists()


def test_generation_fails_instead_of_claiming_missing_phases(tmp_path, monkeypatch) -> None:
    def incomplete(seed, job, minimum_frames):
        return [], [Simulator(seed=seed, job=job).snapshot()]

    monkeypatch.setattr(factory, "_lifecycle_trajectory", incomplete)
    with pytest.raises(ValueError, match="lacks required phase"):
        generate_dataset(tmp_path / "bad", sessions=3, frames_per_session=12, profile="lifecycle")
    assert list(tmp_path.iterdir()) == []


def test_generation_enforces_actual_action_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(factory, "LIFECYCLE_MAX_STEPS", 1)
    with pytest.raises(ValueError, match="step budget"):
        generate_dataset(tmp_path / "bad", sessions=3, frames_per_session=12, profile="lifecycle")
    assert list(tmp_path.iterdir()) == []


def test_forged_phase_and_corresponding_summary_are_rejected(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    record = manifest["samples"][0]
    assert record["phase"] == "spawn"
    record["phase"] = "combat"
    split = manifest["coverage"]["by_split"][record["split"]]
    for group in (split, split["maps"][record["map_id"]]):
        group["phase_counts"]["spawn"] -= 1
        group["phase_counts"]["combat"] += 1
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="metadata disagrees"):
        validate_dataset(mutable_dataset)


@pytest.mark.parametrize(
    "field,value",
    [
        ("player_x", 200.0),
        ("position_bin", "right_edge"),
        ("grounded", 1),
        ("tags", ["high_platform"]),
        ("monsters_alive", 0),
        ("visible_class_counts", {}),
    ],
)
def test_forged_or_mistyped_per_frame_coverage_is_rejected(mutable_dataset, field, value) -> None:
    manifest = _read(mutable_dataset)
    manifest["samples"][0]["coverage"][field] = value
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="metadata disagrees"):
        validate_dataset(mutable_dataset)


def test_forged_aggregate_is_rejected(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    manifest["coverage"]["by_split"]["train"]["tag_counts"]["high_platform"] += 1
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="coverage summary"):
        validate_dataset(mutable_dataset)


def test_rehashed_label_forgery_fails_against_rendered_instances(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    record = manifest["samples"][0]
    path = mutable_dataset / record["label"]
    lines = path.read_text().splitlines()
    row = lines[0].split()
    row[0] = str((int(row[0]) + 1) % 7)
    lines[0] = " ".join(row)
    path.write_text("\n".join(lines) + "\n")
    record["label_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="labels disagree"):
        validate_dataset(mutable_dataset)


def test_rehashed_image_forgery_fails_against_action_replay(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    record = manifest["samples"][0]
    path = mutable_dataset / record["image"]
    with Image.open(path) as source:
        altered = source.copy()
    altered.putpixel((0, 0), (255, 0, 0))
    altered.save(path)
    record["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="image disagrees"):
        validate_dataset(mutable_dataset)


def test_action_changes_cannot_reuse_old_coverage(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    action = manifest["sessions"][0]["trajectory"]["actions"][0]
    action.update(kind="move", direction=1)
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="metadata disagrees|image disagrees"):
        validate_dataset(mutable_dataset)


@pytest.mark.parametrize(
    "mutation", ["step_ms", "actions_limit", "extra_metadata", "extra_root", "dimensions"]
)
def test_malformed_lifecycle_contract_is_rejected(mutable_dataset, mutation) -> None:
    manifest = _read(mutable_dataset)
    if mutation == "step_ms":
        manifest["sessions"][0]["trajectory"]["step_ms"] = True
    elif mutation == "actions_limit":
        manifest["sessions"][0]["trajectory"]["actions"] = [
            {"kind": "idle", "direction": 0, "skill": "basic", "target": ""}
        ] * 4097
    elif mutation == "extra_metadata":
        manifest["samples"][0]["secret"] = "not in the contract"
    elif mutation == "dimensions":
        manifest["samples"][0]["width"] = 8192
    else:
        manifest["imaginary_accuracy"] = 1.0
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="lifecycle"):
        validate_dataset(mutable_dataset)


def test_removing_required_phase_is_not_valid_coverage(mutable_dataset) -> None:
    manifest = _read(mutable_dataset)
    removed = [
        sample
        for sample in manifest["samples"]
        if sample["map_id"] == "meadow" and sample["phase"] == "transition"
    ]
    assert len(removed) == 3
    for sample in removed:
        (mutable_dataset / sample["image"]).unlink()
        (mutable_dataset / sample["label"]).unlink()
    manifest["samples"] = [sample for sample in manifest["samples"] if sample not in removed]
    manifest["frames_per_session"] -= 1
    _write(mutable_dataset, manifest)
    with pytest.raises(ValueError, match="missing required map/phase"):
        validate_dataset(mutable_dataset)


def test_truthful_combat_frames_without_any_selected_hit_do_not_meet_contract(tmp_path) -> None:
    dataset = tmp_path / "without-hits"
    generate_dataset(dataset, sessions=3, frames_per_session=12, seed=123, profile="lifecycle")
    manifest = _read(dataset)
    for session in manifest["sessions"]:
        simulator = Simulator(session["seed"], session["job"])
        renderer = SceneRenderer(style_seed=session["style_seed"])
        first_entries = {}
        replacements = {}
        for sequence in range(len(session["trajectory"]["actions"]) + 1):
            snapshot = simulator.snapshot()
            facts = factory._lifecycle_facts(snapshot, first_entries)
            if (
                facts["phase"] == "combat"
                and "attack_hit" not in snapshot.events
                and snapshot.map_id not in replacements
            ):
                replacements[snapshot.map_id] = (snapshot, facts)
            if sequence < len(session["trajectory"]["actions"]):
                simulator.step(SimAction(**session["trajectory"]["actions"][sequence]))
        assert set(replacements) == {"meadow", "cavern"}
        for record in manifest["samples"]:
            if record["session_id"] != session["id"] or record["phase"] != "combat":
                continue
            snapshot, facts = replacements[record["map_id"]]
            rendered = renderer.render_sample(snapshot)
            assert "attack_hit" not in snapshot.events
            facts["coverage"]["visible_class_counts"] = {
                name: sum(detection.class_id == name for detection in rendered.annotations)
                for name in manifest["classes"]
            }
            image, label = dataset / record["image"], dataset / record["label"]
            image.write_bytes(rendered.frame.png_bytes())
            label.write_text(factory._label_text(rendered))
            record.update(
                **facts,
                sequence=snapshot.sequence,
                timestamp_ms=snapshot.timestamp_ms,
                objects=len(rendered.annotations),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            )
    manifest["samples"].sort(key=lambda record: (record["session_id"], record["sequence"]))
    manifest["coverage"] = factory._coverage_summary(manifest["samples"])
    _write(dataset, manifest)
    with pytest.raises(ValueError, match="selected combat attack_hit"):
        validate_dataset(dataset)


def test_authorized_sprite_lifecycle_is_replayed_without_original_registry(tmp_path) -> None:
    original = tmp_path / "player.png"
    sprite = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for x in range(2, 6):
        for y in range(1, 7):
            sprite.putpixel((x, y), (40, 80, 210, 255))
    sprite.save(original)
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
        )
    )
    pack = load_sprite_pack(registry, mapping)
    dataset = tmp_path / "sprites"
    generate_dataset(
        dataset, sessions=3, frames_per_session=12, profile="lifecycle", sprite_pack=pack
    )
    original.unlink()
    shutil.rmtree(registry)
    assert validate_dataset(dataset)["frames"] == 36


class _AlwaysFalseMonster:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        assert isinstance(frame, ImageFrame)
        assert {field.name for field in fields(frame)} == {
            "sequence",
            "timestamp_ms",
            "width",
            "height",
            "pixels",
        }
        self.calls += 1
        return (Detection("monster", (0, 30, 3, 33), 0.9),)


def test_grouped_evaluation_preserves_aggregates_and_counts_empty_scene_false_positives(
    lifecycle, tmp_path
) -> None:
    detector = _AlwaysFalseMonster()
    report = benchmark_dataset(lifecycle, tmp_path / "benchmark.json", detector)
    assert report["schema_version"] == 2
    assert detector.calls == report["metrics"]["images"] + 3
    groups = report["grouped_metrics"]
    for dimension in ("by_phase", "by_map", "by_player_position"):
        assert sum(value["images"] for value in groups[dimension].values()) == 24
        for metric in ("tp", "fp", "fn"):
            assert (
                sum(value["micro"][metric] for value in groups[dimension].values())
                == (report["metrics"]["micro"][metric])
            )
    cleared = groups["by_phase"]["cleared"]
    assert cleared["per_class"]["monster"]["ground_truth"] == 0
    assert cleared["per_class"]["monster"]["fp"] == cleared["images"]
    assert cleared["per_class"]["monster"]["ap50"] is None
    assert cleared["map50_95"] is None
    assert cleared["sessions"] == 1
    assert groups["by_coverage_tag"]["high_platform"]["images"] >= 2
    assert groups["by_player_position"]["left_edge"]["images"] >= 2


def test_grouping_is_not_fabricated_for_legacy_datasets(tmp_path) -> None:
    dataset = tmp_path / "legacy"
    generate_dataset(dataset, sessions=3, frames_per_session=2)
    result = benchmark_dataset(dataset, tmp_path / "report.json", _AlwaysFalseMonster())
    assert result["schema_version"] == 1
    assert "grouped_metrics" not in result
