"""Independent multi-map mission goals, portable evidence and semantic tamper checks."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from maple_vision.agent import AgentDecision, HybridPerception, VisualAgent
from maple_vision.catalog import import_catalog, initialize_catalog
from maple_vision.contracts import Detection, Status
from maple_vision.lab import RunConfig, _execute_episode, run_episode, verify_perception, verify_run
from maple_vision.map_pack import MapPack, load_map_pack
from maple_vision.profiles import load_class_profile
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator
from maple_vision.vision import PixelPerception

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/maps/three-map-fixture.json"


def rewrite(bundle, relative, value):
    path = bundle / relative
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module", params=("assassin", "dragon_knight", "cleric"))
def completed_map_run(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"map-{request.param}")
    source = root / "original.json"
    shutil.copyfile(EXAMPLE, source)
    pack = load_map_pack(source)
    bundle = root / "run"
    summary = run_episode(
        bundle,
        RunConfig(seed=5, job=request.param, max_steps=1500, record_stride=64),
        map_pack=pack,
    )
    source.unlink()
    return bundle, summary, pack


def test_all_three_jobs_finish_authored_route_with_independent_counts(completed_map_run):
    bundle, summary, pack = completed_map_run
    assert summary["status"] == "completed", summary
    assert summary["independent_goal_check"]
    assert summary["events"]["monster_defeated"] == summary["drops_collected"] == 9
    assert summary["events"]["dialogue"] == 3
    assert summary["events"]["map_changed"] == 2
    assert summary["maps_visited"] == summary["dialogue_maps"] == ["grove", "quarry", "ruins"]
    assert summary["map_pack"] == pack.manifest()
    assert summary["perception_evidence"]["settings"]["map_pack_sha256"] == pack.content_sha256
    assert json.loads((bundle / "initial-state.json").read_text())["schema_version"] == 3
    assert not (bundle.parent / "original.json").exists()
    assert summary["alive"]


def test_deleted_map_file_does_not_prevent_full_replay_and_reinference(completed_map_run):
    bundle, summary, _ = completed_map_run
    before = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    action = verify_run(bundle)
    assert action["verified"] and action["independent_goal_check"]
    assert action["steps_replayed"] == summary["steps"]
    assert action["perception_reproduced"] is False
    actual = verify_perception(bundle)
    assert actual["verified"] and actual["policy_reproduced"]
    assert actual["frames_reinferred"] == summary["steps"] + 1
    assert actual["terminal_decision_compared"]
    assert actual["model_hash_verified"] is False
    assert before == {
        p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()
    }


@pytest.fixture
def short_map_run(tmp_path):
    bundle = tmp_path / "short"
    run_episode(bundle, RunConfig(max_steps=3, record_stride=1), map_pack=load_map_pack(EXAMPLE))
    return bundle


@pytest.mark.parametrize("location", ("summary.json", "initial-state.json", "map-pack.json"))
def test_conflicting_pack_copies_rejected_even_with_new_outer_hash(short_map_run, location):
    value = json.loads((short_map_run / location).read_text())
    metadata = value if location == "map-pack.json" else value["map_pack"]
    metadata["content"]["version"] = "2.0.0"
    altered = MapPack.from_dict(metadata["content"]).manifest()
    if location == "map-pack.json":
        value = altered
    else:
        value["map_pack"] = altered
    rewrite(short_map_run, location, value)
    with pytest.raises(ValueError, match="map pack"):
        verify_run(short_map_run)


def test_embedded_map_file_must_be_in_outer_manifest(short_map_run):
    path = short_map_run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("map-pack.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="map pack"):
        verify_run(short_map_run)


def test_oversized_map_file_is_rejected_before_unbounded_checksum_read(short_map_run, monkeypatch):
    target = short_map_run / "map-pack.json"
    target.write_bytes(b" " * (1024 * 1024 + 1))
    original = Path.read_bytes

    def guarded(path):
        if path == target:
            raise AssertionError("map evidence must use a bounded read before hashing")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="map pack"):
        verify_run(short_map_run)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("width", 320.0)])
def test_summary_map_metadata_requires_exact_types_even_with_new_outer_hash(
    short_map_run, field, value
):
    summary = json.loads((short_map_run / "summary.json").read_text())
    if field == "schema_version":
        summary["map_pack"][field] = value
    else:
        summary["map_pack"]["content"][field] = value
    rewrite(short_map_run, "summary.json", summary)
    with pytest.raises(ValueError):
        verify_run(short_map_run)


def test_changed_map_settings_cannot_claim_matching_reinference(short_map_run):
    summary = json.loads((short_map_run / "summary.json").read_text())
    summary["perception_evidence"]["settings"]["map_pack_sha256"] = "0" * 64
    rewrite(short_map_run, "summary.json", summary)
    assert verify_run(short_map_run)["verified"]  # action replay makes no perception claim
    with pytest.raises(ValueError, match="match run"):
        verify_perception(short_map_run)


def test_even_consistently_rewritten_run_must_start_from_declared_seeded_world(tmp_path):
    pack = load_map_pack(EXAMPLE)
    initial = Simulator(map_id=pack.entry_map, map_pack=pack).save()
    initial["player"]["x"] += 1
    bundle = tmp_path / "forged-start"
    _execute_episode(bundle, RunConfig(max_steps=1), initial_state=initial, map_pack=pack)
    with pytest.raises(ValueError, match="fresh seeded world"):
        verify_run(bundle)


def test_unmanifested_map_metadata_cannot_be_added_to_legacy_run(tmp_path):
    bundle = tmp_path / "legacy"
    run_episode(bundle, RunConfig(max_steps=1))
    summary = json.loads((bundle / "summary.json").read_text())
    summary["map_pack"] = load_map_pack(EXAMPLE).manifest()
    rewrite(bundle, "summary.json", summary)
    with pytest.raises(ValueError, match="unmanifested map"):
        verify_run(bundle)


def test_policy_false_completion_does_not_satisfy_external_oracle(monkeypatch):
    monkeypatch.setattr(
        VisualAgent,
        "decide",
        lambda self, observation: AgentDecision(SimAction(), "done", "completed", "forged"),
    )
    result = run_episode(map_pack=load_map_pack(EXAMPLE))
    assert result["status"] == "failed"
    assert result["reason"] == "independent_goal_check_failed"
    assert not result["independent_goal_check"]


def test_oracle_counts_authored_monsters_instead_of_assuming_builtin_six():
    content = load_map_pack(EXAMPLE).to_dict()
    for definition in content["maps"]:
        definition["monster_x"] = [150]
    pack = MapPack.from_dict(content)
    result = run_episode(map_pack=pack)
    assert result["status"] == "completed", result
    assert result["independent_goal_check"]
    assert result["events"]["monster_defeated"] == result["drops_collected"] == 3


def test_map_and_verified_class_profile_restore_without_original_sources(tmp_path):
    database = tmp_path / "catalog.sqlite"
    initialize_catalog(database)
    import_catalog(database, EXAMPLE.parents[2] / "fixtures/knowledge-demo.json")
    profile = load_class_profile(database, "assassin", "fixture-v1")
    bundle = tmp_path / "run"
    run_episode(
        bundle,
        RunConfig(max_steps=3),
        map_pack=load_map_pack(EXAMPLE),
        class_profile=profile,
    )
    database.unlink()
    assert verify_run(bundle)["verified"]
    assert verify_perception(bundle)["perception_reproduced"]
    initial = json.loads((bundle / "initial-state.json").read_text())
    assert initial["schema_version"] == 3 and "class_profile" in initial
    summary = json.loads((bundle / "summary.json").read_text())
    summary["class_profile"] = None
    rewrite(bundle, "summary.json", summary)
    with pytest.raises(ValueError, match="provenance"):
        verify_run(bundle)


@pytest.mark.parametrize(
    "fault,status",
    [
        ("brief_dropout", "completed"),
        ("duplicate_frames", "completed"),
        ("persistent_dropout", "stopped"),
        ("freeze_input", "failed"),
    ],
)
def test_external_map_faults_reproduce_bounded_failure_or_recovery(tmp_path, fault, status):
    bundle = tmp_path / "fault"
    summary = run_episode(
        bundle, RunConfig(fault=fault, record_stride=64), map_pack=load_map_pack(EXAMPLE)
    )
    assert summary["status"] == status
    result = verify_perception(bundle)
    assert result["recorded_status"] == status
    assert result["frames_reinferred"] == summary["steps"] + 1


def test_model_terrain_route_with_external_markers_never_reads_palette_objects(monkeypatch):
    pack = load_map_pack(EXAMPLE)
    frame = SceneRenderer(map_pack=pack).render(
        Simulator(map_id=pack.entry_map, map_pack=pack).snapshot()
    )
    objects = (Detection("player", (24, 146, 38, 168), 0.9),)

    class Detector:
        def detect(self, frame):
            return objects

    def forbidden(*args):
        raise AssertionError("model mode must not fall back to fixture objects")

    monkeypatch.setattr(PixelPerception, "detect", forbidden)
    observation = HybridPerception(
        Detector(), terrain_source="model", map_markers=pack.markers
    ).observe(frame)
    assert observation.detections == objects
    assert observation.map_id == "grove" and observation.status is Status.KNOWN


@pytest.mark.parametrize("maps", [[], (), ("grove", "grove"), (None,), ("../map",), ("a",) * 9])
def test_visual_policy_accepts_only_immutable_bounded_distinct_route_ids(maps):
    with pytest.raises(ValueError, match="mission maps"):
        VisualAgent(maps=maps)


def test_visual_policy_copies_only_map_ids_not_pack_geometry():
    pack = load_map_pack(EXAMPLE)
    policy = VisualAgent(maps=pack.map_ids)
    assert policy.maps == ("grove", "quarry", "ruins")
    assert not any(isinstance(value, (Simulator, MapPack)) for value in vars(policy).values())
