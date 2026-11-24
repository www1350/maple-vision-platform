"""Knowledge-backed static class rules affect simulation and survive offline replay."""

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from maple_vision.catalog import canonical_json, import_catalog, initialize_catalog
from maple_vision.profiles import (
    class_spec_sha256,
    load_class_profile,
    parse_class_spec,
    parse_loaded_class_profile,
)
from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.simulation import Simulator


def record(spec=None, **overrides):
    value = {
        "kind": "classes",
        "entity_id": "assassin",
        "game_version": "profile-v1",
        "source_uri": "urn:self:class-profile",
        "collected_at": "2026-11-24T00:00:00Z",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "scope": "synthetic_fixture",
        "verification_status": "VERIFIED",
        "payload": asdict(spec or CLASS_SPECS["assassin"]),
    }
    value.update(overrides)
    return value


def catalog(tmp_path, *records):
    db = tmp_path / "knowledge.sqlite"
    initialize_catalog(db)
    source = tmp_path / "profiles.json"
    source.write_text(json.dumps({"schema_version": 1, "records": records}), encoding="utf-8")
    import_catalog(db, source)
    return db


def custom_spec():
    return replace(
        CLASS_SPECS["assassin"],
        max_hp=120,
        max_mp=80,
        attack_damage=40,
        attack_range=180,
        attack_mp_cost=3,
        attack_cooldown_ms=50,
        ammo_cost=2,
        walk_speed=100,
        jump_speed=320,
    )


def test_complete_demo_profiles_load_for_all_three_jobs(tmp_path):
    db = tmp_path / "knowledge.sqlite"
    initialize_catalog(db)
    import_catalog(db, Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json")
    for job in CLASS_SPECS:
        loaded = load_class_profile(db, job, "fixture-v1")
        assert loaded.class_spec == CLASS_SPECS[job]
        assert loaded.profile_sha256 == loaded.records[0].payload_sha256
        assert (
            loaded.provenance_sha256
            == hashlib.sha256(
                canonical_json([asdict(row) for row in loaded.records]).encode()
            ).hexdigest()
        )
        assert loaded.records[0].verification_status == "VERIFIED"
        assert loaded.records[0].scope == "synthetic_fixture"


def test_catalog_class_rules_change_real_combat_and_resources_without_global_mutation(tmp_path):
    db = catalog(tmp_path, record(custom_spec()))
    loaded = load_class_profile(db, "assassin", "profile-v1")
    baseline = Simulator()
    injected = Simulator(class_spec=loaded.class_spec)
    initial = injected.snapshot().player
    assert (initial.max_hp, initial.max_mp, initial.hp, initial.mp) == (120, 80, 120, 80)
    # The original first target is out of default attack range but within the profile's range.
    assert "attack_hit" not in baseline.step(SimAction("attack")).events
    result = injected.step(SimAction("attack"))
    assert "monster_defeated" in result.events
    assert len(result.snapshot.monsters) == 2
    assert (result.snapshot.player.mp, result.snapshot.player.ammo) == (77, 198)
    assert injected.step(SimAction("attack")).accepted  # profile cooldown is one 50ms tick
    assert CLASS_SPECS["assassin"].attack_damage == 18
    assert CLASS_SPECS["assassin"].attack_range == 92
    assert Simulator().snapshot().player.max_hp == 100


def test_injected_speed_and_jump_are_used_by_physics():
    sim = Simulator(class_spec=custom_spec())
    assert sim.step(SimAction("move", 1)).snapshot.player.x == 29
    jump = sim.step(SimAction("jump", 1)).snapshot.player
    baseline = Simulator()
    baseline.step(SimAction("move", 1))
    default_jump = baseline.step(SimAction("jump", 1)).snapshot.player
    assert jump.vx == 100
    assert jump.vy < default_jump.vy
    assert jump.y < default_jump.y
    no_jump_attack = Simulator(class_spec=replace(custom_spec(), can_jump_attack=False))
    blocked = no_jump_attack.step(SimAction("jump_attack"))
    assert not blocked.accepted and blocked.reason == "jump_attack_unavailable"
    assert blocked.snapshot.player.grounded


def test_injected_healing_and_resource_caps_survive_restore():
    spec = replace(CLASS_SPECS["cleric"], max_hp=110, max_mp=150, heal_amount=50)
    state = Simulator(job="cleric", class_spec=spec).save()
    state["player"]["hp"] = 40
    sim = Simulator.restore(state)
    result = sim.step(SimAction("heal"))
    assert (result.snapshot.player.hp, result.snapshot.player.mp) == (90, 138)
    assert result.snapshot.player.max_hp == 110
    result = sim.step(SimAction("potion_hp"))
    assert result.snapshot.player.hp == 110


def test_profile_is_immutable_static_knowledge_and_reset_preserves_it():
    sim = Simulator(seed=8, class_spec=custom_spec())
    before = sim.save()
    assert before["schema_version"] == 2
    with pytest.raises(AttributeError):
        sim.class_spec = CLASS_SPECS["assassin"]
    sim.step(SimAction("move", 1))
    sim.reset()
    assert sim.save() == before
    copy_state = sim.save()
    copy_state["class_profile"]["spec"]["attack_damage"] = 99
    assert sim.class_spec.attack_damage == 40


def test_saved_profile_replays_after_catalog_is_removed(tmp_path):
    db = catalog(tmp_path, record(custom_spec()))
    sim = Simulator(seed=17, class_spec=load_class_profile(db, "assassin", "profile-v1").class_spec)
    sim.step(SimAction("move", 1))
    restored = Simulator.restore(json.loads(json.dumps(sim.save())))
    db.unlink()
    assert restored.class_spec == sim.class_spec
    actions = [SimAction("attack")] * 6 + [SimAction("move", 1)] * 35 + [SimAction("loot")]
    assert [sim.step(action) for action in actions] == [restored.step(action) for action in actions]
    assert sim.save() == restored.save()


@pytest.mark.parametrize("status", ["UNKNOWN", "PROVISIONAL", "CONFLICTED"])
def test_nonverified_profiles_are_never_runtime_rules(tmp_path, status):
    db = catalog(tmp_path, record(verification_status=status))
    with pytest.raises(ValueError, match="VERIFIED"):
        load_class_profile(db, "assassin", "profile-v1")


def test_conflicted_profiles_are_not_resolved_by_latest_timestamp(tmp_path):
    db = catalog(tmp_path, record(), record(custom_spec(), source_uri="urn:second:profile"))
    with pytest.raises(ValueError, match="unconflicted"):
        load_class_profile(db, "assassin", "profile-v1")


def test_identical_verified_sources_keep_both_provenance_records(tmp_path):
    db = catalog(tmp_path, record(), record(source_uri="urn:second:profile"))
    loaded = load_class_profile(db, "assassin", "profile-v1")
    assert len(loaded.records) == 2
    assert loaded.profile_sha256 == class_spec_sha256(CLASS_SPECS["assassin"])
    assert {row.source_uri for row in loaded.records} == {
        "urn:self:class-profile",
        "urn:second:profile",
    }


def test_same_payload_with_unverified_candidate_is_explicitly_refused(tmp_path):
    db = catalog(
        tmp_path, record(), record(source_uri="urn:second:profile", verification_status="UNKNOWN")
    )
    with pytest.raises(ValueError, match="VERIFIED"):
        load_class_profile(db, "assassin", "profile-v1")


def test_game_reference_scope_is_not_treated_as_verified_simulator_rules(tmp_path):
    db = catalog(tmp_path, record(scope="game_reference"))
    with pytest.raises(ValueError, match="synthetic_fixture"):
        load_class_profile(db, "assassin", "profile-v1")


def test_missing_version_or_mismatched_job_is_refused(tmp_path):
    db = catalog(tmp_path, record())
    with pytest.raises(ValueError, match="missing"):
        load_class_profile(db, "assassin", "different-version")
    with pytest.raises(ValueError, match="job"):
        load_class_profile(db, "unknown-job", "profile-v1")
    with pytest.raises(ValueError, match="match"):
        Simulator(job="cleric", class_spec=custom_spec())
    with pytest.raises(ValueError, match="ClassSpec"):
        Simulator(class_spec=asdict(custom_spec()))


def test_catalog_entity_and_embedded_profile_id_must_agree(tmp_path):
    db = catalog(tmp_path, record(CLASS_SPECS["cleric"]))
    with pytest.raises(ValueError, match="id"):
        load_class_profile(db, "assassin", "profile-v1")


@pytest.mark.parametrize(
    "change",
    [
        {"id": "unbounded-job"},
        {"max_hp": 0},
        {"max_mp": -1},
        {"max_hp": True},
        {"attack_damage": float("inf")},
        {"attack_damage": 0},
        {"attack_range": 321},
        {"attack_mp_cost": 1000},
        {"attack_cooldown_ms": 0},
        {"attack_cooldown_ms": 60_001},
        {"ammo_cost": -1},
        {"heal_amount": 1000},
        {"walk_speed": 0},
        {"walk_speed": 201},
        {"jump_speed": 401},
        {"jump_speed": float("nan")},
        {"can_jump_attack": 1},
    ],
)
def test_profile_field_bounds_reject_invalid_runtime_rules(change):
    payload = {**asdict(CLASS_SPECS["assassin"]), **change}
    with pytest.raises(ValueError):
        parse_class_spec(payload)


def test_profile_payload_is_complete_and_has_no_unknown_fields():
    payload = asdict(CLASS_SPECS["assassin"])
    payload.pop("walk_speed")
    with pytest.raises(ValueError, match="all ClassSpec fields"):
        parse_class_spec(payload)
    payload = {**asdict(CLASS_SPECS["assassin"]), "download_script": "unrecognized"}
    with pytest.raises(ValueError, match="all ClassSpec fields"):
        parse_class_spec(payload)


def test_embedded_profile_changes_require_a_matching_hash_and_schema():
    state = Simulator(class_spec=custom_spec()).save()
    corrupted = copy.deepcopy(state)
    corrupted["class_profile"]["spec"]["attack_damage"] += 1
    with pytest.raises(ValueError, match="SHA256"):
        Simulator.restore(corrupted)
    corrupted = copy.deepcopy(state)
    corrupted["class_profile"]["spec"]["max_hp"] = -1
    with pytest.raises(ValueError):
        Simulator.restore(corrupted)
    corrupted = copy.deepcopy(state)
    corrupted["schema_version"] = 1
    with pytest.raises(ValueError, match="fields"):
        Simulator.restore(corrupted)
    corrupted = copy.deepcopy(state)
    corrupted.pop("class_profile")
    with pytest.raises(ValueError, match="fields"):
        Simulator.restore(corrupted)
    corrupted = copy.deepcopy(state)
    wrong_job = replace(custom_spec(), id="cleric")
    corrupted["class_profile"] = {"spec": asdict(wrong_job), "sha256": class_spec_sha256(wrong_job)}
    with pytest.raises(ValueError, match="match"):
        Simulator.restore(corrupted)


@pytest.mark.parametrize(
    "seed,initial_hash,after_hash",
    [
        (
            0,
            "2da22e7377ad99c12c32b8e1f0be173c8cd24e76aec45fe650118aff9fdfb7ce",
            "fe1ecb9edfe7dbe0478313f2696d305cb59079f5dafa01ef785017a633ec121d",
        ),
        (
            42,
            "63b25b4f15c0d08a75c1ac0ba311961b5fde2e7a115e32a8bbf973d1fce149ee",
            "4c98808302ba0ecf5d99da30d4f585f1c0efd407efd326aab2afecb9d20769c4",
        ),
    ],
)
def test_default_v1_save_bytes_and_trajectory_remain_unchanged(seed, initial_hash, after_hash):
    # Digests captured from the released/default simulator immediately before adding profiles.
    sim = Simulator(seed=seed)
    assert sim.save()["schema_version"] == 1
    assert "class_profile" not in sim.save()
    assert hashlib.sha256(canonical_json(sim.save()).encode()).hexdigest() == initial_hash
    for action in (
        [SimAction("move", 1)] * 14 + [SimAction("jump_attack")] + [SimAction("attack")] * 20
    ):
        sim.step(action)
    assert hashlib.sha256(canonical_json(sim.save()).encode()).hexdigest() == after_hash
    assert Simulator.restore(json.loads(json.dumps(sim.save()))).save() == sim.save()


def test_provenance_roundtrip_is_offline_and_bound_to_the_profile(tmp_path):
    db = catalog(tmp_path, record(custom_spec()))
    loaded = load_class_profile(db, "assassin", "profile-v1")
    value = json.loads(json.dumps(asdict(loaded)))
    db.unlink()
    assert parse_loaded_class_profile(value) == loaded
    value["records"][0]["source_uri"] = "urn:changed:source"
    with pytest.raises(ValueError, match="modified"):
        parse_loaded_class_profile(value)


@pytest.mark.parametrize("field", ["profile_sha256", "provenance_sha256"])
def test_provenance_hash_tampering_is_rejected(tmp_path, field):
    db = catalog(tmp_path, record())
    value = asdict(load_class_profile(db, "assassin", "profile-v1"))
    value[field] = "f" * 64
    with pytest.raises(ValueError, match="SHA256"):
        parse_loaded_class_profile(value)


def test_provenance_payload_cannot_be_swapped_with_other_valid_rules(tmp_path):
    db = catalog(tmp_path, record())
    value = asdict(load_class_profile(db, "assassin", "profile-v1"))
    value["class_spec"] = asdict(custom_spec())
    value["profile_sha256"] = class_spec_sha256(custom_spec())
    with pytest.raises(ValueError, match="embedded rules"):
        parse_loaded_class_profile(value)
