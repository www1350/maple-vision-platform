"""Actual capability effects and source-bound independent simulator restore."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.catalog import canonical_json
from maple_vision.class_package import load_class_package
from maple_vision.map_pack import load_map_pack
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.simulation import Simulator

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/classes"


def monster(x=100.0, hp=100, key="target"):
    return {"id": key, "x": x, "hp": hp, "max_hp": hp}


def configured(name="archer", *, x=24, monsters=(), **player):
    package = load_class_package(EXAMPLES / f"{name}.json")
    state = Simulator(job=name, class_package=package).save()
    state["player"].update(x=float(x), **player)
    state["maps"]["meadow"]["monsters"] = list(monsters)
    return Simulator.restore(state)


def cast(skill, direction=1):
    return SimAction("cast", direction, skill)


def wait_until(sim, timestamp):
    while sim.timestamp_ms < timestamp:
        sim.step()


def test_archer_selected_skill_has_real_direction_range_ammo_and_mana_effects():
    sim = configured(monsters=[monster(100), monster(0, key="behind"), monster(200, key="far")])
    result = sim.step(cast("arrow_shot"))
    assert result.accepted and result.events.count("attack_hit") == 1
    health = {item.id: item.hp for item in result.snapshot.monsters}
    assert health == {"target": 78, "behind": 100, "far": 100}
    assert result.snapshot.player.mp == 68 and result.snapshot.player.ammo == 249
    denied = sim.step(cast("arrow_shot"))
    assert denied.reason == "capability_cooldown"
    assert denied.snapshot.player.mp == 68 and denied.snapshot.player.ammo == 249


def test_bandit_compiled_combo_executes_two_different_effects_at_separate_times():
    sim = configured("bandit", x=90, monsters=[monster(125, hp=50)])
    program = sim.capabilities.program_for("combat")
    first, second = program.instantiate(1)
    one = sim.step(first.action)
    assert one.accepted and one.snapshot.monsters[0].hp == 32
    assert one.snapshot.player.hp == 120 and one.snapshot.player.mp == 68
    assert sim.step(second.action).reason == "capability_cooldown"
    wait_until(sim, second.not_before_ms)
    two = sim.step(second.action)
    assert two.accepted and two.snapshot.monsters[0].hp == 12
    assert two.snapshot.player.hp == 117 and two.snapshot.player.mp == 66
    assert sim.save()["capability_state"]["cooldowns"]["weapon"] == 450


def test_program_description_does_not_auto_execute_cancelled_remaining_steps():
    sim = configured("bandit", x=90, monsters=[monster(125, hp=50)])
    first, _ = sim.capabilities.program_for("combat").instantiate(1)
    sim.step(first.action)
    wait_until(sim, 500)
    snapshot = sim.snapshot()
    assert snapshot.monsters[0].hp == 32
    assert snapshot.player.hp == 120 and snapshot.player.mp == 68


def test_health_cost_radial_skill_hits_both_sides_with_target_budget():
    sim = configured(
        "blood_knight",
        x=125,
        monsters=[
            monster(60, key="a"),
            monster(95, key="b"),
            monster(170, key="c"),
            monster(190, key="d"),
        ],
    )
    result = sim.step(cast("blood_wave", -1))
    assert result.accepted and result.events.count("attack_hit") == 3
    assert result.snapshot.player.hp == 148
    assert {item.id: item.hp for item in result.snapshot.monsters} == {
        "a": 70,
        "b": 70,
        "c": 70,
        "d": 100,
    }
    assert result.snapshot.player.mp == 60


@pytest.mark.parametrize("hp", [1, 12])
def test_health_cost_cannot_spend_last_hp_and_denial_is_resource_atomic(hp):
    sim = configured("blood_knight", hp=hp)
    before = sim.save()
    result = sim.step(cast("blood_wave", -1))
    assert not result.accepted and result.reason == "insufficient_hp"
    assert result.snapshot.player.hp == hp and result.snapshot.player.facing == 1
    assert sim.save()["capability_state"] == before["capability_state"]


def test_health_cost_can_leave_exactly_one_hp():
    sim = configured("blood_knight", hp=13)
    result = sim.step(cast("blood_wave"))
    assert result.accepted and result.snapshot.player.hp == 1 and result.snapshot.player.alive


@pytest.mark.parametrize(
    ("player", "reason"), [({"mp": 1}, "insufficient_mp"), ({"ammo": 0}, "insufficient_ammo")]
)
def test_resource_failures_leave_all_other_resources_and_cooldowns_unchanged(player, reason):
    sim = configured(**player)
    before = sim.save()
    result = sim.step(cast("arrow_shot"))
    assert not result.accepted and result.reason == reason
    for key in ("hp", "mp", "ammo"):
        assert sim.save()["player"][key] == before["player"][key]
    assert sim.save()["capability_state"] == before["capability_state"]


def test_airborne_and_grounded_constraints_use_declared_skills():
    sim = configured(monsters=[monster(100)])
    assert sim.step(cast("air_arrow")).reason == "skill_requires_air"
    assert sim.step(SimAction("jump")).accepted
    assert sim.step(cast("arrow_shot")).reason == "skill_requires_ground"
    shot = sim.step(cast("air_arrow"))
    assert shot.accepted and shot.snapshot.player.ammo == 249
    assert shot.snapshot.monsters[0].hp == 82


def test_buff_increases_registered_damage_expires_and_restores_without_external_source():
    sim = configured(monsters=[monster(100)])
    assert sim.step(cast("focus", 0)).accepted
    restored = Simulator.restore(json.loads(json.dumps(sim.save())))
    assert restored.class_package == sim.class_package
    assert restored.capabilities.to_dict() == sim.capabilities.to_dict()
    for current in (sim, restored):
        result = current.step(cast("arrow_shot"))
        assert result.snapshot.monsters[0].hp == 73
        assert result.snapshot.player.buffs == ("focus",)
        wait_until(current, 500)
        assert current.step(cast("focus", 0)).reason == "buff_already_active"
        wait_until(current, 3000)
        assert current.snapshot().player.buffs == ()
        assert "buff_expired:focus" in current.snapshot().events
    assert sim.save() == restored.save()


def test_heal_is_registered_capped_and_does_not_consume_on_full_health():
    sim = configured("blood_knight", hp=140)
    result = sim.step(cast("recover", 0))
    assert result.accepted and result.snapshot.player.hp == 160
    assert result.snapshot.player.mp == 52
    wait_until(sim, 500)
    denied = sim.step(cast("recover", 0))
    assert denied.reason == "hp_full" and denied.snapshot.player.mp == 52


def test_legacy_skill_entry_cannot_bypass_package_registry():
    for kind in ("attack", "jump_attack", "buff", "heal"):
        sim = configured()
        before = sim.save()
        result = sim.step(SimAction(kind))
        assert result.reason == "capability_cast_required"
        assert sim.save()["player"] == before["player"]
        assert sim.save()["capability_state"] == before["capability_state"]
    assert configured().step(cast("unregistered")).reason == "skill_unavailable"
    assert Simulator().step(cast("arrow_shot")).reason == "capability_package_required"


@pytest.mark.parametrize("name", ["archer", "bandit", "blood_knight"])
@pytest.mark.parametrize("external_maps", [False, True])
def test_full_embedded_sources_restore_after_deletion_and_continue_identically(
    name, external_maps, tmp_path
):
    path = tmp_path / "source.json"
    path.write_bytes((EXAMPLES / f"{name}.json").read_bytes())
    package = load_class_package(path)
    maps = load_map_pack(ROOT / "examples/maps/three-map-fixture.json") if external_maps else None
    sim = Simulator(
        seed=7, job=name, class_package=package, map_pack=maps, map_id="grove" if maps else "meadow"
    )
    path.unlink()
    for _ in range(12):
        sim.step(SimAction("move", 1))
    program = sim.capabilities.program_for("combat")
    sim.step(program.instantiate(1)[0].action)
    saved = json.loads(json.dumps(sim.save()))
    assert saved["schema_version"] == 4
    restored = Simulator.restore(saved)
    assert restored.save() == saved
    assert restored.class_package == package
    renderer = SceneRenderer(map_pack=maps)
    assert renderer.render(restored.snapshot()).pixels == renderer.render(sim.snapshot()).pixels
    actions = [SimAction()] * 10 + [
        program.instantiate(1)[-1].action,
        SimAction("move", -1),
        SimAction("jump"),
    ]
    assert [sim.step(action) for action in actions] == [restored.step(action) for action in actions]
    assert sim.save() == restored.save()
    restored.reset()
    assert (
        restored.save()
        == Simulator(
            seed=7,
            job=name,
            class_package=package,
            map_pack=maps,
            map_id="grove" if maps else "meadow",
        ).save()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(schema_version=True),
        lambda s: s.update(schema_version=3),
        lambda s: s.pop("class_package"),
        lambda s: s.pop("capability_state"),
        lambda s: s.update(job="bandit"),
        lambda s: s["class_package"].update(content_sha256="0" * 64),
        lambda s: s["class_package"]["content"]["resources"].update(max_hp=999),
        lambda s: s["capability_state"]["cooldowns"].update(unregistered=0),
        lambda s: s["capability_state"]["cooldowns"].update(weapon=True),
        lambda s: s["capability_state"]["cooldowns"].update(weapon=-1),
        lambda s: s["capability_state"]["cooldowns"].update(weapon=999999),
        lambda s: s["capability_state"]["buffs"].update(arrow_shot=100),
        lambda s: s["capability_state"]["buffs"].update(unregistered=100),
        lambda s: s["capability_state"]["buffs"].update(focus=0),
        lambda s: s["capability_state"]["buffs"].update(focus=999999),
        lambda s: s["player"].update(buff_until_ms=100),
        lambda s: s["player"].update(attack_ready_ms=100),
        lambda s: s["player"].update(skill_ready_ms=100),
    ],
)
def test_restore_rejects_forged_package_identity_and_impossible_ability_state(mutate):
    state = configured().save()
    mutate(state)
    with pytest.raises(ValueError):
        Simulator.restore(state)


def test_package_profile_exclusion_and_unknown_jobs_are_explicit():
    package = load_class_package(EXAMPLES / "archer.json")
    with pytest.raises(ValueError):
        Simulator(job="archer")
    with pytest.raises(ValueError):
        Simulator(job="bandit", class_package=package)
    with pytest.raises(ValueError):
        Simulator(job="archer", class_package=package, class_spec=CLASS_SPECS["assassin"])
    with pytest.raises(ValueError):
        Simulator(job="archer", class_package=package.to_dict())


def test_saved_and_compiled_containers_do_not_mutate_live_state():
    sim = configured()
    first = sim.save()
    altered = deepcopy(first)
    altered["class_package"]["content"]["resources"]["max_hp"] = 1
    altered["capability_state"]["cooldowns"]["weapon"] = 999
    assert sim.save() == first


@pytest.mark.parametrize("variant", ["assassin", "dragon_knight", "cleric", "profile", "maps"])
def test_none_path_keeps_pre_capability_state_and_render_bytes(variant):
    baseline = json.loads((ROOT / "tests/fixtures/capability-legacy-baseline.json").read_text())
    maps = (
        load_map_pack(ROOT / "examples/maps/three-map-fixture.json") if variant == "maps" else None
    )
    profile = (
        replace(CLASS_SPECS["assassin"], attack_damage=28, walk_speed=70)
        if variant == "profile"
        else None
    )
    sim = Simulator(
        seed=7,
        job=variant if variant in CLASS_SPECS else "assassin",
        map_id="grove" if maps else "meadow",
        map_pack=maps,
        class_spec=profile,
        class_package=None,
    )
    renderer = SceneRenderer(map_pack=maps)
    trajectory, frames = hashlib.sha256(), hashlib.sha256()
    actions = [SimAction(**action) for action in baseline["actions"]]
    for action in [None, *actions]:
        if action is not None:
            sim.step(action)
        trajectory.update(canonical_json(sim.save()).encode())
        frames.update(renderer.render(sim.snapshot()).pixels)
    assert trajectory.hexdigest() == baseline["variants"][variant]["trajectory_sha256"]
    assert frames.hexdigest() == baseline["variants"][variant]["frames_sha256"]
