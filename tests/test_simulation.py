"""Fixture mechanics and restore tests; ground truth is used only as a test oracle."""

import copy
import json
import math
import random
from dataclasses import FrozenInstanceError

import pytest

from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.simulation import Simulator


def configured(job="assassin", *, x=24.0, hp=None, mp=None, monsters=None, **player_fields):
    """Set explicit initial conditions through the public validated scenario save format."""
    state = Simulator(job=job).save()
    state["player"]["x"] = x
    if hp is not None:
        state["player"]["hp"] = hp
    if mp is not None:
        state["player"]["mp"] = mp
    state["player"].update(player_fields)
    if monsters is not None:
        state["maps"]["meadow"]["monsters"] = monsters
    return Simulator.restore(state)


def monster(x=80.0, hp=36, key="target"):
    return {"id": key, "x": x, "hp": hp, "max_hp": hp}


def advance(sim, count):
    for _ in range(count):
        sim.step(SimAction())


@pytest.mark.parametrize("job", CLASS_SPECS)
def test_each_class_is_an_explicit_fixture(job):
    sim = Simulator(job=job)
    snapshot = sim.snapshot()
    assert snapshot.player.job == job
    assert snapshot.player.alive
    assert snapshot.player.hp == CLASS_SPECS[job].max_hp
    assert snapshot.map_id == "meadow"
    assert snapshot.player.grounded
    assert len(snapshot.monsters) == 3
    assert snapshot.timestamp_ms == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=-1),
        dict(seed=True),
        dict(seed=2**63),
        dict(step_ms=0),
        dict(step_ms=True),
        dict(job="official-class"),
        dict(map_id="unknown"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Simulator(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="inject"),
        dict(kind="move"),
        dict(kind="climb"),
        dict(direction=True),
        dict(direction=2),
        dict(skill=[]),
        dict(kind="idle", skill="area"),
        dict(kind="loot", target="x"),
        dict(kind="loot", direction=1),
    ],
)
def test_actions_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SimAction(**kwargs)


def test_snapshot_is_an_immutable_copy():
    sim = Simulator()
    before = sim.snapshot()
    sim.step(SimAction("move", 1))
    assert before.player.x == 24
    assert sim.snapshot().player.x == 28
    with pytest.raises(FrozenInstanceError):
        before.player.x = 100


def test_movement_uses_elapsed_time_and_bounds():
    sim = Simulator(step_ms=25)
    for _ in range(100):
        sim.step(SimAction("move", -1))
    result = sim.snapshot()
    assert result.player.x == 0
    assert result.sequence == 100
    assert result.timestamp_ms == 2500
    assert result.player.grounded


def test_jump_has_one_apex_and_no_double_jump():
    sim = configured(monsters=[])
    first = sim.step(SimAction("jump"))
    assert first.accepted
    assert first.snapshot.player.y < 146
    assert first.snapshot.player.vy < 0
    second = sim.step(SimAction("jump"))
    assert not second.accepted
    assert second.reason == "airborne"
    positions = [first.snapshot.player.y, second.snapshot.player.y]
    for _ in range(30):
        positions.append(sim.step().snapshot.player.y)
    assert min(positions) < 100
    assert positions[-1] == 146
    assert sim.snapshot().player.grounded
    apex = positions.index(min(positions))
    assert all(a >= b for a, b in zip(positions[:apex], positions[1 : apex + 1], strict=True))
    assert all(a <= b for a, b in zip(positions[apex:-1], positions[apex + 1 :], strict=True))


def test_jump_lands_on_one_way_platform():
    sim = configured(x=90, monsters=[])
    sim.step(SimAction("jump"))
    advance(sim, 20)
    assert sim.snapshot().player.y == 104  # platform top126 minus player height22
    assert sim.snapshot().player.grounded
    for _ in range(15):
        sim.step(SimAction("move", 1))
    advance(sim, 20)
    assert sim.snapshot().player.y == 146


def test_ladder_climb_and_release():
    sim = configured(x=97, monsters=[])
    for _ in range(20):
        result = sim.step(SimAction("climb", -1))
    assert result.accepted
    assert sim.snapshot().player.y == 104
    assert sim.snapshot().player.grounded
    sim.step(SimAction("move", 1))
    assert sim.snapshot().player.x > 97


def test_attack_facing_cooldown_cost_and_drop():
    sim = configured(monsters=[monster()])
    left = sim.step(SimAction("attack", -1))
    assert left.accepted and "attack_hit" not in left.events
    assert left.snapshot.player.ammo == 199
    assert left.snapshot.player.mp == 58
    assert sim.step(SimAction("attack", 1)).reason == "attack_cooldown"
    advance(sim, 2)
    hit = sim.step(SimAction("attack", 1))
    assert "attack_hit" in hit.events
    assert hit.snapshot.monsters[0].hp == 18
    advance(sim, 3)
    kill = sim.step(SimAction("attack"))
    assert "monster_defeated" in kill.events
    assert not kill.snapshot.monsters
    assert len(kill.snapshot.drops) == 1
    assert kill.snapshot.drops[0].item == "coin"
    assert kill.snapshot.player.ammo == 197


def test_attack_cannot_hit_outside_range_or_vertical_band():
    far = configured(monsters=[monster(200)])
    assert "attack_hit" not in far.step(SimAction("attack")).events
    above = configured(x=90, y=104.0, monsters=[monster(100)])
    assert "attack_hit" not in above.step(SimAction("attack")).events


def test_jump_attack_is_atomic_when_resources_are_missing():
    sim = configured(ammo=0)
    before = sim.snapshot().player
    result = sim.step(SimAction("jump_attack", 1))
    assert not result.accepted
    assert result.reason == "insufficient_ammo"
    assert result.snapshot.player.y == before.y
    assert "jump" not in result.events
    ready = configured(monsters=[monster()])
    result = ready.step(SimAction("jump_attack", 1))
    assert result.accepted
    assert "jump" in result.events and "attack_hit" in result.events
    assert not result.snapshot.player.grounded


def test_area_skill_is_class_specific_and_hits_multiple_targets():
    targets = [monster(90, key="left"), monster(140, key="right"), monster(165, key="far")]
    sim = configured("dragon_knight", x=125, monsters=targets)
    result = sim.step(SimAction("attack", skill="area"))
    assert result.events.count("attack_hit") == 3
    assert result.snapshot.player.mp == 42
    assert result.snapshot.player.ammo == 0
    assert configured().step(SimAction("attack", skill="area")).reason == "skill_unavailable"


def test_loot_requires_proximity_and_is_not_duplicated():
    sim = configured(monsters=[monster(hp=18)])
    sim.step(SimAction("attack"))
    assert sim.step(SimAction("loot")).reason == "no_drop_in_range"
    for _ in range(11):
        sim.step(SimAction("move", 1))
    result = sim.step(SimAction("loot"))
    assert result.accepted and "loot" in result.events
    assert result.snapshot.player.coins >= 1
    assert not result.snapshot.drops
    coins = result.snapshot.player.coins
    assert not sim.step(SimAction("loot")).accepted
    assert sim.snapshot().player.coins == coins


def test_potions_heal_consume_and_share_cooldown():
    sim = configured(hp=20, mp=0, monsters=[])
    result = sim.step(SimAction("potion_hp"))
    assert result.snapshot.player.hp == 65
    assert result.snapshot.player.hp_potions == 7
    assert sim.step(SimAction("potion_mp")).reason == "potion_cooldown"
    advance(sim, 4)
    result = sim.step(SimAction("potion_mp"))
    assert result.snapshot.player.mp == 40
    assert result.snapshot.player.mp_potions == 7
    empty = configured(hp=20, hp_potions=0)
    assert empty.step(SimAction("potion_hp")).reason == "no_hp_potions"
    full = configured()
    assert full.step(SimAction("potion_hp")).reason == "hp_full"
    assert full.snapshot().player.hp_potions == 8


def test_heal_belongs_to_cleric_and_consumes_mp():
    sim = configured("cleric", hp=20, monsters=[])
    result = sim.step(SimAction("heal"))
    assert result.snapshot.player.hp == 55
    assert result.snapshot.player.mp == 108
    assert sim.step(SimAction("heal")).reason == "skill_cooldown"
    assert configured(hp=20).step(SimAction("heal")).reason == "heal_unavailable"
    assert configured("cleric", hp=20, mp=0).step(SimAction("heal")).reason == "insufficient_mp"


def test_buff_increases_damage_then_expires():
    sim = configured(monsters=[monster(hp=100)])
    assert sim.step(SimAction("buff")).accepted
    hit = sim.step(SimAction("attack"))
    assert hit.snapshot.monsters[0].hp == 76
    advance(sim, 98)
    assert not sim.snapshot().player.buffs
    assert "buff_expired" in sim.snapshot().events
    hit = sim.step(SimAction("attack"))
    assert hit.snapshot.monsters[0].hp == 58


def test_contact_damage_has_immunity_and_death_blocks_actions():
    sim = configured(x=80, hp=9, monsters=[monster(80)])
    death = sim.step()
    assert death.snapshot.player.hp == 0
    assert not death.snapshot.player.alive
    assert "death" in death.events
    before = death.snapshot.player
    result = sim.step(SimAction("potion_hp"))
    assert result.reason == "player_dead"
    assert result.snapshot.player == before
    survivor = configured(x=80, hp=100, monsters=[monster(80)])
    survivor.step()
    advance(survivor, 9)
    assert survivor.snapshot().player.hp == 91
    survivor.step()
    assert survivor.snapshot().player.hp == 82


def test_portals_preserve_both_maps_and_resources():
    sim = configured(x=281, hp=80, monsters=[])
    result = sim.step(SimAction("portal", target="to_cavern"))
    assert result.snapshot.map_id == "cavern"
    assert "map_changed" in result.events
    assert result.snapshot.player.hp == 80
    assert len(result.snapshot.monsters) == 3
    back = sim.step(SimAction("portal"))
    assert back.snapshot.map_id == "meadow"
    assert not back.snapshot.monsters
    assert configured().step(SimAction("portal")).reason == "no_portal_in_range"


def test_npc_dialogue_is_local_and_dismisses_when_moving():
    sim = Simulator()
    result = sim.step(SimAction("interact", target="guide"))
    assert result.accepted
    assert result.snapshot.dialogue.startswith("guide:")
    assert "dialogue" in result.events
    assert sim.step(SimAction()).snapshot.dialogue == result.snapshot.dialogue
    assert sim.step(SimAction("move", 1)).snapshot.dialogue is None
    assert sim.step(SimAction("interact", target="keeper")).reason == "no_npc_in_range"


def test_same_seed_and_actions_produce_identical_trajectories():
    actions = [SimAction("move", 1)] * 14 + [SimAction("attack")] * 12
    one, two = Simulator(seed=42), Simulator(seed=42)
    assert [one.step(a) for a in actions] == [two.step(a) for a in actions]
    assert one.save() == two.save()
    assert Simulator(seed=43).snapshot().monsters != Simulator(seed=42).snapshot().monsters


def test_save_roundtrip_json_restores_rng_cooldowns_and_future_events():
    sim = configured(monsters=[monster(80, hp=36), monster(100, hp=36, key="two")])
    sim.step(SimAction("attack"))
    sim.step(SimAction("jump", 1))
    restore = Simulator.restore(json.loads(json.dumps(sim.save(), allow_nan=False)))
    actions = [SimAction("attack")] * 100 + [SimAction("move", 1)] * 18
    assert [sim.step(a) for a in actions] == [restore.step(a) for a in actions]
    assert sim.save() == restore.save()


def test_save_does_not_alias_runtime_state_and_reset_keeps_initial_map():
    sim = Simulator(seed=12, map_id="cavern", job="cleric")
    before = sim.save()
    copy_state = sim.save()
    copy_state["player"]["hp"] = 0
    copy_state["maps"]["cavern"]["monsters"].clear()
    assert sim.save() == before
    sim.step(SimAction("move", 1))
    sim.reset()
    assert sim.save() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("schema_version", True),
        ("ruleset", "official"),
        ("sequence", -1),
        ("sequence", True),
        ("rng_state", 0),
        ("map_id", "missing"),
        ("initial_map_id", []),
        ("events", [5]),
        ("dialogue", 10),
    ],
)
def test_restore_rejects_invalid_root(field, value):
    state = Simulator().save()
    state[field] = value
    with pytest.raises(ValueError):
        Simulator.restore(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", float("nan")),
        ("y", float("inf")),
        ("hp", -1),
        ("mp", 1000),
        ("grounded", 1),
        ("facing", 0),
        ("ammo", True),
        ("y", 100),
        ("climbing", True),
        ("vx", -1000),
    ],
)
def test_restore_rejects_invalid_player(field, value):
    state = Simulator().save()
    state["player"][field] = value
    with pytest.raises(ValueError):
        Simulator.restore(state)


def test_restore_rejects_extra_fields_duplicate_ids_and_bad_items():
    state = Simulator().save()
    state["unrecognized"] = 1
    with pytest.raises(ValueError):
        Simulator.restore(state)
    state = Simulator().save()
    state["maps"]["meadow"]["monsters"].append(
        copy.deepcopy(state["maps"]["meadow"]["monsters"][0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        Simulator.restore(state)
    state = Simulator().save()
    state["maps"]["meadow"]["drops"].append(dict(id="drop-100", x=10, y=160, item=[], quantity=1))
    with pytest.raises(ValueError, match="item"):
        Simulator.restore(state)
    state["maps"]["meadow"]["drops"][0]["item"] = "coin"
    with pytest.raises(ValueError, match="counter"):
        Simulator.restore(state)


def test_random_action_sequences_preserve_physics_resource_and_save_invariants():
    # Seeded independent fuzz of public actions, not an implementation-shaped expected trace.
    rng = random.Random(2791)
    choices = [
        SimAction(),
        SimAction("move", 1),
        SimAction("move", -1),
        SimAction("jump", 1),
        SimAction("jump", -1),
        SimAction("attack"),
        SimAction("loot"),
        SimAction("potion_hp"),
        SimAction("potion_mp"),
        SimAction("buff"),
        SimAction("portal"),
        SimAction("interact"),
        SimAction("climb", -1),
        SimAction("climb", 1),
    ]
    for job in CLASS_SPECS:
        sim = Simulator(seed=31, job=job)
        for index in range(1500):
            result = sim.step(rng.choice(choices))
            p = result.snapshot.player
            assert math.isfinite(p.x) and math.isfinite(p.y)
            assert 0 <= p.x <= 306 and 0 <= p.y <= 146
            assert 0 <= p.hp <= p.max_hp and 0 <= p.mp <= p.max_mp
            assert p.ammo >= 0 and p.hp_potions >= 0 and p.mp_potions >= 0
            if index % 23 == 0:
                sim = Simulator.restore(json.loads(json.dumps(sim.save())))
                assert sim.snapshot() == result.snapshot
