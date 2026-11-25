"""Independent fixed-clock expectations for authored timed skill effects."""

import copy
from pathlib import Path

import pytest

from maple_vision.class_package import ClassPackage, load_class_package
from maple_vision.lab import CapabilityRunConfig, run_episode, verify_perception, verify_run
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "classes"


def setup(job="fire_poison", hp=50, step=50):
    package = load_class_package(EXAMPLES / f"{job}.json")
    simulator = Simulator(job=job, class_package=package, step_ms=step)
    state = simulator.save()
    player = state["player"]
    player["x"] = 100
    monster = state["maps"]["meadow"]["monsters"][0]
    monster.update(x=100, hp=hp, max_hp=hp)
    return Simulator.restore(state), monster["id"]


def health(sim, target):
    return next(m.hp for m in sim.snapshot().monsters if m.id == target)


def test_ticks_at_absolute_deadlines_and_mid_effect_restore():
    sim, target = setup(step=70)
    sim.step(SimAction("cast", skill="ember"))
    assert health(sim, target) == 38
    restored = Simulator.restore(sim.save())
    expected = [(140, 38), (210, 33), (280, 33), (350, 28), (420, 28), (490, 23), (560, 23)]
    for now, hp in expected:
        assert sim.step() == restored.step()
        assert health(sim, target) == hp and sim.timestamp_ms == now
        assert sim.save() == restored.save()
    assert sim.save()["target_effects"] == []


def test_one_drop_at_lethal_tick_and_no_extra_ticks():
    sim, target = setup(hp=17)
    assert sim.step(SimAction("cast", skill="ember")).accepted
    sim.step()
    outcome = sim.step()
    assert "monster_defeated" in outcome.events
    assert len(sim.snapshot().drops) == 1
    assert all(e["target"] != target for e in sim.save()["target_effects"])
    for _ in range(18):
        sim.step()
    assert len(sim.snapshot().drops) == 1


def test_refresh_does_not_stack_or_delay_and_rejected_cast_has_no_cost():
    sim, target = setup(hp=100)
    sim.step(SimAction("cast", skill="ember"))
    mp = sim.snapshot().player.mp
    denied = sim.step(SimAction("cast", skill="ember"))
    assert not denied.accepted and sim.snapshot().player.mp == mp
    while sim.timestamp_ms < 350:
        sim.step()
    existing = sim.save()["target_effects"]
    assert sim.step(SimAction("cast", skill="ember")).accepted
    assert existing == sim.save()["target_effects"]
    sim.step()
    assert health(sim, target) == 100 - 24 - 15 and sim.save()["target_effects"] == []


def test_contact_suppression_expires_and_isolated_per_target():
    sim, _ = setup("ice_lightning")
    initial = sim.snapshot().player.hp
    cast = sim.step(SimAction("cast", skill="frost"))
    assert cast.accepted and "disable_contact_applied" in cast.events
    while sim.timestamp_ms < 450:
        sim.step()
        assert sim.snapshot().player.hp == initial
    assert sim.save()["target_effects"] == []
    assert sim.step().snapshot.player.hp == initial - 9


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(schema_version=4),
        lambda s: s["target_effects"][0].update(target="absent"),
        lambda s: s["target_effects"][0].update(skill="other"),
        lambda s: s["target_effects"][0].update(applied_ms=True),
        lambda s: s["target_effects"][0].update(applied_ms=1),
        lambda s: s["target_effects"][0].update(applied_ms=100),
        lambda s: s["target_effects"].append(copy.copy(s["target_effects"][0])),
    ],
)
def test_invalid_save_rejected(change):
    sim, _ = setup()
    sim.step(SimAction("cast", skill="ember"))
    sim.step()
    state = sim.save()
    change(state)
    with pytest.raises(ValueError):
        Simulator.restore(state)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(schema_version=1),
        lambda p: p["timed_effects"].clear(),
        lambda p: p["timed_effects"][0].update(skill="unknown"),
        lambda p: p["timed_effects"][0].update(duration_ms=99),
        lambda p: p["timed_effects"][0].update(interval_ms=0),
        lambda p: p["timed_effects"][0].update(tick_damage=True),
        lambda p: p["timed_effects"][0].update(kind="disable_contact"),
    ],
)
def test_invalid_rules_rejected(change):
    payload = load_class_package(EXAMPLES / "fire_poison.json").to_dict()
    change(payload)
    with pytest.raises(ValueError):
        ClassPackage.from_dict(payload)


@pytest.mark.parametrize(
    ("job", "event"),
    [("fire_poison", "periodic_hit"), ("ice_lightning", "disable_contact_applied")],
)
def test_actual_visual_mission_and_reinference(job, event, tmp_path):
    package = load_class_package(EXAMPLES / f"{job}.json")
    run = tmp_path / "mission"
    result = run_episode(
        run,
        CapabilityRunConfig(seed=14, job=job, max_steps=1500),
        class_package=package,
        runtime=RuntimeConfig(),
    )
    assert result["status"] == "completed" and result["independent_goal_check"]
    assert result["events"][event] > 0
    assert verify_run(run)["verified"]
    assert verify_perception(run)["structured_runtime_reproduced"]


def test_portal_clears_active_effects_and_save_roundtrips():
    sim, _ = setup()
    sim.step(SimAction("cast", skill="ember"))
    state = sim.save()
    assert state["target_effects"]
    state["player"]["x"] = 290
    transferred = Simulator.restore(state)
    outcome = transferred.step(SimAction("portal"))
    assert outcome.accepted and outcome.snapshot.map_id == "cavern"
    assert transferred.save()["target_effects"] == []
    assert Simulator.restore(transferred.save()).save() == transferred.save()


def test_control_does_not_make_other_monsters_harmless():
    sim, _ = setup("ice_lightning")
    state = sim.save()
    raw = state["class_package"]["content"]
    raw["skills"][0]["levels"][0]["max_targets"] = 1
    raw["bindings"]["area"] = None
    state["class_package"] = ClassPackage.from_dict(raw).manifest()
    state["maps"]["meadow"]["monsters"][1]["x"] = 100
    sim = Simulator.restore(state)
    result = sim.step(SimAction("cast", skill="frost"))
    assert "disable_contact_applied" in result.events and "damage" in result.events
    assert len(sim.save()["target_effects"]) == 1
