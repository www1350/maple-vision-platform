"""One structured policy across authored classes, with real execution boundaries."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.agent import VisualAgent
from maple_vision.catalog import canonical_json
from maple_vision.class_package import ClassPackage, compile_class_package, load_class_package
from maple_vision.contracts import Detection, Status
from maple_vision.lab import CapabilityRunConfig, run_episode, verify_perception, verify_run
from maple_vision.map_pack import load_map_pack
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.simulation import Simulator
from maple_vision.state_estimation import TemporalStateEstimator
from maple_vision.task_runtime import TaskRuntime, TaskRuntimeConfig
from maple_vision.vision import PixelPerception, VisualObservation

ROOT = Path(__file__).resolve().parents[1]


def package(name="bandit"):
    return load_class_package(ROOT / "examples/classes" / f"{name}.json")


def runtime(name="bandit", **kwargs):
    return TaskRuntime(name, capabilities=compile_class_package(package(name)), **kwargs)


def observation(sequence=0, *, timestamp_ms=None, target_x=70, hp=1.0, mp=1.0, **changes):
    value = VisualObservation(
        sequence,
        sequence * 50 if timestamp_ms is None else timestamp_ms,
        Status.KNOWN,
        1.0,
        (
            Detection("player", (100, 146, 114, 168), 1.0),
            Detection("platform", (0, 168, 320, 180), 1.0),
            Detection("monster", (target_x, 152, target_x + 18, 168), 1.0),
        ),
        hp,
        mp,
        "meadow",
    )
    return replace(value, **changes)


def decide(policy, value, now_ms=None):
    now_ms = value.timestamp_ms if now_ms is None else now_ms
    belief = TemporalStateEstimator().update(value, now_ms)
    return policy.decide(belief, now_ms)


def execute(policy, decision, *, dispatch_ms=None, receipt_ms=None, accepted=True, reason="cast"):
    assert decision.intent is not None
    when = decision.intent.created_at_ms if dispatch_ms is None else dispatch_ms
    result = policy.scheduler.dispatch(decision.intent, when)
    assert result.accepted
    return policy.record_execution(
        decision.intent.intent_id, accepted, reason, when if receipt_ms is None else receipt_ms
    )


def test_external_class_identity_and_compiled_cache_are_strict():
    caps = compile_class_package(package())
    with pytest.raises(ValueError):
        TaskRuntime("archer", capabilities=caps)
    with pytest.raises(ValueError):
        TaskRuntime("bandit", capabilities=caps, class_spec=CLASS_SPECS["assassin"])
    with pytest.raises(ValueError):
        TaskRuntime("bandit")
    with pytest.raises(ValueError):
        TaskRuntime(
            "bandit",
            capabilities=replace(caps, class_spec=replace(caps.class_spec, ammo_cost=False)),
        )
    with pytest.raises(ValueError):
        TaskRuntime(
            "bandit",
            capabilities=caps,
            config=TaskRuntimeConfig(intent_ttl_ms=100, plan_timeout_ms=100),
        )


def test_dsl_becomes_a_real_ordered_plan_with_registered_cast_locks():
    policy = runtime()
    first = decide(policy, observation())
    assert first.action == SimAction("cast", -1, "slash")
    assert first.intent.locks == ("combat", "resource", "skill")
    plan = policy.scheduler.plan
    assert [(step.action.skill, step.not_before_ms) for step in plan.steps] == [
        ("slash", 0),
        ("finisher", 150),
    ]
    execute(policy, first)
    waiting = decide(policy, observation(1))
    assert waiting.intent is None and waiting.reason == "waiting_for_plan_step"
    second = decide(policy, observation(2, timestamp_ms=150))
    assert second.intent.plan_id == first.intent.plan_id and second.intent.step_index == 1
    assert second.action.skill == "finisher"
    assert second.intent.source_sequence == 2
    assert not policy.scheduler.dispatch(first.intent, 150).accepted


def test_delayed_receipt_controls_cooldown_instead_of_static_program_offsets():
    policy = runtime()
    first = decide(policy, observation())
    execute(policy, first, dispatch_ms=100, receipt_ms=200)
    state = policy.snapshot()["capabilities"]
    assert state["cooldown_ready_ms"]["weapon"] == 350
    assert set(state["skill_attempts"]) == {"slash"}
    waiting = decide(policy, observation(1, timestamp_ms=250))
    assert waiting.intent is None and waiting.reason == "capability_cooldown"
    next_step = decide(policy, observation(2, timestamp_ms=350))
    assert next_step.action.skill == "finisher"
    execute(policy, next_step)
    assert policy.snapshot()["capabilities"]["cooldown_ready_ms"]["weapon"] == 650
    still_waiting = decide(policy, observation(3, timestamp_ms=450))
    assert still_waiting.action == SimAction()
    assert policy.snapshot()["capabilities"]["skill_attempts"]["finisher"]["count"] == 1


def test_same_capture_time_cannot_refund_unobserved_resource_spending():
    policy = runtime()
    first = decide(policy, observation(mp=0.3))
    execute(policy, first)
    assert policy.snapshot()["capabilities"]["resource_estimate"]["mp_lower_bound"] == 19
    second = decide(policy, observation(1, timestamp_ms=0, mp=0.3), now_ms=150)
    assert second.action.skill == "finisher"
    assert second.evidence["capabilities"]["resource_estimate"]["mp_lower_bound"] == 19
    execute(policy, second)
    bounds = policy.snapshot()["capabilities"]["resource_estimate"]
    assert bounds["mp_lower_bound"] == 17 and bounds["unobserved_mp_spend"] == 4
    assert bounds["unobserved_hp_spend"] == 3


def test_survival_preempts_remaining_combo_without_charging_an_unexecuted_skill():
    policy = runtime()
    first = decide(policy, observation())
    execute(policy, first)
    emergency = decide(policy, observation(1, timestamp_ms=150, hp=0.2))
    assert emergency.task == "survival" and emergency.action.kind == "potion_hp"
    assert emergency.intent.plan_id != first.intent.plan_id
    capabilities = policy.snapshot()["capabilities"]
    assert set(capabilities["skill_attempts"]) == {"slash"}
    assert capabilities["resource_estimate"]["unobserved_hp_spend"] == 0
    assert not policy.scheduler.dispatch(first.intent, 150).accepted


@pytest.mark.parametrize("change", ["pause", "source", "unknown", "target"])
def test_compound_continuations_cancel_on_new_context_without_false_effect_success(change):
    policy = runtime()
    first = decide(policy, observation())
    execute(policy, first)
    if change == "pause":
        stopped = policy.interrupt("test_pause", 50)
        assert stopped.evidence["verification"][0]["status"] == "UNKNOWN"
        value = observation(2, timestamp_ms=150)
    elif change == "unknown":
        stopped = decide(policy, observation(1, status=Status.UNKNOWN))
        assert stopped.state == "paused"
        value = observation(2, timestamp_ms=150)
    elif change == "source":
        value = observation(1, timestamp_ms=150, source_id="second-camera")
    else:
        value = observation(1, timestamp_ms=150, target_x=240)
    next_decision = decide(policy, value)
    assert next_decision.intent is not None
    assert next_decision.intent.plan_id != first.intent.plan_id
    assert next_decision.intent.step_index == 0
    assert set(policy.snapshot()["capabilities"]["skill_attempts"]) == {"slash"}
    assert all(
        effect["status"] not in {"SUCCESS", "LIKELY_SUCCESS"}
        for effect in next_decision.evidence["verification"]
    )
    assert next_decision.action.skill != "finisher"


def test_denied_cast_records_failure_and_attempt_budget_without_inventing_effect():
    policy = runtime(config=TaskRuntimeConfig(max_denied_actions=1))
    first = decide(policy, observation())
    execute(policy, first, accepted=False, reason="insufficient_mp")
    result = decide(policy, observation(1))
    assert result.state == "failed"
    assert result.evidence["verification"][0]["status"] == "FAILED"
    attempt = policy.snapshot()["capabilities"]["skill_attempts"]["slash"]
    assert attempt["accepted"] is False and attempt["count"] == 1


def test_new_observation_with_insufficient_resources_cancels_finisher():
    policy = runtime()
    first = decide(policy, observation())
    execute(policy, first)
    following = decide(policy, observation(1, timestamp_ms=150, mp=0.01))
    assert following.action.kind == "potion_mp"
    assert following.intent.plan_id != first.intent.plan_id
    assert "finisher" not in policy.snapshot()["capabilities"]["skill_attempts"]


def test_zero_mp_class_does_not_attempt_impossible_mana_replenishment():
    value = package().to_dict()
    value["resources"]["max_mp"] = 0
    for skill in value["skills"]:
        for level in skill["levels"]:
            level["mp_cost"] = 0
    caps = compile_class_package(ClassPackage.from_dict(value))
    policy = TaskRuntime("bandit", capabilities=caps)
    result = decide(policy, observation(mp=0))
    assert result.action.kind == "cast" and result.action.skill == "slash"


def test_real_pixels_select_hp_area_and_heal_programs_and_keep_damage_unconfirmed():
    source = package("blood_knight")
    renderer, perception = SceneRenderer(), PixelPerception()
    for low_hp in (False, True):
        estimator = TemporalStateEstimator()
        state = Simulator(job=source.class_id, class_package=source).save()
        state["player"]["x"] = 125.0
        if low_hp:
            state["player"]["hp"] = 55
        state["maps"]["meadow"]["monsters"] = [
            {"id": key, "x": x, "hp": 100, "max_hp": 100}
            for key, x in (("left", 90.0), ("right", 170.0))
        ]
        sim = Simulator.restore(state)
        policy = TaskRuntime(source.class_id, capabilities=sim.capabilities)
        obs = perception.observe(renderer.render(sim.snapshot()))
        decision = policy.decide(estimator.update(obs, 0), 0)
        assert decision.action.skill == ("recover" if low_hp else "blood_wave")
        dispatch = policy.scheduler.dispatch(decision.intent, 0)
        outcome = sim.step(dispatch.action)
        policy.record_execution(decision.intent.intent_id, outcome.accepted, outcome.reason, 0)
        following = perception.observe(renderer.render(outcome.snapshot))
        next_decision = policy.decide(estimator.update(following, 50), 50)
        verification = next_decision.evidence["verification"][0]
        assert verification["status"] == ("SUCCESS" if low_hp else "UNKNOWN")
        if low_hp:
            assert outcome.snapshot.player.hp == 90
        else:
            assert outcome.snapshot.player.hp == 148
            assert [monster.hp for monster in outcome.snapshot.monsters] == [70, 70]


@pytest.mark.parametrize("name", ["archer", "bandit", "blood_knight"])
@pytest.mark.parametrize("external_maps", [False, True], ids=["two_maps", "three_maps"])
def test_all_external_packages_complete_real_visual_missions_with_shared_runtime(
    name, external_maps, monkeypatch
):
    def reject_legacy(*args, **kwargs):
        raise AssertionError("new classes cannot delegate to VisualAgent")

    monkeypatch.setattr(VisualAgent, "decide", reject_legacy)
    maps = load_map_pack(ROOT / "examples/maps/three-map-fixture.json") if external_maps else None
    result = run_episode(
        config=CapabilityRunConfig(seed=7, job=name, max_steps=1500),
        class_package=package(name),
        runtime=RuntimeConfig(),
        map_pack=maps,
    )
    count = 3 if external_maps else 2
    assert result["status"] == "completed" and result["independent_goal_check"] and result["alive"]
    assert result["events"]["monster_defeated"] == result["drops_collected"] == 3 * count
    assert result["events"]["dialogue"] == count and result["events"]["map_changed"] == count - 1
    assert result["events"]["cast"] > 0
    if name == "archer":
        assert result["events"]["skill:air_arrow"] == count
        assert result["events"]["skill:arrow_shot"] > 0
    if name == "bandit":
        assert result["events"]["skill:slash"] == result["events"]["skill:finisher"] == 3 * count


@pytest.mark.parametrize("name", ["archer", "bandit", "blood_knight"])
def test_frozen_inputs_fail_boundedly_with_unknown_cast_effects(name):
    result = run_episode(
        config=CapabilityRunConfig(seed=7, job=name, max_steps=180, fault="freeze_input"),
        class_package=package(name),
        runtime=RuntimeConfig(),
    )
    assert result["status"] == "failed" and not result["independent_goal_check"]
    assert result["steps"] < 180
    assert result["reason"] in {"combat_progress_timeout", "recovery_budget_exhausted"}


def test_capability_run_replay_and_reinference_use_embedded_package_after_source_deletion(tmp_path):
    original = tmp_path / "author.json"
    original.write_text(json.dumps(package().to_dict()))
    loaded = load_class_package(original)
    bundle = tmp_path / "run"
    summary = run_episode(
        bundle,
        CapabilityRunConfig(job="bandit", max_steps=8, record_stride=1),
        class_package=loaded,
        runtime=RuntimeConfig(),
    )
    original.unlink()
    assert summary["steps"] == 8
    assert verify_run(bundle)["verified"]
    assert verify_perception(bundle)["structured_runtime_reproduced"]


# Recorded before capability edits from runtime source SHA256
# 73389c477ee4bb41edfa035a0b466d087d0444f20f992765ef54da6d9aec8fae.
@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ("assassin", "668cc99ab2bf5da33a302a05551fc7881633f510a3f8746ce2f37ac0f5881f1f"),
        ("dragon_knight", "19230116724324e27677c3a40dbd1492233c86129c0a54c28a4c855a3efed65a"),
        ("cleric", "30a15ab04b352252ac6350910e01b36c9c9834fe962591923a6d2055c6b75ee1"),
    ],
)
def test_none_path_preserves_complete_legacy_decision_dispatch_receipt_and_snapshot_trace(
    job, expected
):
    sim, renderer, perception = Simulator(seed=7, job=job), SceneRenderer(), PixelPerception()
    estimator, policy = TemporalStateEstimator(), TaskRuntime(job, capabilities=None)
    trace = hashlib.sha256()
    for _ in range(1500):
        observation = perception.observe(renderer.render(sim.snapshot()))
        decision = policy.decide(estimator.update(observation, sim.timestamp_ms), sim.timestamp_ms)
        trace.update(canonical_json(decision.to_dict()).encode())
        if decision.state in {"completed", "failed", "stopped"}:
            trace.update(canonical_json(policy.snapshot()).encode())
            break
        if decision.intent is None:
            sim.step()
        else:
            dispatch = policy.scheduler.dispatch(decision.intent, sim.timestamp_ms)
            before = sim.timestamp_ms
            outcome = sim.step(dispatch.action)
            receipt = policy.record_execution(
                decision.intent.intent_id, outcome.accepted, outcome.reason, before
            )
            trace.update(canonical_json(dispatch.to_dict()).encode())
            trace.update(canonical_json(receipt.to_dict()).encode())
            trace.update(canonical_json(policy.snapshot()).encode())
    assert decision.state == "completed"
    assert "capabilities" not in policy.snapshot() and "capabilities" not in decision.evidence
    assert trace.hexdigest() == expected
