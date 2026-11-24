"""Executable task/intent contracts and independent offline mission outcomes."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from maple_vision.agent import VisualAgent
from maple_vision.contracts import Detection, Status
from maple_vision.lab import RunConfig, run_episode
from maple_vision.map_pack import load_map_pack
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.state_estimation import TemporalStateEstimator
from maple_vision.task_runtime import (
    ActionIntent,
    ActionPlan,
    ActionScheduler,
    ActionStep,
    EffectStatus,
    ExecutionReceipt,
    RuntimeDecision,
    TaskArbiter,
    TaskCandidate,
    TaskInstance,
    TaskRuntime,
    TaskRuntimeConfig,
    TaskStatus,
    verify_action_effect,
)
from maple_vision.vision import VisualObservation


def observation(sequence=0, *, x=100, monsters=(70,), hp=0.8, mp=0.3, **changes):
    value = VisualObservation(
        sequence,
        sequence * 50,
        Status.KNOWN,
        1.0,
        (
            Detection("player", (x, 146, x + 14, 168), 1.0),
            Detection("platform", (0, 168, 320, 180), 1.0),
            *(Detection("monster", (mx, 148, mx + 18, 168), 1.0) for mx in monsters),
        ),
        hp,
        mp,
        "meadow",
    )
    return replace(value, **changes)


def belief(value):
    return TemporalStateEstimator().update(value, value.timestamp_ms)


def plan(scheduler=None, *, steps=None, created_at_ms=0, source_sequence=0, **changes):
    value = ActionPlan(
        "test-plan",
        "task-combat",
        0 if scheduler is None else scheduler.generation,
        source_sequence,
        created_at_ms,
        "meadow",
        created_at_ms,
        created_at_ms + 1000,
        steps or (ActionStep(SimAction("move", -1)),),
    )
    return replace(value, **changes)


def issued(action=SimAction("move", 1)):
    scheduler = ActionScheduler()
    scheduler.start(plan(steps=(ActionStep(action),)), 0)
    intent = scheduler.issue(0, 0, "meadow", 0)
    assert intent is not None
    return scheduler, intent


def execute(runtime, decision, *, accepted=True, reason="accepted", timestamp_ms=None):
    intent = decision.intent
    assert intent is not None
    now_ms = intent.created_at_ms if timestamp_ms is None else timestamp_ms
    dispatched = runtime.scheduler.dispatch(intent, now_ms)
    assert dispatched.accepted
    return runtime.record_execution(intent.intent_id, accepted, reason, now_ms)


def tick(runtime, sequence, **changes):
    value = observation(sequence, **changes)
    return runtime.decide(belief(value), value.timestamp_ms)


def test_config_is_complete_versioned_and_detached_on_roundtrip():
    config = TaskRuntimeConfig(history_limit=16, minimum_detection_confidence=0.7)
    document = config.to_dict()
    assert TaskRuntimeConfig.from_dict(json.loads(json.dumps(document))) == config
    document["history_limit"] = 999
    assert config.history_limit == 16
    with pytest.raises(FrozenInstanceError):
        config.intent_ttl_ms = 12


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"intent_ttl_ms": True},
        {"intent_ttl_ms": 0},
        {"intent_ttl_ms": 1001, "plan_timeout_ms": 1000},
        {"effect_timeout_ms": 60_001},
        {"max_unknown_steps": -1},
        {"history_limit": 7},
        {"clear_confirmations": 0},
        {"max_recoveries": 101},
        {"minimum_detection_confidence": True},
        {"minimum_detection_confidence": float("nan")},
        {"minimum_detection_confidence": 1.1},
        {"undeclared": 1},
    ],
)
def test_config_rejects_unsafe_or_ambiguous_settings(changes):
    with pytest.raises(ValueError):
        TaskRuntimeConfig.from_dict(TaskRuntimeConfig().to_dict() | changes)


def test_config_requires_every_declared_setting():
    document = TaskRuntimeConfig().to_dict()
    document.pop("effect_timeout_ms")
    with pytest.raises(ValueError):
        TaskRuntimeConfig.from_dict(document)


def test_task_lifecycle_is_immutable_and_terminal_states_cannot_resume():
    task = TaskInstance("task-combat", "combat", TaskStatus.PENDING, 0, 0)
    running = task.transition(TaskStatus.RUNNING, 10, "selected")
    paused = running.transition(TaskStatus.PAUSED, 20, "survival")
    resumed = paused.transition(TaskStatus.RUNNING, 30, "safe")
    assert task.status is TaskStatus.PENDING and resumed.activations == 2
    done = resumed.transition(TaskStatus.COMPLETED, 40, "visible_phase_condition_met")
    assert done.to_dict()["status"] == "completed"
    with pytest.raises(ValueError, match="terminal"):
        done.transition(TaskStatus.RUNNING, 50, "invalid")
    with pytest.raises(ValueError):
        running.transition(TaskStatus.PENDING, 20, "invalid")
    with pytest.raises(ValueError):
        running.transition(TaskStatus.PAUSED, 0, "rollback")


def test_arbiter_priority_preempts_active_task_and_ties_are_deterministic():
    arbiter = TaskArbiter()
    combat_a = TaskCandidate("combat-a", "combat", 400, "target")
    combat_b = TaskCandidate("combat-b", "combat", 400, "target")
    survival = TaskCandidate("survival", "survival", 1000, "hp")
    assert arbiter.select((combat_b, combat_a)) == combat_a
    assert arbiter.select((combat_a, combat_b), "combat-b") == combat_b
    assert arbiter.select((combat_a, survival), "combat-a") == survival
    assert arbiter.select(()) is None
    with pytest.raises(ValueError):
        arbiter.select((combat_a, combat_a))


def test_intent_and_plan_serialization_are_strict_and_immutable():
    scheduler, intent = issued(SimAction("jump_attack"))
    assert ActionIntent.from_dict(intent.to_dict()) == intent
    assert ActionPlan.from_dict(scheduler.plan.to_dict()) == scheduler.plan
    document = intent.to_dict()
    document["action"]["kind"] = "idle"
    assert intent.action.kind == "jump_attack"
    with pytest.raises(ValueError, match="locks"):
        ActionIntent.from_dict(document)
    with pytest.raises(ValueError):
        ActionIntent.from_dict(intent.to_dict() | {"hidden_state": {}})
    with pytest.raises(ValueError):
        ActionIntent.from_dict(intent.to_dict() | {"source_sequence": True})
    with pytest.raises(ValueError):
        ActionPlan.from_dict(scheduler.plan.to_dict() | {"steps": []})
    with pytest.raises(ValueError):
        replace(intent, deadline_ms=intent.created_at_ms)
    with pytest.raises(ValueError):
        replace(scheduler.plan, steps=(ActionStep(SimAction(), 1000),))


@pytest.mark.parametrize(
    "changes",
    [
        {"intent_id": "other-intent"},
        {"plan_id": "other-plan"},
        {"task_id": "other-task"},
        {"generation": 1},
        {"source_sequence": 1},
        {"source_map": "cavern"},
        {"deadline_ms": 251},
        {"action": SimAction("attack"), "locks": ("combat",)},
    ],
)
def test_dispatch_rejects_tampered_context_and_never_returns_an_active_action(changes):
    scheduler, intent = issued()
    rejected = scheduler.dispatch(replace(intent, **changes), 0)
    assert rejected.accepted is False and rejected.action == SimAction()
    assert scheduler.dispatch(intent, 0).accepted


def test_only_one_authorized_execution_can_be_in_flight_and_denial_consumes_it():
    scheduler, intent = issued()
    assert scheduler.dispatch(intent, 0).accepted
    assert not scheduler.dispatch(intent, 0).accepted
    with pytest.raises(ValueError, match="receipt"):
        scheduler.issue(1, 50, "meadow", 50)
    with pytest.raises(ValueError):
        scheduler.complete(ExecutionReceipt("different", True, "move", 50))
    scheduler.complete(ExecutionReceipt(intent.intent_id, False, "blocked", 50))
    assert scheduler.plan is scheduler.pending is scheduler.in_flight is None
    assert not scheduler.dispatch(intent, 50).accepted
    with pytest.raises(ValueError):
        scheduler.complete(ExecutionReceipt(intent.intent_id, True, "move", 50))


@pytest.mark.parametrize("operation", ["expire", "cancel"])
def test_expired_and_cancelled_intents_are_inert(operation):
    scheduler, intent = issued()
    now_ms = intent.deadline_ms if operation == "expire" else 0
    if operation == "cancel":
        scheduler.cancel("paused", now_ms)
    rejected = scheduler.dispatch(intent, now_ms)
    assert not rejected.accepted and rejected.action == SimAction()
    assert scheduler.plan is scheduler.pending is scheduler.in_flight is None


def test_compound_sequence_requires_receipts_fresh_context_and_earliest_start():
    scheduler = ActionScheduler()
    scheduler.start(
        plan(steps=(ActionStep(SimAction("move", -1)), ActionStep(SimAction("jump_attack"), 50))),
        0,
    )
    first = scheduler.issue(0, 0, "meadow", 0)
    assert first is not None and first.step_index == 0
    assert scheduler.dispatch(first, 0).accepted
    scheduler.complete(ExecutionReceipt(first.intent_id, True, "move", 0))
    assert scheduler.issue(1, 25, "meadow", 25) is None
    assert scheduler.last_reason == "waiting_for_plan_step"
    second = scheduler.issue(2, 50, "meadow", 50)
    assert second is not None and second.step_index == 1
    assert second.action == SimAction("jump_attack")
    assert second.source_sequence == 2 and second.locks == ("combat", "movement")
    scheduler.cancel("survival", 50)
    assert scheduler.dispatch(second, 50).accepted is False
    assert scheduler.plan is None


@pytest.mark.parametrize(
    ("sequence", "capture", "map_id", "now_ms"),
    [(0, 50, "meadow", 50), (1, 0, "cavern", 50), (1, 100, "meadow", 50)],
)
def test_continuations_reject_replayed_wrong_map_and_future_context(
    sequence, capture, map_id, now_ms
):
    scheduler = ActionScheduler()
    scheduler.start(plan(steps=(ActionStep(SimAction()), ActionStep(SimAction("attack")))), 0)
    first = scheduler.issue(0, 0, "meadow", 0)
    assert first is not None and scheduler.dispatch(first, 0).accepted
    scheduler.complete(ExecutionReceipt(first.intent_id, True, "idle", 0))
    assert scheduler.issue(sequence, capture, map_id, now_ms) is None
    assert scheduler.plan is None


def test_scheduler_finite_plan_timeout_and_receipt_clock_validation():
    scheduler, intent = issued()
    assert scheduler.dispatch(intent, 10).accepted
    for timestamp in (9, 751):
        with pytest.raises(ValueError):
            scheduler.complete(ExecutionReceipt(intent.intent_id, True, "move", timestamp))
    scheduler.complete(ExecutionReceipt(intent.intent_id, True, "move", 10))
    next_plan = plan(scheduler, created_at_ms=50, source_sequence=1, deadline_ms=100)
    scheduler.start(next_plan, 50)
    assert scheduler.issue(2, 100, "meadow", 100) is None
    assert scheduler.last_reason == "plan_timeout"
    with pytest.raises(ValueError, match="rollback"):
        scheduler.cancel("rollback", 99)


@pytest.mark.parametrize("action", [SimAction("attack"), SimAction("loot"), SimAction("buff")])
def test_actuator_acceptance_and_object_disappearance_do_not_confirm_effect(action):
    _, intent = issued(action)
    before = observation(0)
    after = observation(1, monsters=())
    outcome = verify_action_effect(
        intent, ExecutionReceipt(intent.intent_id, True, "accepted", 0), before, after, 50, 500
    )
    assert outcome.status is EffectStatus.UNKNOWN


@pytest.mark.parametrize(
    ("action", "after_changes", "expected"),
    [
        (SimAction("move", 1), {"x": 104}, EffectStatus.SUCCESS),
        (SimAction("potion_hp"), {"hp": 1.0}, EffectStatus.SUCCESS),
        (SimAction("potion_mp"), {"mp": 0.5}, EffectStatus.SUCCESS),
        (SimAction("heal"), {"hp": 1.0}, EffectStatus.SUCCESS),
        (SimAction("interact"), {"dialogue_visible": True}, EffectStatus.SUCCESS),
        (SimAction("portal"), {"map_id": "cavern"}, EffectStatus.SUCCESS),
        (SimAction("move", 1), {"map_id": "cavern"}, EffectStatus.UNKNOWN),
    ],
)
def test_positive_effects_are_specific_visible_conditions(action, after_changes, expected):
    _, intent = issued(action)
    result = verify_action_effect(
        intent,
        ExecutionReceipt(intent.intent_id, True, "accepted", 0),
        observation(0),
        observation(1, **after_changes),
        50,
        500,
    )
    assert result.status is expected


def test_jump_attack_motion_is_only_likely_and_does_not_confirm_a_hit():
    _, intent = issued(SimAction("jump_attack"))
    before = observation()
    after = replace(
        observation(1),
        detections=(Detection("player", (100, 135, 114, 157), 1.0), *before.detections[1:]),
    )
    result = verify_action_effect(
        intent, ExecutionReceipt(intent.intent_id, True, "jump_attack", 0), before, after, 50, 500
    )
    assert result.status is EffectStatus.LIKELY_SUCCESS
    assert result.reason == "jump_observed_attack_unverified"


@pytest.mark.parametrize(
    "after",
    [
        observation(1, x=110, status=Status.UNKNOWN),
        observation(1, x=110, status=Status.STALE),
        observation(1, x=110, confidence=0.1),
        observation(0, x=110),
        observation(1, x=110, timestamp_ms=501),
        observation(11, x=110),
    ],
)
def test_unknown_duplicate_future_or_late_images_never_certify_motion(after):
    _, intent = issued()
    receipt = ExecutionReceipt(intent.intent_id, True, "move", 0)
    result = verify_action_effect(intent, receipt, observation(), after, 500, 500)
    assert result.status is EffectStatus.TIMEOUT


def test_effect_identity_rejection_and_nonacceptance_are_distinct_from_timeout():
    _, intent = issued()
    before = observation()
    accepted = ExecutionReceipt(intent.intent_id, True, "move", 0)
    assert verify_action_effect(intent, accepted, before, None, 499, 500) is None
    result = verify_action_effect(intent, accepted, before, None, 500, 500)
    assert result.status is EffectStatus.TIMEOUT
    result = verify_action_effect(
        intent, replace(accepted, accepted=False, reason="wall"), before, None, 0, 500
    )
    assert result.status is EffectStatus.FAILED
    with pytest.raises(ValueError, match="identity"):
        verify_action_effect(intent, accepted, replace(before, map_id="cavern"), None, 0, 500)
    with pytest.raises(ValueError, match="clock"):
        verify_action_effect(intent, replace(accepted, timestamp_ms=50), before, None, 0, 500)


def test_survival_preempts_compound_combat_and_invalidates_remaining_steps():
    runtime = TaskRuntime()
    first = tick(runtime, 0)
    assert first.action == SimAction("move", -1)
    initial_plan = runtime.scheduler.plan
    assert len(initial_plan.steps) == 2
    execute(runtime, first)
    emergency = tick(runtime, 1, hp=0.2)
    assert emergency.task == "survival" and emergency.action == SimAction("potion_hp")
    assert emergency.intent.generation > first.intent.generation
    assert runtime.scheduler.plan.plan_id != initial_plan.plan_id
    tasks = runtime.snapshot()["tasks"]
    combat = next(task for task in tasks if task["kind"] == "combat")
    assert combat["status"] == "paused" and combat["reason"] == "preempted_by_survival"
    assert not runtime.scheduler.dispatch(first.intent, 50).accepted
    execute(runtime, emergency)


def test_unavailable_belief_pauses_and_invalidates_delayed_intents():
    runtime = TaskRuntime()
    first = tick(runtime, 0)
    unknown = tick(runtime, 1, status=Status.UNKNOWN, confidence=0)
    assert unknown.state == "paused" and unknown.intent is None
    assert unknown.action == SimAction()
    assert runtime.scheduler.plan is None
    assert runtime.scheduler.dispatch(first.intent, 50).accepted is False
    resumed = tick(runtime, 2)
    assert resumed.state == "running" and resumed.intent is not None
    assert resumed.intent.generation > first.intent.generation


def test_pause_discards_unresolved_effects_without_later_success():
    runtime = TaskRuntime()
    first = tick(runtime, 0)
    execute(runtime, first)
    paused = runtime.interrupt("operator_pause", 50)
    effects = paused.evidence["verification"]
    assert len(effects) == 1 and effects[0]["status"] == "UNKNOWN"
    assert runtime.snapshot()["pending_verification"] == []
    resumed = tick(runtime, 2, x=20)
    assert resumed.evidence["verification"] == []


def test_missing_receipt_is_a_failure_and_a_denied_action_is_never_an_effect_success():
    missing = TaskRuntime()
    tick(missing, 0)
    failed = tick(missing, 1)
    assert failed.state == "failed" and failed.reason == "missing_execution_receipt"
    runtime = TaskRuntime(config=TaskRuntimeConfig(max_denied_actions=1))
    first = tick(runtime, 0)
    execute(runtime, first, accepted=False, reason="collision")
    failed = tick(runtime, 1)
    assert failed.state == "failed" and failed.reason == "execution_rejection_budget_exhausted"
    assert failed.evidence["verification"][0]["status"] == "FAILED"


def test_unknown_budget_and_task_timeout_are_bounded_and_terminal():
    runtime = TaskRuntime(config=TaskRuntimeConfig(max_unknown_steps=2))
    assert tick(runtime, 0, status=Status.UNKNOWN).state == "paused"
    assert tick(runtime, 1, status=Status.UNKNOWN).state == "stopped"
    assert tick(runtime, 2).intent is None
    runtime = TaskRuntime(config=TaskRuntimeConfig(max_task_steps=1))
    execute(runtime, tick(runtime, 0))
    failed = tick(runtime, 1)
    assert failed.state == "failed" and failed.reason == "task_timeout"
    assert (
        next(t for t in runtime.snapshot()["tasks"] if t["kind"] == "combat")["status"] == "timeout"
    )


def test_frozen_movement_exhausts_recovery_budget_without_hanging():
    runtime = TaskRuntime(config=TaskRuntimeConfig(stuck_steps=2, max_recoveries=1))
    actions = []
    for sequence in range(10):
        decision = tick(runtime, sequence, x=10, monsters=(250,))
        if decision.state == "failed":
            break
        actions.append(decision.action.kind)
        execute(runtime, decision, reason="idle")
    assert decision.state == "failed" and decision.reason == "recovery_budget_exhausted"
    assert runtime.recoveries == 1 and actions.count("jump") == 1


def test_low_confidence_targets_cannot_generate_actions_or_imply_scene_clear():
    runtime = TaskRuntime(config=TaskRuntimeConfig(max_task_steps=3))
    for sequence in range(3):
        base = observation(sequence)
        low_target = replace(base.detections[-1], confidence=0.1)
        value = replace(base, detections=(*base.detections[:-1], low_target))
        decision = runtime.decide(belief(value), sequence * 50)
        assert decision.task == "idle" and decision.action == SimAction()
        assert runtime.phase == "fight"
        execute(runtime, decision, reason="idle")
    assert tick(runtime, 3).state == "failed"


def test_runtime_context_reset_replans_compound_from_a_new_epoch():
    runtime = TaskRuntime()
    first = tick(runtime, 0)
    execute(runtime, first)
    changed = replace(belief(observation(1)), epoch=2)
    following = runtime.decide(changed, 50)
    assert following.intent.plan_id != first.intent.plan_id
    assert following.intent.step_index == 0
    assert any(
        event["event"] == "perception_context_changed" for event in following.evidence["lifecycle"]
    )
    assert following.evidence["verification"][0]["status"] == "UNKNOWN"


def test_unknown_frames_break_clear_scene_confirmation_continuity():
    runtime = TaskRuntime(config=TaskRuntimeConfig(clear_confirmations=2))
    execute(runtime, tick(runtime, 0, monsters=()))
    assert tick(runtime, 1, status=Status.UNKNOWN).state == "paused"
    execute(runtime, tick(runtime, 2, monsters=()))
    assert runtime.phase == "fight"
    execute(runtime, tick(runtime, 3, monsters=()))
    assert runtime.phase == "collect"


def test_interruption_and_snapshot_are_inert_detached_and_terminal_tasks_are_closed():
    runtime = TaskRuntime()
    decision = tick(runtime, 0)
    before = runtime.snapshot()
    assert runtime.snapshot() == before
    before["scheduler"]["pending"]["action"]["kind"] = "idle"
    assert runtime.scheduler.pending == decision.intent
    before["recent_lifecycle"][0]["event"] = "external_mutation"
    assert runtime.snapshot()["recent_lifecycle"][0]["event"] != "external_mutation"
    evidence = decision.evidence
    evidence.clear()
    assert decision.evidence
    stopped = runtime.interrupt("step_budget", 0, terminal=True)
    assert stopped.state == "stopped" and stopped.intent is None
    assert all(task["status"] == "cancelled" for task in runtime.snapshot()["tasks"])
    assert tick(runtime, 1).state == "stopped"
    with pytest.raises(ValueError):
        RuntimeDecision(SimAction("attack"), None, "combat", "running", "invalid")


def test_runtime_requires_typed_beliefs_and_declared_job_route_identity():
    with pytest.raises(ValueError):
        TaskRuntime().decide(observation(), 0)
    for maps in ((), ("meadow", "meadow"), ("MEADOW",)):
        with pytest.raises(ValueError):
            TaskRuntime(maps=maps)
    with pytest.raises(ValueError):
        TaskRuntime("other")
    with pytest.raises(ValueError):
        TaskRuntime("assassin", class_spec=CLASS_SPECS["cleric"])


def test_memory_and_evidence_are_bounded_and_deterministic():
    def run():
        runtime = TaskRuntime(config=TaskRuntimeConfig(history_limit=8))
        documents = []
        for sequence in range(25):
            decision = tick(runtime, sequence, monsters=())
            assert decision.state == "running"
            execute(runtime, decision, reason="idle")
            documents.append(decision.to_dict())
        snapshot = runtime.snapshot()
        assert len(snapshot["recent_lifecycle"]) <= 8
        assert len(snapshot["pending_verification"]) <= 8
        assert sum(task["status"] == "running" for task in snapshot["tasks"]) == 1
        return documents, snapshot

    assert run() == run()


@pytest.mark.parametrize("job", ["assassin", "dragon_knight", "cleric"])
@pytest.mark.parametrize(
    "external_route", [False, True], ids=["two_builtin_maps", "three_authored_maps"]
)
def test_real_missions_use_tasks_and_independently_verify_every_route_goal(
    job, external_route, monkeypatch
):
    def reject_legacy(*args, **kwargs):
        raise AssertionError("structured runtime must not delegate policy to VisualAgent")

    monkeypatch.setattr(VisualAgent, "decide", reject_legacy)
    pack = (
        load_map_pack(Path(__file__).resolve().parents[1] / "examples/maps/three-map-fixture.json")
        if external_route
        else None
    )
    summary = run_episode(
        config=RunConfig(seed=7, job=job, max_steps=1500),
        map_pack=pack,
        runtime=RuntimeConfig(),
    )
    number = 3 if external_route else 2
    assert summary["status"] == "completed", summary
    assert summary["independent_goal_check"] and summary["alive"]
    assert summary["events"]["monster_defeated"] == summary["drops_collected"] == 3 * number
    assert summary["events"]["dialogue"] == number
    assert summary["events"]["map_changed"] == number - 1
    assert summary["dialogue_maps"] == sorted(pack.map_ids if pack else ("meadow", "cavern"))
    terminal = summary["runtime_evidence"]["terminal"]
    assert terminal["decision"]["state"] == "completed"
    tasks = terminal["state"]["tasks"]
    assert all(task["status"] in {"completed", "cancelled"} for task in tasks)
    assert len(tasks) <= number * 8
    if job == "assassin":
        assert summary["events"]["jump"] >= number
