"""Independent structured-runtime evidence and execution counterexamples.

Recomputed file hashes are deliberate: integrity checks alone must not hide a
contradictory engine identity, stale intent or forged terminal state.
"""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.contracts import Detection, Status
from maple_vision.lab import RunConfig, run_episode, verify_perception, verify_run
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import SimAction
from maple_vision.state_estimation import StateConfig, TemporalStateEstimator
from maple_vision.task_runtime import (
    ActionPlan,
    ActionScheduler,
    ActionStep,
    DispatchResult,
    EffectStatus,
    ExecutionReceipt,
    TaskRuntime,
    TaskRuntimeConfig,
    verify_action_effect,
)
from maple_vision.vision import PixelPerception, VisualObservation


def _read(bundle, name):
    return json.loads((bundle / name).read_text(encoding="utf-8"))


def _rewrite(bundle, name, value):
    path = bundle / name
    if name.endswith(".jsonl"):
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in value)
    else:
        text = json.dumps(value, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    if name != "manifest.json":
        manifest = _read(bundle, "manifest.json")
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        _rewrite(bundle, "manifest.json", manifest)


def _rows(bundle):
    return [
        json.loads(line)
        for line in (bundle / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture(scope="module")
def original_structured_bundle(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("v05-independent") / "original"
    summary = run_episode(
        bundle,
        RunConfig(max_steps=4, record_stride=1),
        runtime=RuntimeConfig(),
    )
    assert summary["schema_version"] == 2
    assert summary["steps"] == 4
    assert verify_run(bundle)["verified"]
    return bundle


@pytest.fixture
def structured_bundle(original_structured_bundle, tmp_path):
    return Path(shutil.copytree(original_structured_bundle, tmp_path / "copy"))


def test_review_short_structured_run_reexecutes_without_rewriting_evidence(structured_bundle):
    before = {
        path.relative_to(structured_bundle): path.read_bytes()
        for path in structured_bundle.rglob("*")
        if path.is_file()
    }
    result = verify_perception(structured_bundle)
    assert result["verified"] and result["policy_reproduced"]
    assert result["frames_reinferred"] == 5
    assert result["terminal_decision_compared"] is False
    assert before == {
        path.relative_to(structured_bundle): path.read_bytes()
        for path in structured_bundle.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("part", ["manifest", "summary", "trace"])
def test_review_structured_markers_cannot_be_partially_downgraded(structured_bundle, part):
    name = f"{part}.jsonl" if part == "trace" else f"{part}.json"
    if part == "trace":
        value = _rows(structured_bundle)
        value[1]["schema_version"] = 1
    else:
        value = _read(structured_bundle, name)
        value["schema_version"] = 1
    _rewrite(structured_bundle, name, value)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_unlisted_deleted_config_cannot_turn_structured_run_into_legacy(
    structured_bundle,
):
    manifest = _read(structured_bundle, "manifest.json")
    manifest["files"].pop("runtime-config.json")
    _rewrite(structured_bundle, "manifest.json", manifest)
    (structured_bundle / "runtime-config.json").unlink()
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_runtime_config_is_bounded_before_any_unbounded_hash_read(
    structured_bundle, monkeypatch
):
    target = structured_bundle / "runtime-config.json"
    target.write_bytes(b" " * (64 * 1024 + 1))
    read_bytes = Path.read_bytes

    def forbid_unbounded(path):
        if path == target:
            raise AssertionError("runtime sidecar must be bounded before checksum I/O")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_unbounded)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


@pytest.mark.parametrize("bad_version", [True, 1.0], ids=["boolean", "float"])
def test_review_sidecar_schema_is_not_accepted_through_python_numeric_equality(
    structured_bundle, bad_version
):
    value = _read(structured_bundle, "runtime-config.json")
    value["schema_version"] = bad_version
    _rewrite(structured_bundle, "runtime-config.json", value)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_boolean_trace_sequence_does_not_equal_frame_one(structured_bundle):
    rows = _rows(structured_bundle)
    assert rows[1]["sequence"] == 1
    rows[1]["sequence"] = True
    _rewrite(structured_bundle, "trace.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_integer_execution_acceptance_does_not_match_boolean_receipt(structured_bundle):
    rows = _rows(structured_bundle)
    row = next(row for row in rows if row["runtime"]["receipt"] is not None)
    assert type(row["execution"]["accepted"]) is bool
    row["execution"]["accepted"] = int(row["execution"]["accepted"])
    _rewrite(structured_bundle, "trace.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_receipt_cannot_authorize_a_different_intent(structured_bundle):
    rows = _rows(structured_bundle)
    row = next(row for row in rows if row["runtime"]["receipt"] is not None)
    row["runtime"]["receipt"]["intent_id"] = "review-other-intent"
    _rewrite(structured_bundle, "trace.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


@pytest.mark.parametrize(
    ("path", "forged"),
    [
        (("runtime", "belief", "schema_version"), True),
        (("runtime", "belief", "now_ms"), 1),
        (("runtime", "belief", "actionable"), False),
        (("guard", "state"), "paused"),
        (("runtime", "decision", "intent", "deadline_ms"), 251),
    ],
    ids=["belief-schema", "belief-clock", "nonactionable", "guard-paused", "intent-lifetime"],
)
def test_review_rehashed_dispatch_cannot_contradict_its_declared_runtime_contract(
    structured_bundle, path, forged
):
    rows = _rows(structured_bundle)
    row = rows[0]
    assert row["runtime"]["decision"]["intent"] is not None
    node = row
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = forged
    _rewrite(structured_bundle, "trace.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


def test_review_expired_source_at_freshness_boundary_cannot_authorize_dispatch(tmp_path):
    bundle = tmp_path / "expired-source"
    runtime = RuntimeConfig(tasks=TaskRuntimeConfig(max_observation_age_ms=50))
    run_episode(bundle, RunConfig(max_steps=4), runtime=runtime)
    rows = _rows(bundle)
    row = rows[1]
    assert row["runtime"]["dispatch"]["timestamp_ms"] == 50
    row["observation"]["timestamp_ms"] = 0
    row["runtime"]["belief"]["observation"]["timestamp_ms"] = 0
    row["runtime"]["decision"]["intent"]["source_timestamp_ms"] = 0
    _rewrite(bundle, "trace.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(bundle)


def test_review_rehashed_sidecar_settings_still_need_all_metadata_bindings(structured_bundle):
    forged = RuntimeConfig(state=StateConfig(field_ttl_ms=2000)).manifest()
    _rewrite(structured_bundle, "runtime-config.json", forged)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


@pytest.mark.parametrize("location", ["step", "terminal"])
def test_review_reinference_checks_derived_beliefs_even_after_rehash(structured_bundle, location):
    if location == "step":
        rows = _rows(structured_bundle)
        rows[1]["runtime"]["belief"]["reason"] = "review_forged_belief"
        _rewrite(structured_bundle, "trace.jsonl", rows)
    else:
        summary = _read(structured_bundle, "summary.json")
        summary["runtime_evidence"]["terminal"]["belief"]["reason"] = "review_forged_belief"
        _rewrite(structured_bundle, "summary.json", summary)
    with pytest.raises(ValueError):
        verify_perception(structured_bundle)


def test_review_budget_exit_cannot_claim_an_evaluated_terminal_policy(structured_bundle):
    summary = _read(structured_bundle, "summary.json")
    terminal = summary["runtime_evidence"]["terminal"]
    assert terminal["policy_evaluated"] is False and terminal["decision"] is None
    terminal["policy_evaluated"] = True
    terminal["decision"] = _rows(structured_bundle)[-1]["runtime"]["decision"]
    _rewrite(structured_bundle, "summary.json", summary)
    with pytest.raises(ValueError):
        verify_run(structured_bundle)


@pytest.mark.parametrize(
    ("fault", "limit"), [("none", 4), ("duplicate_frames", 28), ("persistent_dropout", 40)]
)
def test_review_every_acquired_frame_has_one_state_update_including_pause_and_terminal(
    tmp_path, monkeypatch, fault, limit
):
    calls = []
    update = TemporalStateEstimator.update

    def observe_once(self, observation, now_ms):
        calls.append((observation.sequence, observation.timestamp_ms, now_ms))
        return update(self, observation, now_ms)

    monkeypatch.setattr(TemporalStateEstimator, "update", observe_once)
    bundle = tmp_path / fault
    summary = run_episode(bundle, RunConfig(max_steps=limit, fault=fault), runtime=RuntimeConfig())
    assert len(calls) == summary["steps"] + 1
    assert [clock for _, _, clock in calls] == list(range(0, summary["steps"] * 50 + 1, 50))
    terminal = summary["runtime_evidence"]["terminal"]
    assert terminal["belief"]["now_ms"] == summary["steps"] * 50
    if fault == "duplicate_frames":
        assert summary["paused_steps"] > 0
        assert [sequence for sequence, _, _ in calls[19:26]] == [19] * 7
        assert calls[26][0] == 26
    if fault == "persistent_dropout":
        assert summary["status"] == "stopped"
        assert summary["perception_evidence"]["terminal"]["cause"] == "supervisor"
        assert terminal["policy_evaluated"] is False
        assert terminal["decision"] is None


def test_review_lab_executes_only_the_dispatch_gate_output(tmp_path, monkeypatch):
    attempted = []

    def deny(self, intent, now_ms):
        attempted.append(intent.action.kind)
        return DispatchResult(intent.intent_id, False, "review_denied", SimAction(), now_ms)

    monkeypatch.setattr(ActionScheduler, "dispatch", deny)
    bundle = tmp_path / "denied"
    summary = run_episode(bundle, RunConfig(max_steps=4), runtime=RuntimeConfig())
    assert attempted and any(kind != "idle" for kind in attempted)
    assert summary["events"] == {}
    for row in _rows(bundle):
        assert SimAction(**row["execution"]["applied_action"]) == SimAction()
        assert row["runtime"]["receipt"] is None
    assert verify_run(bundle)["verified"]


def test_review_state_rejection_can_stop_runtime_before_any_simulator_action(tmp_path, monkeypatch):
    update = TemporalStateEstimator.update

    def reject_current_measurement(self, observation, now_ms):
        belief = update(self, observation, now_ms)
        return replace(belief, actionable=False, reason="review_state_rejection")

    monkeypatch.setattr(TemporalStateEstimator, "update", reject_current_measurement)
    bundle = tmp_path / "state-stopped"
    config = RuntimeConfig(tasks=TaskRuntimeConfig(max_unknown_steps=1))
    summary = run_episode(bundle, RunConfig(max_steps=4), runtime=config)
    assert summary["status"] == "stopped"
    assert summary["steps"] == 0 and _rows(bundle) == []
    assert summary["perception_evidence"]["terminal"]["cause"] == "policy"
    terminal = summary["runtime_evidence"]["terminal"]
    assert terminal["policy_evaluated"] is True
    assert terminal["decision"]["state"] == "stopped"
    assert terminal["interruption"] is None
    assert verify_run(bundle)["verified"]


def test_review_future_observation_cannot_poison_next_valid_pipeline_frame(tmp_path, monkeypatch):
    observe = PixelPerception.observe

    def future_frame(self, frame):
        observation = observe(self, frame)
        if frame.sequence == 1:
            return replace(observation, sequence=10_000, timestamp_ms=100_000)
        return observation

    monkeypatch.setattr(PixelPerception, "observe", future_frame)
    bundle = tmp_path / "future-frame"
    summary = run_episode(bundle, RunConfig(max_steps=4), runtime=RuntimeConfig())
    rows = _rows(bundle)
    rejected, resumed = rows[1], rows[2]
    assert rejected["observation"]["sequence"] == 10_000
    assert rejected["guard"]["state"] == "paused"
    assert rejected["runtime"]["belief"]["actionable"] is False
    assert rejected["runtime"]["belief"]["last_accepted_sequence"] == 0
    assert rejected["runtime"]["policy_evaluated"] is False
    assert rejected["runtime"]["decision"]["intent"] is None
    assert resumed["guard"]["state"] == "running"
    assert resumed["runtime"]["belief"]["last_accepted_sequence"] == 2
    assert resumed["runtime"]["belief"]["actionable"] is True
    assert resumed["runtime"]["policy_evaluated"] is True
    assert summary["runtime_evidence"]["terminal"]["belief"]["last_accepted_sequence"] == 4


def test_review_freeze_records_idle_outcome_without_certifying_requested_attacks(tmp_path):
    bundle = tmp_path / "frozen"
    summary = run_episode(
        bundle, RunConfig(max_steps=80, fault="freeze_input"), runtime=RuntimeConfig()
    )
    rows = _rows(bundle)
    frozen = [row for row in rows if row["sequence"] >= 20]
    attempts = {
        row["runtime"]["decision"]["intent"]["intent_id"]
        for row in frozen
        if row["decision"]["action"]["kind"] == "attack"
    }
    assert attempts and summary["status"] == "failed" and summary["steps"] < 80
    for row in frozen:
        assert SimAction(**row["execution"]["applied_action"]) == SimAction()
        assert row["execution"]["accepted"] is True
        assert row["execution"]["reason"] == "idle"
        assert row["runtime"]["receipt"]["applied_action"] == row["execution"]["applied_action"]
        assert row["runtime"]["receipt"]["accepted"] is True
    decisions = [row["runtime"]["decision"] for row in rows]
    decisions.append(summary["runtime_evidence"]["terminal"]["decision"])
    verifications = {
        effect["intent_id"]: effect["status"]
        for decision in decisions
        if decision is not None
        for effect in decision["evidence"]["verification"]
    }
    assert attempts <= verifications.keys()
    assert all(verifications[intent_id] == "UNKNOWN" for intent_id in attempts)
    result = verify_perception(bundle)
    assert result["verified"] and result["frames_reinferred"] == summary["steps"] + 1


def test_review_effect_cannot_be_confirmed_across_perception_source_changes():
    before = VisualObservation(
        0,
        0,
        Status.KNOWN,
        1.0,
        (Detection("player", (20, 146, 34, 168), 1.0),),
        0.7,
        0.8,
        "meadow",
        source_id="review-source-a",
    )
    after = replace(
        before,
        sequence=1,
        timestamp_ms=50,
        source_id="review-source-b",
        detections=(Detection("player", (40, 146, 54, 168), 1.0),),
    )
    scheduler = ActionScheduler()
    plan = ActionPlan(
        plan_id="review-source-plan",
        task_id="review-motion",
        generation=0,
        source_sequence=0,
        source_timestamp_ms=0,
        source_map="meadow",
        created_at_ms=0,
        deadline_ms=1000,
        steps=(ActionStep(SimAction("move", 1)),),
    )
    scheduler.start(plan, 0)
    intent = scheduler.issue(0, 0, "meadow", 0)
    assert intent is not None
    result = verify_action_effect(
        intent, ExecutionReceipt(intent.intent_id, True, "move", 0), before, after, 50, 500
    )
    assert result is None or result.status not in {
        EffectStatus.SUCCESS,
        EffectStatus.LIKELY_SUCCESS,
    }


def test_review_compound_plan_is_rebuilt_after_perception_source_changes():
    initial = VisualObservation(
        0,
        0,
        Status.KNOWN,
        1.0,
        (
            Detection("player", (100, 146, 114, 168), 1.0),
            Detection("monster", (40, 152, 50, 168), 1.0),
            Detection("platform", (0, 168, 320, 192), 1.0),
        ),
        0.7,
        0.8,
        "meadow",
        source_id="review-source-a",
    )
    estimator = TemporalStateEstimator()
    runtime = TaskRuntime()

    def decide_and_execute(observation):
        now_ms = observation.timestamp_ms
        decision = runtime.decide(estimator.update(observation, now_ms), now_ms)
        assert decision.intent is not None
        assert runtime.scheduler.dispatch(decision.intent, now_ms).accepted
        runtime.record_execution(decision.intent.intent_id, True, "review_executed", now_ms)
        return decision

    first = decide_and_execute(initial)
    assert first.action.kind == "buff"
    compound = decide_and_execute(replace(initial, sequence=1, timestamp_ms=50))
    assert compound.action == SimAction("move", -1)
    assert runtime.scheduler.plan is not None and len(runtime.scheduler.plan.steps) == 2
    changed = replace(initial, sequence=2, timestamp_ms=100, source_id="review-source-b")
    fresh = runtime.decide(estimator.update(changed, 100), 100)
    assert fresh.intent is not None and compound.intent is not None
    assert fresh.intent.plan_id != compound.intent.plan_id
    assert fresh.intent.step_index == 0


def test_review_reinference_restores_nondefault_state_and_task_settings(tmp_path):
    config = RuntimeConfig(
        state=StateConfig(field_ttl_ms=1900, track_ttl_ms=800, max_match_distance=24),
        tasks=TaskRuntimeConfig(intent_ttl_ms=100, plan_timeout_ms=800, history_limit=8),
    )
    bundle = tmp_path / "custom-runtime"
    run_episode(bundle, RunConfig(max_steps=4), runtime=config)
    result = verify_perception(bundle)
    assert result["verified"] and result["structured_runtime_reproduced"]
    assert result["runtime_settings"] == config.manifest()
    assert result["belief_frames_recomputed"] == 5


def test_review_unmanifested_runtime_file_cannot_be_injected_into_legacy_run(tmp_path):
    bundle = tmp_path / "legacy"
    summary = run_episode(bundle, RunConfig(max_steps=1, record_stride=1))
    assert summary["schema_version"] == 1
    manifest = _read(bundle, "manifest.json")
    assert manifest["schema_version"] == 1
    assert "runtime-config.json" not in manifest["files"]
    (bundle / "runtime-config.json").write_text(
        json.dumps(RuntimeConfig().manifest()), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        verify_run(bundle)


@pytest.mark.parametrize("second_sequence", [4, 5], ids=["regressed", "reused"])
def test_review_compound_plan_cannot_execute_again_from_an_old_observation(second_sequence):
    scheduler = ActionScheduler()
    plan = ActionPlan(
        plan_id="review-compound",
        task_id="combat-meadow",
        generation=0,
        source_sequence=0,
        source_timestamp_ms=0,
        source_map="meadow",
        created_at_ms=0,
        deadline_ms=1000,
        steps=(ActionStep(SimAction("move", 1)), ActionStep(SimAction("attack", 1), 50)),
    )
    scheduler.start(plan, 0)
    first = scheduler.issue(5, 50, "meadow", 50)
    assert first is not None and scheduler.dispatch(first, 50).accepted
    scheduler.complete(ExecutionReceipt(first.intent_id, True, "move", 100))
    try:
        second = scheduler.issue(second_sequence, 100, "meadow", 100)
    except ValueError:
        return
    assert second is None or scheduler.dispatch(second, 100).accepted is False
