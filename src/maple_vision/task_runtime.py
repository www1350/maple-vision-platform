"""Versioned, bounded task planning and single-use simulator intent execution.

This module consumes visual beliefs and static class knowledge only. It never reads
frames, simulator snapshots, hidden events, inventory or monster health. Execution
acceptance and visual effect evidence are deliberately separate contracts.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from math import ceil
from typing import Any

from .class_package import (
    CapabilityClassSpec,
    CompiledCapabilities,
    CompiledProgram,
    CompiledSkill,
    compile_class_package,
)
from .contracts import Detection, Status, confidence_value
from .mission_program import MissionProgram
from .sim_types import CLASS_SPECS, ClassSpec, SimAction
from .state_estimation import BeliefState
from .vision import VisualObservation

_MAX_INTEGER = 2**53 - 1
_TASK_KINDS = frozenset(
    {"survival", "recovery", "buff", "combat", "loot", "interaction", "transition", "idle"}
)
_LOCKS = {
    "idle": ("clock",),
    "move": ("movement",),
    "jump": ("movement",),
    "climb": ("movement",),
    "attack": ("combat",),
    "jump_attack": ("combat", "movement"),
    "potion_hp": ("resource",),
    "potion_mp": ("resource",),
    "buff": ("resource", "skill"),
    "heal": ("resource", "skill"),
    "loot": ("interaction",),
    "interact": ("interaction",),
    "portal": ("interaction", "movement"),
    "cast": ("combat", "resource", "skill"),
}


def _integer(value: object, name: str, low: int = 0, high: int = _MAX_INTEGER) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _text(value: object, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 240:
        raise ValueError(f"{name} must be bounded nonempty text")
    if identifier and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}", value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _map(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value) is None:
        raise ValueError("invalid source map")
    return value


def _schema(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("unsupported task runtime schema")


def _object(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {label} fields")
    return value


def _document(value: object) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )
    return result


def _action(value: object) -> SimAction:
    data = _object(value, {"kind", "direction", "skill", "target"}, "action")
    return SimAction(**data)


def _center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class TaskRuntimeConfig:
    schema_version: int = 1
    intent_ttl_ms: int = 250
    plan_timeout_ms: int = 2000
    effect_timeout_ms: int = 500
    max_observation_age_ms: int = 250
    max_task_steps: int = 1000
    clear_confirmations: int = 4
    max_unknown_steps: int = 12
    max_denied_actions: int = 6
    max_attack_attempts: int = 12
    stuck_steps: int = 8
    max_recoveries: int = 2
    history_limit: int = 128
    minimum_detection_confidence: float = 0.5

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("intent_ttl_ms", "plan_timeout_ms", "effect_timeout_ms"):
            _integer(getattr(self, name), name, 1, 60_000)
        _integer(self.max_observation_age_ms, "max_observation_age_ms", 1, 60_000)
        for name in ("max_task_steps", "max_unknown_steps", "max_denied_actions"):
            _integer(getattr(self, name), name, 1, 100_000)
        for name in ("clear_confirmations", "max_attack_attempts", "stuck_steps"):
            _integer(getattr(self, name), name, 1, 1000)
        _integer(self.max_recoveries, "max_recoveries", 0, 100)
        _integer(self.history_limit, "history_limit", 8, 1024)
        confidence_value(self.minimum_detection_confidence)
        if self.intent_ttl_ms > self.plan_timeout_ms:
            raise ValueError("intent lifetime exceeds plan lifetime")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> TaskRuntimeConfig:
        data = _object(value, {field.name for field in fields(cls)}, "task runtime config")
        return cls(**data)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class EffectStatus(StrEnum):
    SUCCESS = "SUCCESS"
    LIKELY_SUCCESS = "LIKELY_SUCCESS"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class TaskCandidate:
    task_id: str
    kind: str
    priority: int
    reason: str
    target: Detection | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text(self.task_id, "task_id", identifier=True)
        if not isinstance(self.kind, str) or self.kind not in _TASK_KINDS:
            raise ValueError("invalid task kind")
        _integer(self.priority, "priority", 0, 1000)
        _text(self.reason, "reason")
        if self.target is not None and not isinstance(self.target, Detection):
            raise ValueError("candidate target must be a Detection")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    kind: str
    status: TaskStatus
    created_at_ms: int
    updated_at_ms: int
    activations: int = 0
    reason: str = "created"
    schema_version: int = 1

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text(self.task_id, "task_id", identifier=True)
        if (
            not isinstance(self.kind, str)
            or self.kind not in _TASK_KINDS
            or not isinstance(self.status, TaskStatus)
        ):
            raise ValueError("invalid task instance state")
        _integer(self.created_at_ms, "created_at_ms")
        _integer(self.updated_at_ms, "updated_at_ms", self.created_at_ms)
        _integer(self.activations, "activations", 0, 100_000)
        _text(self.reason, "reason")

    def transition(self, status: TaskStatus, now_ms: int, reason: str) -> TaskInstance:
        _integer(now_ms, "now_ms", self.updated_at_ms)
        if not isinstance(status, TaskStatus):
            raise ValueError("invalid task transition")
        terminal = {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
        }
        if self.status in terminal:
            raise ValueError("terminal task cannot be reactivated")
        if status is TaskStatus.PENDING:
            raise ValueError("task cannot return to pending")
        return replace(
            self,
            status=status,
            updated_at_ms=now_ms,
            reason=reason,
            activations=self.activations
            + int(status is TaskStatus.RUNNING and self.status is not TaskStatus.RUNNING),
        )

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))


class TaskArbiter:
    """Choose priority, then retain an equally eligible task, then stable ID order."""

    def select(
        self, candidates: tuple[TaskCandidate, ...], active_task_id: str | None = None
    ) -> TaskCandidate | None:
        if (
            not isinstance(candidates, tuple)
            or len(candidates) > 32
            or any(not isinstance(item, TaskCandidate) for item in candidates)
            or len({item.task_id for item in candidates}) != len(candidates)
        ):
            raise ValueError("invalid task candidates")
        if active_task_id is not None:
            _text(active_task_id, "active_task_id", identifier=True)
        return min(
            candidates,
            key=lambda item: (-item.priority, item.task_id != active_task_id, item.task_id),
            default=None,
        )


@dataclass(frozen=True)
class ActionStep:
    action: SimAction
    not_before_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.action, SimAction):
            raise ValueError("plan step requires a SimAction")
        _integer(self.not_before_ms, "not_before_ms", 0, 60_000)

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> ActionStep:
        data = _object(value, {"action", "not_before_ms"}, "action step")
        return cls(_action(data["action"]), data["not_before_ms"])


@dataclass(frozen=True)
class ActionPlan:
    plan_id: str
    task_id: str
    generation: int
    source_sequence: int
    source_timestamp_ms: int
    source_map: str
    created_at_ms: int
    deadline_ms: int
    steps: tuple[ActionStep, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("plan_id", "task_id"):
            _text(getattr(self, name), name, identifier=True)
        for name in ("generation", "source_sequence", "source_timestamp_ms", "created_at_ms"):
            _integer(getattr(self, name), name)
        _map(self.source_map)
        _integer(self.deadline_ms, "deadline_ms", self.created_at_ms + 1)
        if self.source_timestamp_ms > self.created_at_ms:
            raise ValueError("plan source is from the future")
        if (
            not isinstance(self.steps, tuple)
            or not 1 <= len(self.steps) <= 8
            or any(not isinstance(step, ActionStep) for step in self.steps)
            or tuple(step.not_before_ms for step in self.steps)
            != tuple(sorted(step.not_before_ms for step in self.steps))
            or self.steps[-1].not_before_ms >= self.deadline_ms - self.created_at_ms
            or self.deadline_ms - self.created_at_ms > 60_000
        ):
            raise ValueError("invalid bounded action plan")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> ActionPlan:
        data = dict(_object(value, {field.name for field in fields(cls)}, "action plan"))
        if not isinstance(data["steps"], list):
            raise ValueError("plan steps must be a JSON list")
        data["steps"] = tuple(ActionStep.from_dict(step) for step in data["steps"])
        return cls(**data)


@dataclass(frozen=True)
class ActionIntent:
    intent_id: str
    plan_id: str
    task_id: str
    generation: int
    step_index: int
    source_sequence: int
    source_timestamp_ms: int
    source_map: str
    created_at_ms: int
    deadline_ms: int
    action: SimAction
    locks: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("intent_id", "plan_id", "task_id"):
            _text(getattr(self, name), name, identifier=True)
        for name in ("generation", "source_sequence", "source_timestamp_ms", "created_at_ms"):
            _integer(getattr(self, name), name)
        _integer(self.step_index, "step_index", 0, 7)
        _map(self.source_map)
        _integer(self.deadline_ms, "deadline_ms", self.created_at_ms + 1)
        if (
            self.source_timestamp_ms > self.created_at_ms
            or self.deadline_ms - self.created_at_ms > 60_000
            or not isinstance(self.action, SimAction)
            or self.locks != _LOCKS[self.action.kind]
        ):
            raise ValueError("invalid intent timing, action or resource locks")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> ActionIntent:
        data = dict(_object(value, {field.name for field in fields(cls)}, "action intent"))
        data["action"] = _action(data["action"])
        if not isinstance(data["locks"], list):
            raise ValueError("intent locks must be a JSON list")
        data["locks"] = tuple(data["locks"])
        return cls(**data)


@dataclass(frozen=True)
class DispatchResult:
    intent_id: str
    accepted: bool
    reason: str
    action: SimAction
    timestamp_ms: int

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id", identifier=True)
        if type(self.accepted) is not bool or not isinstance(self.action, SimAction):
            raise ValueError("invalid dispatch result")
        _integer(self.timestamp_ms, "timestamp_ms")
        _text(self.reason, "reason")
        if not self.accepted and self.action != SimAction():
            raise ValueError("rejected dispatch must be inert")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))


@dataclass(frozen=True)
class ExecutionReceipt:
    intent_id: str
    accepted: bool
    reason: str
    timestamp_ms: int

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id", identifier=True)
        if type(self.accepted) is not bool:
            raise ValueError("execution accepted must be boolean")
        _text(self.reason, "reason")
        _integer(self.timestamp_ms, "timestamp_ms")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))


class ActionScheduler:
    """One plan, one registered intent, one in-flight execution, bounded state.

    ``dispatch`` authorizes but does not call an actuator. Its returned action is
    the only action an adapter may execute. A rejected dispatch is always idle.
    No image, detection, observation or simulator object enters this boundary.
    """

    def __init__(self, config: TaskRuntimeConfig | None = None) -> None:
        self.config = config or TaskRuntimeConfig()
        if not isinstance(self.config, TaskRuntimeConfig):
            raise ValueError("invalid scheduler config")
        self.generation = 0
        self.plan: ActionPlan | None = None
        self.pending: ActionIntent | None = None
        self.in_flight: ActionIntent | None = None
        self.step_index = 0
        self.last_reason = "idle"
        self._dispatch_ms = 0
        self._last_now_ms = 0
        self._last_source_sequence = -1
        self._last_source_timestamp_ms = -1

    def _time(self, now_ms: int) -> None:
        _integer(now_ms, "now_ms")
        if now_ms < self._last_now_ms:
            raise ValueError("scheduler clock rollback")
        self._last_now_ms = now_ms

    def start(self, plan: ActionPlan, now_ms: int) -> None:
        self._time(now_ms)
        if not isinstance(plan, ActionPlan):
            raise ValueError("scheduler requires an ActionPlan")
        if self.plan is not None or self.pending is not None or self.in_flight is not None:
            raise ValueError("scheduler already owns a plan")
        if plan.generation != self.generation or plan.created_at_ms != now_ms:
            raise ValueError("plan generation or creation time mismatch")
        self.plan, self.step_index, self.last_reason = plan, 0, "plan_started"

    def cancel(self, reason: str, now_ms: int) -> None:
        self._time(now_ms)
        _text(reason, "reason")
        self.plan = self.pending = self.in_flight = None
        self.step_index = 0
        self.generation = _integer(self.generation + 1, "generation")
        self.last_reason = reason

    def issue(
        self, source_sequence: int, source_timestamp_ms: int, source_map: str, now_ms: int
    ) -> ActionIntent | None:
        self._time(now_ms)
        _integer(source_sequence, "source_sequence")
        _integer(source_timestamp_ms, "source_timestamp_ms")
        _map(source_map)
        if self.pending is not None or self.in_flight is not None:
            raise ValueError("previous intent has no execution receipt")
        plan = self.plan
        if plan is None:
            self.last_reason = "no_plan"
            return None
        if now_ms >= plan.deadline_ms:
            self.cancel("plan_timeout", now_ms)
            return None
        if source_map != plan.source_map or source_sequence < plan.source_sequence:
            self.cancel("plan_context_changed", now_ms)
            return None
        if (
            source_sequence <= self._last_source_sequence
            or source_timestamp_ms < self._last_source_timestamp_ms
        ):
            self.cancel("source_not_monotonic", now_ms)
            return None
        deadline = min(
            plan.deadline_ms,
            now_ms + self.config.intent_ttl_ms,
            source_timestamp_ms + self.config.max_observation_age_ms,
        )
        if source_timestamp_ms > now_ms or deadline <= now_ms:
            self.cancel("source_not_fresh", now_ms)
            return None
        step = plan.steps[self.step_index]
        if now_ms < plan.created_at_ms + step.not_before_ms:
            self.last_reason = "waiting_for_plan_step"
            return None
        intent = ActionIntent(
            f"intent-{self.generation}-{self.step_index}-{source_sequence}",
            plan.plan_id,
            plan.task_id,
            self.generation,
            self.step_index,
            source_sequence,
            source_timestamp_ms,
            source_map,
            now_ms,
            deadline,
            step.action,
            _LOCKS[step.action.kind],
        )
        self.pending, self.last_reason = intent, "intent_registered"
        self._last_source_sequence = source_sequence
        self._last_source_timestamp_ms = source_timestamp_ms
        return intent

    def dispatch(self, intent: ActionIntent, now_ms: int) -> DispatchResult:
        self._time(now_ms)
        if not isinstance(intent, ActionIntent):
            raise ValueError("dispatch requires a validated ActionIntent")
        reason = "authorized"
        if self.in_flight is not None or self.pending != intent:
            reason = "intent_not_registered_or_already_dispatched"
        elif intent.generation != self.generation:
            reason = "intent_cancelled"
        elif now_ms < intent.created_at_ms:
            reason = "intent_from_future"
        elif now_ms >= intent.deadline_ms:
            self.cancel("intent_expired", now_ms)
            reason = "intent_expired"
        if reason != "authorized":
            return DispatchResult(intent.intent_id, False, reason, SimAction(), now_ms)
        self.pending, self.in_flight = None, intent
        self._dispatch_ms = now_ms
        self.last_reason = "intent_dispatched"
        return DispatchResult(intent.intent_id, True, reason, intent.action, now_ms)

    def complete(self, receipt: ExecutionReceipt) -> ActionIntent:
        if not isinstance(receipt, ExecutionReceipt):
            raise ValueError("scheduler requires an execution receipt")
        intent = self.in_flight
        if intent is None or receipt.intent_id != intent.intent_id:
            raise ValueError("execution receipt does not match an in-flight intent")
        if (
            receipt.timestamp_ms < self._dispatch_ms
            or receipt.timestamp_ms > intent.deadline_ms + self.config.effect_timeout_ms
        ):
            raise ValueError("execution receipt time is outside its bounded lifetime")
        self._time(receipt.timestamp_ms)
        self.in_flight = None
        if not receipt.accepted:
            self.cancel("execution_rejected", receipt.timestamp_ms)
        else:
            assert self.plan is not None
            self.step_index += 1
            if self.step_index == len(self.plan.steps):
                self.cancel("plan_dispatched", receipt.timestamp_ms)
            else:
                self.last_reason = "step_dispatched"
        return intent

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "pending": None if self.pending is None else self.pending.to_dict(),
            "in_flight": None if self.in_flight is None else self.in_flight.to_dict(),
            "step_index": self.step_index,
            "last_reason": self.last_reason,
            "last_source_sequence": self._last_source_sequence,
            "last_source_timestamp_ms": self._last_source_timestamp_ms,
        }


@dataclass(frozen=True)
class EffectVerification:
    intent_id: str
    status: EffectStatus
    reason: str
    source_sequence: int
    observed_sequence: int | None
    timestamp_ms: int

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id", identifier=True)
        if not isinstance(self.status, EffectStatus):
            raise ValueError("invalid effect status")
        _text(self.reason, "reason")
        _integer(self.source_sequence, "source_sequence")
        if self.observed_sequence is not None:
            _integer(self.observed_sequence, "observed_sequence")
        _integer(self.timestamp_ms, "timestamp_ms")

    def to_dict(self) -> dict[str, Any]:
        return _document(asdict(self))


def verify_action_effect(
    intent: ActionIntent,
    receipt: ExecutionReceipt,
    before: VisualObservation,
    after: VisualObservation | None,
    now_ms: int,
    timeout_ms: int,
    *,
    capability_kind: str | None = None,
) -> EffectVerification | None:
    """Return a visible operational effect, uncertainty, or an elapsed timeout.

    Attack, loot and buff lack sufficient visible state in this fixture. Neither
    an accepted actuator call nor a disappearing object establishes their effect.
    A successful motion/resource/dialogue/map observation is narrowly scoped to
    that measured condition; it is not an independent mission-goal assertion.
    """
    if not isinstance(intent, ActionIntent) or not isinstance(receipt, ExecutionReceipt):
        raise ValueError("invalid effect inputs")
    if not isinstance(before, VisualObservation) or (
        after is not None and not isinstance(after, VisualObservation)
    ):
        raise ValueError("effect verifier consumes observations only")
    if (
        receipt.intent_id != intent.intent_id
        or before.sequence != intent.source_sequence
        or before.timestamp_ms != intent.source_timestamp_ms
        or before.map_id != intent.source_map
    ):
        raise ValueError("effect input identity mismatch")
    _integer(now_ms, "now_ms")
    _integer(timeout_ms, "timeout_ms", 1, 60_000)
    if capability_kind is not None and (
        not isinstance(capability_kind, str) or capability_kind not in {"damage", "heal", "buff"}
    ):
        raise ValueError("invalid capability effect kind")
    if receipt.timestamp_ms < intent.created_at_ms or now_ms < receipt.timestamp_ms:
        raise ValueError("effect input clock mismatch")

    def result(status: EffectStatus, reason: str) -> EffectVerification:
        return EffectVerification(
            intent.intent_id,
            status,
            reason,
            intent.source_sequence,
            None if after is None else after.sequence,
            now_ms,
        )

    if not receipt.accepted:
        return result(EffectStatus.FAILED, f"execution_rejected:{receipt.reason}")
    if after is not None and after.source_id != before.source_id:
        return result(EffectStatus.UNKNOWN, "perception_source_changed")
    usable = (
        after is not None
        and before.status is Status.KNOWN
        and after.status is Status.KNOWN
        and before.confidence >= 0.5
        and after.confidence >= 0.5
        and after.sequence > before.sequence
        and before.timestamp_ms <= after.timestamp_ms <= now_ms
        and after.timestamp_ms >= receipt.timestamp_ms
        and after.timestamp_ms < receipt.timestamp_ms + timeout_ms
        and before.map_id is not None
        and after.map_id is not None
    )
    if usable:
        assert after is not None
        kind = intent.action.kind
        if after.map_id != before.map_id:
            return result(
                EffectStatus.SUCCESS if kind == "portal" else EffectStatus.UNKNOWN,
                "observed_map_change" if kind == "portal" else "verification_context_changed",
            )
        if kind == "interact" and not before.dialogue_visible and after.dialogue_visible:
            return result(EffectStatus.SUCCESS, "observed_dialogue_open")
        if (kind in {"potion_hp", "heal"} or (kind == "cast" and capability_kind == "heal")) and (
            before.hp_ratio is not None
            and after.hp_ratio is not None
            and after.hp_ratio > before.hp_ratio
        ):
            return result(EffectStatus.SUCCESS, "observed_hp_increase")
        if kind == "potion_mp" and (
            before.mp_ratio is not None
            and after.mp_ratio is not None
            and after.mp_ratio > before.mp_ratio
        ):
            return result(EffectStatus.SUCCESS, "observed_mp_increase")
        first = [d for d in before.detections if d.class_id == "player"]
        second = [d for d in after.detections if d.class_id == "player"]
        if len(first) == len(second) == 1 and min(first[0].confidence, second[0].confidence) >= 0.5:
            dx = _center(second[0])[0] - _center(first[0])[0]
            dy = _center(second[0])[1] - _center(first[0])[1]
            if kind == "move" and dx * intent.action.direction > 0.5:
                return result(EffectStatus.SUCCESS, "observed_directional_movement")
            if kind == "climb" and dy * intent.action.direction > 0.5:
                return result(EffectStatus.LIKELY_SUCCESS, "observed_vertical_movement")
            if kind in {"jump", "jump_attack"} and dy < -0.5:
                return result(
                    EffectStatus.SUCCESS if kind == "jump" else EffectStatus.LIKELY_SUCCESS,
                    "observed_jump" if kind == "jump" else "jump_observed_attack_unverified",
                )
        if kind in {"attack", "loot", "buff", "idle", "cast"}:
            return result(EffectStatus.UNKNOWN, "effect_not_observable_from_fixture_state")
    if now_ms >= receipt.timestamp_ms + timeout_ms:
        return result(EffectStatus.TIMEOUT, "effect_deadline_elapsed")
    return None


@dataclass(frozen=True)
class RuntimeDecision:
    action: SimAction
    intent: ActionIntent | None
    task: str
    state: str
    reason: str
    _evidence_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.action, SimAction):
            raise ValueError("invalid runtime action")
        if self.intent is not None and (
            not isinstance(self.intent, ActionIntent) or self.intent.action != self.action
        ):
            raise ValueError("runtime action and intent disagree")
        if self.intent is None and self.action != SimAction():
            raise ValueError("runtime action requires an intent")
        if not isinstance(self.state, str) or self.state not in {
            "running",
            "paused",
            "stopped",
            "failed",
            "completed",
        }:
            raise ValueError("invalid runtime decision state")
        if self.state != "running" and self.intent is not None:
            raise ValueError("nonrunning runtime cannot issue an intent")
        _text(self.task, "task")
        _text(self.reason, "reason")
        if not isinstance(self._evidence_json, str) or len(self._evidence_json) > 2_000_000:
            raise ValueError("invalid runtime evidence")
        evidence = json.loads(self._evidence_json)
        if not isinstance(evidence, dict):
            raise ValueError("runtime evidence must be an object")
        json.dumps(evidence, allow_nan=False)

    @property
    def evidence(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self._evidence_json)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "action": _document(asdict(self.action)),
            "intent": None if self.intent is None else self.intent.to_dict(),
            "task": self.task,
            "state": self.state,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class _PlanningContext:
    facing: int
    jumped_map: bool
    last_attack_ms: int
    descent_direction: int
    recovery_direction: int


def _grounded(
    observation: VisualObservation, player: Detection, minimum_confidence: float = 0.5
) -> bool:
    px = _center(player)[0]
    return any(
        item.class_id == "platform"
        and item.confidence >= minimum_confidence
        and item.bbox[0] <= px <= item.bbox[2]
        and abs(player.bbox[3] - item.bbox[1]) <= 4
        for item in observation.detections
    )


class _CapabilityAttempts:
    """Local attempts and conservative visual resource bounds, never inventory truth."""

    def __init__(self, capabilities: CompiledCapabilities) -> None:
        self.capabilities = capabilities
        self.ready = {group: 0 for group in capabilities.cooldown_groups}
        self.attempts: dict[str, dict[str, Any]] = {}
        self.buff_attempt_until: dict[str, int] = {}
        self.hp = self.mp = 0
        self.ammo_upper_budget = capabilities.package.resources.initial_ammo
        self._measured_hp = self._measured_mp = 0
        self._capture_ms = -1
        self._spent_hp = self._spent_mp = 0
        self._last_resource_attempt_ms = -1

    def observe(self, observation: VisualObservation) -> None:
        assert observation.hp_ratio is not None and observation.mp_ratio is not None
        spec = self.capabilities.class_spec
        # Fixture HUD is rounded to a 100-pixel bar. Half a pixel is an
        # uncertainty interval, not a fractional unit available to spend.
        hp = max(0, ceil((observation.hp_ratio - 0.005) * spec.max_hp - 1e-9))
        mp = max(0, ceil((observation.mp_ratio - 0.005) * spec.max_mp - 1e-9))
        if observation.timestamp_ms > self._capture_ms:
            self._measured_hp, self._measured_mp = hp, mp
        else:
            self._measured_hp = min(self._measured_hp, hp)
            self._measured_mp = min(self._measured_mp, mp)
        self._capture_ms = observation.timestamp_ms
        if observation.timestamp_ms > self._last_resource_attempt_ms:
            self._spent_hp = self._spent_mp = 0
        self._refresh_bounds()

    def _refresh_bounds(self) -> None:
        self.hp = max(0, self._measured_hp - self._spent_hp)
        self.mp = max(0, self._measured_mp - self._spent_mp)

    def program_reason(self, program: CompiledProgram, now_ms: int, grounded: bool) -> str | None:
        if program.posture == "ground" and not grounded:
            return "capability_requires_ground"
        if program.posture == "air" and grounded:
            return "capability_requires_air"
        if any(now_ms < self.ready[group] for group, _ in program.cooldown_ready_offsets):
            return "capability_cooldown"
        if self.hp <= program.hp_cost:
            return "capability_hp_budget"
        if self.mp < program.mp_cost:
            return "capability_mp_budget"
        if self.ammo_upper_budget < program.ammo_cost:
            return "capability_ammo_budget"
        if any(now_ms < self.buff_attempt_until.get(skill, 0) for skill in program.skill_ids):
            return "capability_buff_attempt_window"
        return None

    def skill_reason(self, skill: CompiledSkill, now_ms: int, grounded: bool) -> str | None:
        if now_ms < self.ready[skill.cooldown_group]:
            return "capability_cooldown"
        if skill.posture == "ground" and not grounded:
            return "capability_requires_ground"
        if skill.posture == "air" and grounded:
            return "capability_requires_air"
        if self.hp <= skill.hp_cost:
            return "capability_hp_budget"
        if self.mp < skill.mp_cost:
            return "capability_mp_budget"
        if self.ammo_upper_budget < skill.ammo_cost:
            return "capability_ammo_budget"
        return None

    def attempted(self, skill: CompiledSkill, receipt: ExecutionReceipt) -> None:
        # A receipt exists only after registered dispatch. Delayed receipt time
        # yields a later, conservative readiness estimate; static program offsets
        # are never used as evidence that an unexecuted step spent resources.
        self.ready[skill.cooldown_group] = max(
            self.ready[skill.cooldown_group], receipt.timestamp_ms + skill.cooldown_ms
        )
        self._last_resource_attempt_ms = receipt.timestamp_ms
        self._spent_hp += skill.hp_cost
        self._spent_mp += skill.mp_cost
        self.ammo_upper_budget = max(0, self.ammo_upper_budget - skill.ammo_cost)
        self._refresh_bounds()
        previous = self.attempts.get(skill.id)
        self.attempts[skill.id] = {
            "count": 1 if previous is None else previous["count"] + 1,
            "timestamp_ms": receipt.timestamp_ms,
            "accepted": receipt.accepted,
            "reason": receipt.reason,
        }
        if skill.kind == "buff":
            self.buff_attempt_until[skill.id] = receipt.timestamp_ms + skill.duration_ms

    def to_dict(self) -> dict[str, Any]:
        return _document(
            {
                "class_id": self.capabilities.class_id,
                "package_sha256": self.capabilities.package_sha256,
                "cooldown_ready_ms": self.ready,
                "skill_attempts": self.attempts,
                "buff_attempt_until_ms": self.buff_attempt_until,
                "resource_estimate": {
                    "hp_lower_bound": self.hp,
                    "mp_lower_bound": self.mp,
                    "ammo_upper_budget": self.ammo_upper_budget,
                    "observation_timestamp_ms": self._capture_ms,
                    "last_resource_attempt_ms": self._last_resource_attempt_ms,
                    "unobserved_hp_spend": self._spent_hp,
                    "unobserved_mp_spend": self._spent_mp,
                },
            }
        )


class ActionPlanner:
    """Turn a selected visible task goal into a bounded immutable action plan."""

    def __init__(self, config: TaskRuntimeConfig) -> None:
        if not isinstance(config, TaskRuntimeConfig):
            raise ValueError("invalid planner config")
        self.config = config
        self.last_reason = "idle"
        self.descent_direction = 0
        self.last_program_id: str | None = None

    def plan(
        self,
        candidate: TaskCandidate,
        observation: VisualObservation,
        *,
        spec: ClassSpec | CapabilityClassSpec,
        context: _PlanningContext,
        generation: int,
        plan_id: str,
        now_ms: int,
        capabilities: CompiledCapabilities | None = None,
        capability_state: _CapabilityAttempts | None = None,
    ) -> ActionPlan:
        if not isinstance(candidate, TaskCandidate) or not isinstance(
            spec, (ClassSpec, CapabilityClassSpec)
        ):
            raise ValueError("invalid planner inputs")
        if capabilities is not None and capability_state is None:
            raise ValueError("capability planner requires attempt state")
        if not isinstance(observation, VisualObservation) or observation.map_id is None:
            raise ValueError("planner requires a current map observation")
        players = [item for item in observation.detections if item.class_id == "player"]
        if len(players) != 1 or observation.status is not Status.KNOWN:
            raise ValueError("planner requires an unambiguous current player")
        player = players[0]
        px, py = _center(player)
        target = candidate.target
        self.descent_direction = context.descent_direction
        self.last_program_id = None
        steps: tuple[ActionStep, ...]

        def single(action: SimAction, reason: str) -> tuple[ActionStep, ...]:
            self.last_reason = reason
            return (ActionStep(action),)

        def compiled(program: CompiledProgram, direction: int) -> tuple[ActionStep, ...]:
            self.last_program_id = program.id
            self.last_reason = f"capability_program:{program.id}"
            return tuple(
                ActionStep(step.action, step.not_before_ms)
                for step in program.instantiate(direction)
            )

        grounded = _grounded(observation, player, self.config.minimum_detection_confidence)
        program: CompiledProgram | None = None
        if capabilities is not None and capability_state is not None:
            if candidate.kind == "survival" and candidate.reason == "heal":
                program = capabilities.program_for("heal")
            elif candidate.kind == "buff":
                program = capabilities.program_for("buff")
            elif candidate.kind == "combat":
                program = capabilities.program_for("combat")
                airborne, area = (
                    capabilities.program_for("airborne"),
                    capabilities.program_for("area"),
                )
                if (
                    airborne is not None
                    and not context.jumped_map
                    and capability_state.program_reason(airborne, now_ms, grounded) is None
                ):
                    program = airborne
                if (
                    area is not None
                    and capability_state.program_reason(area, now_ms, grounded) is None
                ):
                    nearby = sum(
                        item.class_id == "monster"
                        and item.confidence >= self.config.minimum_detection_confidence
                        and abs(_center(item)[0] - px) <= area.range_px
                        and abs(_center(item)[1] - py) <= 38
                        for item in observation.detections
                    )
                    if nearby >= 2:
                        program = area

        if candidate.kind == "survival":
            if program is not None:
                steps = compiled(program, context.facing)
            else:
                kind = {"heal": "heal", "hp": "potion_hp", "mp": "potion_mp"}[candidate.reason]
                steps = single(SimAction(kind), f"restore_{candidate.reason}")
        elif candidate.kind == "recovery":
            steps = single(SimAction("jump", context.recovery_direction), "bounded_stuck_recovery")
        elif candidate.kind == "buff":
            steps = (
                compiled(program, context.facing)
                if program is not None
                else single(SimAction("buff"), "maintain_fixture_buff")
            )
        elif candidate.kind == "combat" and target is not None:
            tx, ty = _center(target)
            direction = 1 if tx >= px else -1
            reach = (
                program.range_px * 0.9
                if program is not None
                else (65 if spec.id == "dragon_knight" else spec.attack_range * 0.9)
            )
            if context.descent_direction and player.bbox[3] < target.bbox[3] - 2:
                steps = single(
                    SimAction("move", context.descent_direction), "finish_platform_descent"
                )
            elif abs(ty - py) > 38:
                self.descent_direction = 0
                supports = [
                    d
                    for d in observation.detections
                    if d.class_id == "platform"
                    and d.confidence >= self.config.minimum_detection_confidence
                    and abs(player.bbox[3] - d.bbox[1]) <= 2
                    and d.bbox[0] < player.bbox[2]
                    and d.bbox[2] > player.bbox[0]
                ]
                if supports and player.bbox[3] < target.bbox[3] - 2:
                    surfaces = [
                        d
                        for d in observation.detections
                        if d.class_id == "platform"
                        and d.confidence >= self.config.minimum_detection_confidence
                        and abs(d.bbox[1] - supports[0].bbox[1]) < 2
                    ]
                    left, right = min(d.bbox[0] for d in surfaces), max(d.bbox[2] for d in surfaces)
                    self.descent_direction = -1 if px - left < right - px else 1
                    steps = single(
                        SimAction("move", self.descent_direction), "leave_elevated_platform"
                    )
                else:
                    steps = single(SimAction(), "wait_for_jump_descent")
            elif abs(tx - px) > reach:
                self.descent_direction = 0
                steps = single(SimAction("move", direction), "approach_visible_monster")
            elif program is not None and capability_state is not None:
                self.descent_direction = 0
                unavailable = capability_state.program_reason(program, now_ms, grounded)
                steps = (
                    single(SimAction(), unavailable)
                    if unavailable
                    else compiled(program, direction)
                )
            elif now_ms - context.last_attack_ms < spec.attack_cooldown_ms:
                self.descent_direction = 0
                steps = single(SimAction(), "attack_cooldown")
            elif (
                spec.id == "assassin"
                and spec.can_jump_attack
                and not context.jumped_map
                and _grounded(observation, player, self.config.minimum_detection_confidence)
            ):
                self.descent_direction = 0
                if context.facing != direction and self.config.plan_timeout_ms > 50:
                    self.last_reason = "face_then_jump_attack"
                    steps = (
                        ActionStep(SimAction("move", direction)),
                        ActionStep(SimAction("jump_attack"), 50),
                    )
                else:
                    steps = single(SimAction("jump_attack"), "fixture_jump_attack")
            else:
                self.descent_direction = 0
                skill = "area" if spec.id == "dragon_knight" else "basic"
                steps = single(SimAction("attack", direction, skill), "attack_visible_monster")
        elif candidate.kind in {"loot", "interaction", "transition"} and target is not None:
            tx = _center(target)[0]
            threshold = {"loot": 20, "interaction": 25, "transition": 22}[candidate.kind]
            if abs(tx - px) > threshold:
                steps = single(
                    SimAction("move", 1 if tx > px else -1), f"approach_visible_{candidate.kind}"
                )
            else:
                action = {"loot": "loot", "interaction": "interact", "transition": "portal"}[
                    candidate.kind
                ]
                steps = single(SimAction(action), f"execute_{candidate.kind}")
        else:
            steps = single(SimAction(), candidate.reason)
        return ActionPlan(
            plan_id,
            candidate.task_id,
            generation,
            observation.sequence,
            observation.timestamp_ms,
            observation.map_id,
            now_ms,
            now_ms + self.config.plan_timeout_ms,
            steps,
        )


@dataclass(frozen=True)
class _PendingEffect:
    intent: ActionIntent
    receipt: ExecutionReceipt
    before: VisualObservation


class TaskRuntime:
    """Bounded task arbitration for the declared three-job visual mission.

    Task candidates, planning, scheduling, receipts and verification are separate
    executable stages. Optional programs order fight/collect/talk objectives
    per map. Transitions follow the bound route.
    """

    def __init__(
        self,
        job: str = "assassin",
        maps: tuple[str, ...] = ("meadow", "cavern"),
        *,
        class_spec: ClassSpec | None = None,
        config: TaskRuntimeConfig | None = None,
        capabilities: CompiledCapabilities | None = None,
        mission: MissionProgram | None = None,
    ) -> None:
        if capabilities is not None and (
            type(capabilities) is not CompiledCapabilities
            or json.dumps(
                compile_class_package(capabilities.package).to_dict(),
                sort_keys=True,
                allow_nan=False,
            )
            != json.dumps(capabilities.to_dict(), sort_keys=True, allow_nan=False)
        ):
            raise ValueError("invalid or altered compiled capabilities")
        if not isinstance(job, str) or (
            job not in CLASS_SPECS if capabilities is None else job != capabilities.class_id
        ):
            raise ValueError("unknown task runtime fixture job")
        if capabilities is not None and class_spec is not None:
            raise ValueError("capabilities and legacy class profile are mutually exclusive")
        if not isinstance(maps, tuple) or not 1 <= len(maps) <= 8:
            raise ValueError("task route must contain 1..8 map IDs")
        for map_id in maps:
            _map(map_id)
        if len(set(maps)) != len(maps):
            raise ValueError("task route IDs must be distinct")
        if class_spec is not None and (
            not isinstance(class_spec, ClassSpec) or class_spec.id != job
        ):
            raise ValueError("task runtime class profile disagrees with job")
        self.capabilities = capabilities
        self.spec: ClassSpec | CapabilityClassSpec = (
            capabilities.class_spec
            if capabilities is not None
            else (CLASS_SPECS[job] if class_spec is None else class_spec)
        )
        self.config = TaskRuntimeConfig() if config is None else config
        if not isinstance(self.config, TaskRuntimeConfig):
            raise ValueError("invalid task runtime config")
        if capabilities is not None and any(
            program.duration_ms >= self.config.plan_timeout_ms
            for program in capabilities.program_entries
        ):
            raise ValueError("capability program exceeds configured plan time budget")
        self._capability_state = None if capabilities is None else _CapabilityAttempts(capabilities)
        self.mission = mission
        if mission is not None:
            if type(mission) is not MissionProgram:
                raise ValueError("invalid mission program")
            mission.bind_route(maps)
        self.maps = maps if mission is None else mission.maps
        first_phase = "fight" if mission is None else mission.stages[0].phases[0]
        self.phase, self.state, self.reason = first_phase, "running", "created"
        self.map_index = 0
        self.recoveries = 0
        self.scheduler = ActionScheduler(self.config)
        self.arbiter = TaskArbiter()
        self.planner = ActionPlanner(self.config)
        self._tasks: dict[str, TaskInstance] = {}
        self._active_task_id: str | None = None
        self._plan_counter = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=self.config.history_limit)
        self._tick_events: list[dict[str, Any]] = []
        self._tick_effects: list[dict[str, Any]] = []
        self._effects: deque[_PendingEffect] = deque()
        self._issued_before: VisualObservation | None = None
        self._last_receipt: ExecutionReceipt | None = None
        self._last_now_ms = 0
        self._last_sequence = -1
        self._last_capture_ms = -1
        self._last_source_id: str | None = None
        self._last_belief_epoch: int | None = None
        self._unknown_steps = 0
        self._phase_steps = 0
        self._empty_frames = 0
        self._last_attack_ms = -100_000
        self._last_potion_ms = -100_000
        self._last_skill_ms = -100_000
        self._buff_ms = -100_000
        self._jumped_maps: set[str] = set()
        self._facing = 1
        self._descent_direction = 0
        self._last_position: tuple[float, float] | None = None
        self._last_action = SimAction()
        self._stuck_steps = 0
        self._last_monster_count: int | None = None
        self._attack_attempts = 0
        self._denied_actions = 0
        self._last_decision: RuntimeDecision | None = None

    def _event(self, kind: str, now_ms: int, **details: object) -> None:
        event = {"event": kind, "timestamp_ms": now_ms, **details}
        self._tick_events.append(event)
        self._history.append(event)

    def _decision(
        self,
        reason: str,
        *,
        candidates: tuple[TaskCandidate, ...] = (),
        selected: TaskCandidate | None = None,
        intent: ActionIntent | None = None,
    ) -> RuntimeDecision:
        self.reason = reason
        evidence = {
            "schema_version": 1,
            "phase": self.phase,
            "map_index": self.map_index,
            "candidates": [item.to_dict() for item in candidates],
            "selected": None if selected is None else selected.to_dict(),
            "task_instance": None
            if self._active_task_id is None
            else self._tasks[self._active_task_id].to_dict(),
            "scheduler": self.scheduler.snapshot(),
            "verification": self._tick_effects,
            "lifecycle": self._tick_events,
            "previous_receipt": None
            if self._last_receipt is None
            else self._last_receipt.to_dict(),
        }
        if self._capability_state is not None:
            evidence["capabilities"] = self._capability_state.to_dict()
        result = RuntimeDecision(
            SimAction() if intent is None else intent.action,
            intent,
            self.phase if selected is None else selected.kind,
            self.state,
            reason,
            json.dumps(evidence, sort_keys=True, ensure_ascii=True, allow_nan=False),
        )
        self._last_decision = result
        return result

    def _finish(self, state: str, reason: str, now_ms: int) -> RuntimeDecision:
        self.state = state
        self.scheduler.cancel(reason, now_ms)
        self._issued_before = None
        for task_id, instance in tuple(self._tasks.items()):
            if instance.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED}:
                status = TaskStatus.CANCELLED
                if task_id == self._active_task_id:
                    status = TaskStatus.COMPLETED if state == "completed" else TaskStatus.FAILED
                    if "timeout" in reason:
                        status = TaskStatus.TIMEOUT
                self._tasks[task_id] = instance.transition(status, now_ms, reason)
        self._event("runtime_terminal", now_ms, state=state, reason=reason)
        return self._decision(reason)

    def interrupt(self, reason: str, now_ms: int, *, terminal: bool = False) -> RuntimeDecision:
        _text(reason, "reason")
        _integer(now_ms, "now_ms", self._last_now_ms)
        if type(terminal) is not bool:
            raise ValueError("terminal must be boolean")
        self._tick_events, self._tick_effects = [], []
        self._last_now_ms = now_ms
        if self.state in {"completed", "failed", "stopped"}:
            return self._decision(self.reason)
        if self.scheduler.in_flight is not None:
            self._event("unacknowledged_intent_cancelled", now_ms)
        self.scheduler.cancel(reason, now_ms)
        self._issued_before = None
        self._discard_effects("verification_interrupted", now_ms)
        self._last_action, self._last_position = SimAction(), None
        self._empty_frames = 0
        if terminal:
            for task_id, instance in tuple(self._tasks.items()):
                if instance.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED}:
                    self._tasks[task_id] = instance.transition(TaskStatus.CANCELLED, now_ms, reason)
        if self._active_task_id is not None:
            instance = self._tasks[self._active_task_id]
            if instance.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED}:
                status = TaskStatus.CANCELLED if terminal else TaskStatus.PAUSED
                self._tasks[instance.task_id] = instance.transition(status, now_ms, reason)
        self.state = "stopped" if terminal else "paused"
        self._event("interrupted", now_ms, reason=reason, terminal=terminal)
        return self._decision(reason)

    def _advance(self, phase: str, now_ms: int) -> None:
        previous = self.phase
        role = {"fight": "combat", "collect": "loot", "talk": "interaction", "travel": "transition"}
        task_id = f"task-{self.map_index}-{role.get(previous, 'idle')}"
        instance = self._tasks.get(task_id)
        if instance is not None and instance.status in {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSED,
        }:
            self._tasks[task_id] = instance.transition(
                TaskStatus.COMPLETED, now_ms, "visible_phase_condition_met"
            )
        if self._active_task_id is not None and self._active_task_id != task_id:
            active = self._tasks[self._active_task_id]
            if active.status is TaskStatus.RUNNING:
                self._tasks[active.task_id] = active.transition(
                    TaskStatus.PAUSED, now_ms, "mission_phase_changed"
                )
        self.scheduler.cancel("mission_phase_changed", now_ms)
        self._active_task_id = None
        self.phase, self._phase_steps, self._empty_frames = phase, 0, 0
        self._event("mission_phase_changed", now_ms, previous=previous, following=phase)

    def _next_objective(self, fallback: str, now_ms: int) -> None:
        phase = fallback
        if self.mission is not None:
            phases = self.mission.stages[self.map_index].phases
            index = phases.index(self.phase) + 1
            if index < len(phases):
                phase = phases[index]
            else:
                phase = "done" if self.map_index + 1 == len(self.maps) else "travel"
        self._advance(phase, now_ms)

    def _evaluate_effects(self, observation: VisualObservation, now_ms: int) -> None:
        retained: deque[_PendingEffect] = deque()
        while self._effects:
            pending = self._effects.popleft()
            verification = verify_action_effect(
                pending.intent,
                pending.receipt,
                pending.before,
                observation,
                now_ms,
                self.config.effect_timeout_ms,
                capability_kind=(
                    self.capabilities.skills[pending.intent.action.skill].kind
                    if self.capabilities is not None and pending.intent.action.kind == "cast"
                    else None
                ),
            )
            if verification is None:
                retained.append(pending)
            else:
                self._tick_effects.append(verification.to_dict())
        self._effects = retained

    def _discard_effects(
        self, reason: str, now_ms: int, observed_sequence: int | None = None
    ) -> None:
        """An interrupted context cannot subsequently certify its old effects."""
        while self._effects:
            pending = self._effects.popleft()
            verification = EffectVerification(
                pending.intent.intent_id,
                EffectStatus.UNKNOWN if pending.receipt.accepted else EffectStatus.FAILED,
                reason if pending.receipt.accepted else "execution_rejected",
                pending.intent.source_sequence,
                observed_sequence,
                now_ms,
            )
            self._tick_effects.append(verification.to_dict())

    def _candidates(self, observation: VisualObservation, now_ms: int) -> tuple[TaskCandidate, ...]:
        candidates: list[TaskCandidate] = []
        player = next(item for item in observation.detections if item.class_id == "player")
        px = _center(player)[0]

        def add(kind: str, priority: int, reason: str, target: Detection | None = None) -> None:
            task_id = f"task-{self.map_index}-{kind}"
            instance = self._tasks.get(task_id)
            if instance is None or instance.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
            }:
                candidates.append(TaskCandidate(task_id, kind, priority, reason, target))

        hp, mp = observation.hp_ratio, observation.mp_ratio
        assert hp is not None and mp is not None
        if self.capabilities is not None and self._capability_state is not None:
            attempts = self._capability_state
            grounded = _grounded(observation, player, self.config.minimum_detection_confidence)
            combat = self.capabilities.program_for("combat")
            healing = self.capabilities.program_for("heal")
            assert combat is not None
            need_hp = hp < 0.4 or attempts.hp <= combat.hp_cost
            need_mp = (self.spec.max_mp > 0 and mp < 0.25) or attempts.mp < combat.mp_cost
            if (
                (hp < 0.5 or need_hp)
                and healing is not None
                and attempts.program_reason(healing, now_ms, grounded) is None
            ):
                add("survival", 1000, "heal")
            elif now_ms - self._last_potion_ms >= 500:
                if need_hp:
                    add("survival", 1000, "hp")
                elif need_mp:
                    add("survival", 1000, "mp")
        elif (
            hp < 0.5 and self.spec.heal_amount and mp > 0.2 and now_ms - self._last_skill_ms >= 500
        ):
            add("survival", 1000, "heal")
        elif now_ms - self._last_potion_ms >= 500:
            if hp < 0.4:
                add("survival", 1000, "hp")
            elif mp < 0.25:
                add("survival", 1000, "mp")
        if self._stuck_steps >= self.config.stuck_steps:
            add("recovery", 900, "stuck")
        monsters = tuple(
            d
            for d in observation.detections
            if d.class_id == "monster" and d.confidence >= self.config.minimum_detection_confidence
        )
        if self.phase == "fight" and monsters:
            if self.capabilities is not None and self._capability_state is not None:
                buff = self.capabilities.program_for("buff")
                if (
                    buff is not None
                    and self._capability_state.program_reason(
                        buff,
                        now_ms,
                        _grounded(observation, player, self.config.minimum_detection_confidence),
                    )
                    is None
                ):
                    add("buff", 500, "capability_buff_attempt_window_elapsed")
            elif (
                now_ms - self._buff_ms >= 5000 and now_ms - self._last_skill_ms >= 500 and mp > 0.3
            ):
                add("buff", 500, "buff_expired_by_local_attempt_clock")
            add(
                "combat",
                400,
                "visible_monster",
                min(monsters, key=lambda d: abs(_center(d)[0] - px)),
            )
        elif self.phase == "collect":
            drops = tuple(
                d
                for d in observation.detections
                if d.class_id == "loot" and d.confidence >= self.config.minimum_detection_confidence
            )
            if drops:
                add("loot", 300, "visible_loot", min(drops, key=lambda d: abs(_center(d)[0] - px)))
        elif self.phase in {"talk", "travel"}:
            role = "npc" if self.phase == "talk" else "portal"
            targets = tuple(
                d
                for d in observation.detections
                if d.class_id == role and d.confidence >= self.config.minimum_detection_confidence
            )
            if targets:
                add(
                    "interaction" if self.phase == "talk" else "transition",
                    200 if self.phase == "talk" else 100,
                    f"visible_{role}",
                    min(targets, key=lambda d: abs(_center(d)[0] - px)),
                )
        add("idle", 0, "confirming_scene_or_waiting_for_target")
        return tuple(candidates)

    def _capability_step_reason(
        self, action: SimAction, observation: VisualObservation, now_ms: int
    ) -> str | None:
        assert self.capabilities is not None and self._capability_state is not None
        skill = self.capabilities.skills.get(action.skill)
        if skill is None:
            return "capability_skill_unregistered"
        player = next(item for item in observation.detections if item.class_id == "player")
        unavailable = self._capability_state.skill_reason(
            skill,
            now_ms,
            _grounded(observation, player, self.config.minimum_detection_confidence),
        )
        if unavailable is not None:
            return unavailable
        if skill.kind == "damage":
            px, py = _center(player)
            direction = action.direction or self._facing
            targets = (
                target
                for target in observation.detections
                if target.class_id == "monster"
                and target.confidence >= self.config.minimum_detection_confidence
            )
            if not any(
                abs(_center(target)[0] - px) <= skill.range_px - 1
                and abs(_center(target)[1] - py) <= skill.vertical_range_px - 1
                and (skill.shape == "radial" or (_center(target)[0] - px) * direction >= 0)
                for target in targets
            ):
                return "capability_target_context_changed"
        return None

    def _activate(self, candidate: TaskCandidate, now_ms: int) -> None:
        if self._active_task_id != candidate.task_id:
            if self._active_task_id is not None:
                previous = self._tasks[self._active_task_id]
                if previous.status is TaskStatus.RUNNING:
                    reason = f"preempted_by_{candidate.kind}"
                    self._tasks[previous.task_id] = previous.transition(
                        TaskStatus.PAUSED, now_ms, reason
                    )
                    self._event("task_paused", now_ms, task_id=previous.task_id, reason=reason)
            self.scheduler.cancel("task_selection_changed", now_ms)
            self._active_task_id = candidate.task_id
        instance = self._tasks.get(candidate.task_id)
        if instance is None:
            instance = TaskInstance(
                candidate.task_id, candidate.kind, TaskStatus.PENDING, now_ms, now_ms
            )
            self._event("task_created", now_ms, task_id=instance.task_id, task_kind=instance.kind)
        if instance.status is not TaskStatus.RUNNING:
            instance = instance.transition(TaskStatus.RUNNING, now_ms, candidate.reason)
            self._event("task_activated", now_ms, task_id=instance.task_id)
        self._tasks[candidate.task_id] = instance

    def decide(self, belief: BeliefState, now_ms: int) -> RuntimeDecision:
        _integer(now_ms, "now_ms")
        self._tick_events, self._tick_effects = [], []
        if now_ms < self._last_now_ms:
            return self._finish("stopped", "clock_rollback", self._last_now_ms)
        self._last_now_ms = now_ms
        if self.state in {"completed", "failed", "stopped"}:
            return self._decision(self.reason)
        if not isinstance(belief, BeliefState):
            raise ValueError("task runtime requires a typed visual belief")
        observation, actionable = belief.observation, belief.actionable
        players = [item for item in observation.detections if item.class_id == "player"]
        valid = (
            actionable
            and observation.status is Status.KNOWN
            and observation.confidence >= 0.5
            and len(players) == 1
            and players[0].confidence >= self.config.minimum_detection_confidence
            and observation.hp_ratio is not None
            and observation.mp_ratio is not None
            and observation.map_id is not None
            and observation.sequence > self._last_sequence
            and self._last_capture_ms <= observation.timestamp_ms <= now_ms
            and now_ms - observation.timestamp_ms < self.config.max_observation_age_ms
        )
        if not valid:
            self._unknown_steps += 1
            return self.interrupt(
                "unavailable_or_stale_belief",
                now_ms,
                terminal=self._unknown_steps >= self.config.max_unknown_steps,
            )
        self._last_sequence, self._last_capture_ms = observation.sequence, observation.timestamp_ms
        self._unknown_steps = 0
        self.state = "running"
        if self.scheduler.pending is not None or self.scheduler.in_flight is not None:
            return self._finish("failed", "missing_execution_receipt", now_ms)
        if self._last_source_id is not None and (
            observation.source_id != self._last_source_id
            or (
                belief.epoch != self._last_belief_epoch
                and observation.map_id == self.maps[self.map_index]
            )
        ):
            self.scheduler.cancel("perception_context_changed", now_ms)
            self._discard_effects("perception_context_changed", now_ms, observation.sequence)
            self._last_action, self._last_position = SimAction(), None
            self._empty_frames, self._descent_direction = 0, 0
            self._event("perception_context_changed", now_ms)
        self._last_source_id, self._last_belief_epoch = observation.source_id, belief.epoch
        if self._capability_state is not None:
            self._capability_state.observe(observation)
        self._evaluate_effects(observation, now_ms)
        if observation.hp_ratio == 0:
            return self._finish("stopped", "player_dead", now_ms)
        if observation.map_id != self.maps[self.map_index]:
            if (
                self.phase == "travel"
                and self.map_index + 1 < len(self.maps)
                and observation.map_id == self.maps[self.map_index + 1]
            ):
                phase = (
                    "fight"
                    if self.mission is None
                    else self.mission.stages[self.map_index + 1].phases[0]
                )
                self._advance(phase, now_ms)
                self.map_index += 1
                self._last_position, self._last_action = None, SimAction()
                self._stuck_steps = self._attack_attempts = self._denied_actions = 0
                self._last_monster_count = None
                self._descent_direction = 0
            else:
                return self._finish("failed", "unexpected_map", now_ms)
        self._phase_steps += 1
        if self._phase_steps > self.config.max_task_steps:
            return self._finish("failed", "task_timeout", now_ms)
        if self._denied_actions >= self.config.max_denied_actions:
            return self._finish("failed", "execution_rejection_budget_exhausted", now_ms)
        position = _center(players[0])
        if self._last_action.kind == "move" and self._last_position is not None:
            distance = abs(position[0] - self._last_position[0]) + abs(
                position[1] - self._last_position[1]
            )
            self._stuck_steps = self._stuck_steps + 1 if distance < 0.5 else 0
        self._last_position = position
        if (
            self._stuck_steps >= self.config.stuck_steps
            and self.recoveries >= self.config.max_recoveries
        ):
            return self._finish("failed", "recovery_budget_exhausted", now_ms)
        monsters = [item for item in observation.detections if item.class_id == "monster"]
        if self.phase == "fight":
            if len(monsters) != self._last_monster_count:
                self._last_monster_count = len(monsters)
                self._attack_attempts = 0
            if self._attack_attempts >= self.config.max_attack_attempts:
                return self._finish("failed", "combat_progress_timeout", now_ms)
            self._empty_frames = self._empty_frames + 1 if not monsters else 0
            if self._empty_frames >= self.config.clear_confirmations:
                self._next_objective("collect", now_ms)
        elif self.phase == "collect":
            drops = [item for item in observation.detections if item.class_id == "loot"]
            self._empty_frames = self._empty_frames + 1 if not drops else 0
            if self._empty_frames >= self.config.clear_confirmations:
                self._next_objective("talk", now_ms)
        elif self.phase == "talk" and observation.dialogue_visible:
            self._next_objective(
                "done" if self.map_index + 1 == len(self.maps) else "travel", now_ms
            )
        if self.phase == "done":
            return self._finish("completed", "visual_mission_conditions_met", now_ms)
        candidates = self._candidates(observation, now_ms)
        selected = self.arbiter.select(candidates, self._active_task_id)
        if selected is None:
            return self._finish("failed", "no_eligible_task", now_ms)
        self._activate(selected, now_ms)
        active_plan = self.scheduler.plan
        if active_plan is not None:
            next_action = active_plan.steps[self.scheduler.step_index].action
            if self.capabilities is not None and now_ms >= active_plan.deadline_ms:
                self.scheduler.cancel("plan_timeout", now_ms)
                self._event("plan_cancelled", now_ms, reason="plan_timeout")
            elif (
                self.capabilities is not None
                and next_action.kind == "cast"
                and now_ms
                >= active_plan.created_at_ms
                + active_plan.steps[self.scheduler.step_index].not_before_ms
            ):
                unavailable = self._capability_step_reason(next_action, observation, now_ms)
                if unavailable == "capability_cooldown":
                    self._last_action = SimAction()
                    return self._decision(unavailable, candidates=candidates, selected=selected)
                if unavailable is not None:
                    self.scheduler.cancel(unavailable, now_ms)
                    self._event("plan_cancelled", now_ms, reason=unavailable)
            elif next_action.kind in {"jump", "jump_attack"} and not _grounded(
                observation, players[0], self.config.minimum_detection_confidence
            ):
                self.scheduler.cancel("compound_precondition_changed", now_ms)
                self._event("plan_cancelled", now_ms, reason="compound_precondition_changed")
        if self.scheduler.plan is None:
            if selected.kind == "recovery":
                self.recoveries += 1
                self._stuck_steps = 0
            context = _PlanningContext(
                self._facing,
                observation.map_id in self._jumped_maps,
                self._last_attack_ms,
                self._descent_direction,
                self._last_action.direction if self._last_action.direction else 1,
            )
            self._plan_counter += 1
            plan = self.planner.plan(
                selected,
                observation,
                spec=self.spec,
                context=context,
                generation=self.scheduler.generation,
                plan_id=f"plan-{self._plan_counter}",
                now_ms=now_ms,
                capabilities=self.capabilities,
                capability_state=self._capability_state,
            )
            self._descent_direction = self.planner.descent_direction
            self.scheduler.start(plan, now_ms)
            self._event("plan_created", now_ms, plan_id=plan.plan_id, task_id=plan.task_id)
        intent = self.scheduler.issue(
            observation.sequence, observation.timestamp_ms, observation.map_id, now_ms
        )
        if intent is None:
            self._last_action = SimAction()
            return self._decision(
                self.scheduler.last_reason, candidates=candidates, selected=selected
            )
        self._issued_before = observation
        self._last_action = intent.action
        if intent.action.kind in {"attack", "jump_attack"}:
            self._last_attack_ms = now_ms
            self._attack_attempts += 1
        if intent.action.kind in {"potion_hp", "potion_mp"}:
            self._last_potion_ms = now_ms
        if intent.action.kind in {"heal", "buff"}:
            self._last_skill_ms = now_ms
        if intent.action.kind == "buff":
            self._buff_ms = now_ms
        return self._decision(
            self.planner.last_reason, candidates=candidates, selected=selected, intent=intent
        )

    def record_execution(
        self, intent_id: str, accepted: bool, reason: str, timestamp_ms: int
    ) -> ExecutionReceipt:
        receipt = ExecutionReceipt(intent_id, accepted, reason, timestamp_ms)
        before = self._issued_before
        if before is None:
            raise ValueError("runtime has no source observation for this execution")
        intent = self.scheduler.complete(receipt)
        if before.sequence != intent.source_sequence:
            raise ValueError("runtime receipt source identity mismatch")
        if self.capabilities is not None and self._capability_state is not None:
            if intent.action.kind == "cast":
                skill = self.capabilities.skills[intent.action.skill]
                self._capability_state.attempted(skill, receipt)
                if skill.kind == "damage":
                    self._last_attack_ms = timestamp_ms
                    self._attack_attempts += 1
            airborne = self.capabilities.program_for("airborne")
            if (
                intent.action.kind == "jump"
                and airborne is not None
                and self.planner.last_program_id == airborne.id
            ):
                self._jumped_maps.add(intent.source_map)
        self._last_now_ms = max(self._last_now_ms, timestamp_ms)
        self._issued_before = None
        self._last_receipt = receipt
        if accepted:
            if intent.action.kind != "idle":
                self._denied_actions = 0
            if intent.action.direction:
                self._facing = intent.action.direction
            if intent.action.kind == "jump_attack":
                self._jumped_maps.add(intent.source_map)
        else:
            self._denied_actions += 1
        if len(self._effects) >= self.config.history_limit:
            discarded = self._effects.popleft()
            self._event(
                "effect_retention_budget_exhausted",
                timestamp_ms,
                intent_id=discarded.intent.intent_id,
            )
        self._effects.append(_PendingEffect(intent, receipt, before))
        return receipt

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "state": self.state,
            "phase": self.phase,
            "reason": self.reason,
            "map_index": self.map_index,
            "recoveries": self.recoveries,
            "source_id": self._last_source_id,
            "belief_epoch": self._last_belief_epoch,
            "active_task_id": self._active_task_id,
            "tasks": [self._tasks[key].to_dict() for key in sorted(self._tasks)],
            "scheduler": self.scheduler.snapshot(),
            "pending_verification": [
                {"intent": pending.intent.to_dict(), "receipt": pending.receipt.to_dict()}
                for pending in self._effects
            ],
            "last_receipt": None if self._last_receipt is None else self._last_receipt.to_dict(),
            "last_decision": None if self._last_decision is None else self._last_decision.to_dict(),
            "recent_lifecycle": [_document(event) for event in self._history],
            "counters": {
                "phase_steps": self._phase_steps,
                "unknown_steps": self._unknown_steps,
                "attack_attempts": self._attack_attempts,
                "denied_actions": self._denied_actions,
                "stuck_steps": self._stuck_steps,
                "plans_created": self._plan_counter,
            },
        }
        if self.mission is not None:
            result["mission_sha256"] = self.mission.content_sha256
        if self._capability_state is not None:
            result["capabilities"] = self._capability_state.to_dict()
        return result
