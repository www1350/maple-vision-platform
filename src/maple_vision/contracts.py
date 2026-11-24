"""Small, immutable P0 contracts; gameplay state is deliberately not invented."""

from dataclasses import dataclass
from enum import Enum, StrEnum
from math import isfinite
from typing import Protocol

SCHEMA_VERSION = 1


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def confidence_value(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number in [0, 1]")


class Status(str, Enum):  # noqa: UP042 -- preserve existing Status.__str__ compatibility
    KNOWN = "known"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONFLICTED = "conflicted"


class Action(StrEnum):
    NOOP = "noop"


class ExecutionStatus(StrEnum):
    SKIPPED = "skipped"
    REJECTED = "rejected"


class VerificationStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp_ms: int
    pixels: bytes

    def __post_init__(self) -> None:
        nonnegative_int(self.sequence, "sequence")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.pixels, bytes):
            raise ValueError("pixels must be immutable bytes")


@dataclass(frozen=True)
class Detection:
    class_id: str
    bbox: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        nonempty(self.class_id, "class_id")
        confidence_value(self.confidence)
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise ValueError("bbox must be an immutable xyxy tuple")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in self.bbox
        ):
            raise ValueError("bbox coordinates must be finite numbers")
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have non-negative origin and positive area")


@dataclass(frozen=True)
class Observation:
    sequence: int
    timestamp_ms: int
    status: Status
    confidence: float = 0.0
    source_id: str = "noop"
    detections: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        nonnegative_int(self.sequence, "sequence")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.status, Status):
            raise ValueError("status must be a Status")
        confidence_value(self.confidence)
        nonempty(self.source_id, "source_id")
        if not isinstance(self.detections, tuple) or any(
            not isinstance(item, Detection) for item in self.detections
        ):
            raise ValueError("detections must be a tuple of Detection values")


@dataclass(frozen=True)
class WorldState:
    sequence: int
    timestamp_ms: int
    status: Status
    confidence: float
    source_id: str

    def __post_init__(self) -> None:
        nonnegative_int(self.sequence, "sequence")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.status, Status):
            raise ValueError("status must be a Status")
        confidence_value(self.confidence)
        nonempty(self.source_id, "source_id")


@dataclass(frozen=True)
class Intent:
    intent_id: str
    source_sequence: int
    created_at_ms: int
    deadline_ms: int
    action: Action
    reason: str

    def __post_init__(self) -> None:
        nonempty(self.intent_id, "intent_id")
        nonnegative_int(self.source_sequence, "source_sequence")
        nonnegative_int(self.created_at_ms, "created_at_ms")
        nonnegative_int(self.deadline_ms, "deadline_ms")
        if self.deadline_ms <= self.created_at_ms:
            raise ValueError("deadline_ms must be later than created_at_ms")
        if not isinstance(self.action, Action):
            raise ValueError("P0 supports only Action.NOOP")
        nonempty(self.reason, "reason")


@dataclass(frozen=True)
class Execution:
    intent_id: str
    timestamp_ms: int
    status: ExecutionStatus
    reason: str

    def __post_init__(self) -> None:
        nonempty(self.intent_id, "intent_id")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.status, ExecutionStatus):
            raise ValueError("invalid execution status")
        nonempty(self.reason, "reason")


@dataclass(frozen=True)
class Verification:
    intent_id: str
    timestamp_ms: int
    status: VerificationStatus
    reason: str

    def __post_init__(self) -> None:
        nonempty(self.intent_id, "intent_id")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.status, VerificationStatus):
            raise ValueError("invalid verification status")
        nonempty(self.reason, "reason")


@dataclass(frozen=True)
class TickResult:
    observation: Observation
    world: WorldState
    intent: Intent
    execution: Execution
    verification: Verification


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...


class Perception(Protocol):
    def observe(self, frame: Frame) -> Observation: ...


class StateEstimator(Protocol):
    def estimate(self, observation: Observation, now_ms: int) -> WorldState: ...


class Policy(Protocol):
    def decide(self, world: WorldState, now_ms: int) -> Intent: ...


class Controller(Protocol):
    def execute(self, intent: Intent, now_ms: int) -> Execution: ...


class Verifier(Protocol):
    def verify(self, intent: Intent, execution: Execution) -> Verification: ...


class Simulator(Protocol):
    def reset(self, seed: int) -> None: ...
    def step(self) -> Frame: ...
