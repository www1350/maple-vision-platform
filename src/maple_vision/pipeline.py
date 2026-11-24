"""Offline foundation pipeline. No component implements OS or game input."""

from .contracts import (
    Action,
    Controller,
    Execution,
    ExecutionStatus,
    Frame,
    FrameSource,
    Intent,
    Observation,
    Perception,
    Policy,
    StateEstimator,
    Status,
    TickResult,
    Verification,
    VerificationStatus,
    Verifier,
    WorldState,
    nonnegative_int,
)


class NoOpPerception:
    def observe(self, frame: Frame) -> Observation:
        return Observation(frame.sequence, frame.timestamp_ms, Status.UNKNOWN)


def run_once(source: FrameSource, perception: Perception) -> Observation | None:
    """Legacy observation-only API, retained for existing callers."""
    frame = source.read()
    if frame is None:
        return None
    return perception.observe(frame)


class FreshnessEstimator:
    def __init__(self, max_age_ms: int = 250) -> None:
        nonnegative_int(max_age_ms, "max_age_ms")
        self.max_age_ms = max_age_ms

    def estimate(self, observation: Observation, now_ms: int) -> WorldState:
        nonnegative_int(now_ms, "now_ms")
        if now_ms < observation.timestamp_ms:
            raise ValueError("observation timestamp is in the future")
        status = observation.status
        if now_ms - observation.timestamp_ms > self.max_age_ms:
            status = Status.STALE
        return WorldState(
            observation.sequence,
            observation.timestamp_ms,
            status,
            observation.confidence,
            observation.source_id,
        )


class NoOpPolicy:
    def decide(self, world: WorldState, now_ms: int) -> Intent:
        return Intent(
            intent_id=f"noop-{world.sequence}",
            source_sequence=world.sequence,
            created_at_ms=now_ms,
            deadline_ms=now_ms + 250,
            action=Action.NOOP,
            reason=f"p0_noop:world_{world.status.value}",
        )


class NoOpController:
    def execute(self, intent: Intent, now_ms: int) -> Execution:
        nonnegative_int(now_ms, "now_ms")
        if not intent.created_at_ms <= now_ms < intent.deadline_ms:
            return Execution(
                intent.intent_id, now_ms, ExecutionStatus.REJECTED, "intent_outside_validity"
            )
        return Execution(
            intent.intent_id, now_ms, ExecutionStatus.SKIPPED, "noop_has_no_side_effect"
        )


class NoOpVerifier:
    def verify(self, intent: Intent, execution: Execution) -> Verification:
        if execution.intent_id != intent.intent_id:
            raise ValueError("execution intent_id does not match intent")
        if execution.timestamp_ms < intent.created_at_ms:
            return Verification(
                intent.intent_id,
                execution.timestamp_ms,
                VerificationStatus.FAILED,
                "execution_precedes_intent",
            )
        if execution.timestamp_ms >= intent.deadline_ms:
            return Verification(
                intent.intent_id,
                execution.timestamp_ms,
                VerificationStatus.TIMEOUT,
                "intent_deadline_exceeded",
            )
        if execution.status is ExecutionStatus.REJECTED:
            return Verification(
                intent.intent_id,
                execution.timestamp_ms,
                VerificationStatus.FAILED,
                execution.reason,
            )
        return Verification(
            intent.intent_id,
            execution.timestamp_ms,
            VerificationStatus.NOT_APPLICABLE,
            "noop_is_not_evidence_of_gameplay_success",
        )


class Pipeline:
    """One source clock per instance; reject reordering and mismatched identities."""

    def __init__(
        self,
        source: FrameSource,
        perception: Perception,
        estimator: StateEstimator,
        policy: Policy,
        controller: Controller,
        verifier: Verifier,
    ) -> None:
        self.source = source
        self.perception = perception
        self.estimator = estimator
        self.policy = policy
        self.controller = controller
        self.verifier = verifier
        self._last_sequence = -1
        self._last_timestamp_ms = -1
        self._last_now_ms = -1

    def tick(self, now_ms: int | None = None) -> TickResult | None:
        frame = self.source.read()
        if frame is None:
            return None
        if frame.sequence <= self._last_sequence:
            raise ValueError("frame sequence must strictly increase")
        if frame.timestamp_ms < self._last_timestamp_ms:
            raise ValueError("frame timestamp must not move backwards")
        current_ms = frame.timestamp_ms if now_ms is None else now_ms
        nonnegative_int(current_ms, "now_ms")
        if current_ms < frame.timestamp_ms:
            raise ValueError("frame timestamp is in the future")
        if current_ms < self._last_now_ms:
            raise ValueError("tick clock must not move backwards")
        observation = self.perception.observe(frame)
        identity = (frame.sequence, frame.timestamp_ms)
        if (observation.sequence, observation.timestamp_ms) != identity:
            raise ValueError("observation does not match its frame")
        world = self.estimator.estimate(observation, current_ms)
        if (world.sequence, world.timestamp_ms) != identity:
            raise ValueError("world state does not match its observation")
        intent = self.policy.decide(world, current_ms)
        if intent.source_sequence != world.sequence:
            raise ValueError("intent does not match its world state")
        if intent.created_at_ms != current_ms:
            raise ValueError("intent timestamp does not match the current tick")
        execution = self.controller.execute(intent, current_ms)
        if execution.intent_id != intent.intent_id:
            raise ValueError("execution does not match its intent")
        verification = self.verifier.verify(intent, execution)
        if verification.intent_id != intent.intent_id:
            raise ValueError("verification does not match its intent")
        self._last_sequence = frame.sequence
        self._last_timestamp_ms = frame.timestamp_ms
        self._last_now_ms = current_ms
        return TickResult(observation, world, intent, execution, verification)
