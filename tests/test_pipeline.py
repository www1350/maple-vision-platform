from collections.abc import Iterator

import pytest

from maple_vision.contracts import (
    Action,
    Execution,
    ExecutionStatus,
    Frame,
    FrameSource,
    Intent,
    Observation,
    Perception,
    Status,
    VerificationStatus,
)
from maple_vision.pipeline import (
    FreshnessEstimator,
    NoOpController,
    NoOpPerception,
    NoOpPolicy,
    NoOpVerifier,
    Pipeline,
)
from maple_vision.sources import DemoSource


class Frames:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames: Iterator[Frame] = iter(frames)

    def read(self) -> Frame | None:
        return next(self.frames, None)


class CountingController(NoOpController):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, intent: Intent, now_ms: int) -> Execution:
        self.calls += 1
        return super().execute(intent, now_ms)


def make_pipeline(source: FrameSource, perception: Perception | None = None) -> Pipeline:
    return Pipeline(
        source,
        perception or NoOpPerception(),
        FreshnessEstimator(),
        NoOpPolicy(),
        CountingController(),
        NoOpVerifier(),
    )


def test_complete_noop_chain_is_correlated_without_fabricated_success() -> None:
    pipeline = make_pipeline(DemoSource(3))
    for sequence in range(3):
        result = pipeline.tick()
        assert result is not None
        assert result.observation.sequence == result.world.sequence == sequence
        assert result.intent.source_sequence == sequence
        assert (
            result.intent.intent_id == result.execution.intent_id == result.verification.intent_id
        )
        assert result.world.status is Status.UNKNOWN
        assert result.execution.status is ExecutionStatus.SKIPPED
        assert result.verification.status is VerificationStatus.NOT_APPLICABLE
    assert pipeline.tick() is None
    assert isinstance(pipeline.controller, CountingController)
    assert pipeline.controller.calls == 3


@pytest.mark.parametrize("status", list(Status))
def test_estimator_preserves_current_observation_status(status: Status) -> None:
    observation = Observation(0, 100, status, 0.4)
    assert FreshnessEstimator().estimate(observation, 150).status is status


def test_freshness_boundary_is_inclusive_then_becomes_stale() -> None:
    estimator = FreshnessEstimator(250)
    observation = Observation(0, 100, Status.KNOWN, 0.9)
    assert estimator.estimate(observation, 350).status is Status.KNOWN
    assert estimator.estimate(observation, 351).status is Status.STALE
    with pytest.raises(ValueError, match="future"):
        estimator.estimate(observation, 99)


@pytest.mark.parametrize("second", [Frame(0, 10, b""), Frame(1, 9, b"")])
def test_reordered_frames_never_reach_controller(second: Frame) -> None:
    pipeline = make_pipeline(Frames([Frame(0, 10, b""), second]))
    assert pipeline.tick() is not None
    with pytest.raises(ValueError):
        pipeline.tick()
    assert isinstance(pipeline.controller, CountingController)
    assert pipeline.controller.calls == 1


class MismatchedPerception:
    def observe(self, frame: Frame) -> Observation:
        return Observation(frame.sequence + 1, frame.timestamp_ms, Status.KNOWN)


def test_mismatched_observation_does_not_reach_controller() -> None:
    pipeline = make_pipeline(DemoSource(), MismatchedPerception())
    with pytest.raises(ValueError, match="does not match its frame"):
        pipeline.tick()
    assert isinstance(pipeline.controller, CountingController)
    assert pipeline.controller.calls == 0


def test_expired_intent_is_rejected_and_verified_as_timeout() -> None:
    intent = Intent("deadline", 0, 100, 200, Action.NOOP, "test")
    execution = NoOpController().execute(intent, 200)
    assert execution.status is ExecutionStatus.REJECTED
    assert NoOpVerifier().verify(intent, execution).status is VerificationStatus.TIMEOUT


def test_execution_before_intent_is_failed() -> None:
    intent = Intent("early", 0, 100, 200, Action.NOOP, "test")
    execution = NoOpController().execute(intent, 99)
    assert execution.status is ExecutionStatus.REJECTED
    assert NoOpVerifier().verify(intent, execution).status is VerificationStatus.FAILED


def test_verifier_rejects_unrelated_receipt() -> None:
    intent = Intent("one", 0, 0, 100, Action.NOOP, "test")
    execution = Execution("two", 0, ExecutionStatus.SKIPPED, "test")
    with pytest.raises(ValueError, match="intent_id"):
        NoOpVerifier().verify(intent, execution)
