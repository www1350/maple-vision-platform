"""Independent regression cases found during the P0 implementation review."""

import pytest

from maple_vision.contracts import Execution, ExecutionStatus, Intent, VerificationStatus
from maple_vision.pipeline import (
    FreshnessEstimator,
    NoOpController,
    NoOpPerception,
    NoOpPolicy,
    NoOpVerifier,
    Pipeline,
)
from maple_vision.sources import DemoSource


class RecordingController(NoOpController):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, intent: Intent, now_ms: int) -> Execution:
        self.calls += 1
        return super().execute(intent, now_ms)


@pytest.mark.parametrize("second_now", [50, None])
def test_tick_clock_rollback_is_rejected_before_controller(second_now: int | None) -> None:
    controller = RecordingController()
    pipeline = Pipeline(
        DemoSource(2, 50),
        NoOpPerception(),
        FreshnessEstimator(),
        NoOpPolicy(),
        controller,
        NoOpVerifier(),
    )
    assert pipeline.tick(now_ms=1000) is not None
    with pytest.raises(ValueError, match="backwards"):
        pipeline.tick(now_ms=second_now)
    assert controller.calls == 1


class LateController:
    def execute(self, intent: Intent, now_ms: int) -> Execution:
        # A valid receipt identity alone cannot justify a successful verification.
        return Execution(
            intent.intent_id, intent.deadline_ms, ExecutionStatus.SKIPPED, "late_receipt"
        )


def test_late_execution_receipt_does_not_claim_completion() -> None:
    pipeline = Pipeline(
        DemoSource(1),
        NoOpPerception(),
        FreshnessEstimator(),
        NoOpPolicy(),
        LateController(),
        NoOpVerifier(),
    )
    result = pipeline.tick()
    assert result is not None
    assert result.verification.status is VerificationStatus.TIMEOUT
