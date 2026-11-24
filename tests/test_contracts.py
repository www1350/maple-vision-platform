from dataclasses import FrozenInstanceError

import pytest

from maple_vision.contracts import Action, Detection, Frame, Intent, Observation, Status


@pytest.mark.parametrize("sequence,timestamp", [(-1, 0), (0, -1), (True, 0)])
def test_frame_rejects_invalid_identity(sequence: int, timestamp: int) -> None:
    with pytest.raises(ValueError):
        Frame(sequence, timestamp, b"")


def test_frame_is_immutable() -> None:
    frame = Frame(0, 0, b"pixels")
    with pytest.raises(FrozenInstanceError):
        setattr(frame, "sequence", 1)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_observation_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Observation(0, 0, Status.KNOWN, confidence)


@pytest.mark.parametrize(
    "bbox",
    [(-1.0, 0.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, float("inf"), 1.0)],
)
def test_detector_box_has_positive_finite_extent(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="bbox"):
        Detection("monster", bbox, 0.9)


def test_intent_cannot_already_be_expired_when_created() -> None:
    with pytest.raises(ValueError, match="deadline"):
        Intent("i1", 0, 100, 100, Action.NOOP, "test")


def test_unknown_is_not_implicitly_known() -> None:
    observation = Observation(2, 50, Status.UNKNOWN)
    assert observation.status is Status.UNKNOWN
    assert observation.confidence == 0
    assert observation.detections == ()
