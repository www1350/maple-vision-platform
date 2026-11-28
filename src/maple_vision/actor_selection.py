"""Bind one reviewed actor anchor to a conservative, window-local track.

Multiple people remain normal scene observations. This selector follows only
the reviewed geometric subject and permanently invalidates an uncertain chain;
it does not establish character ownership or authorize input.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import hypot
from typing import Literal

from .catalog import canonical_json
from .contracts import Detection, Status
from .state_estimation import (
    _TRACK_ID,
    BeliefState,
    Box,
    StateConfig,
    TrackBelief,
    _box,
    _center,
    _clock,
    _integer,
    _iou,
    _number,
    _predicted_box,
    _text,
)

ACTOR_SELECTION_ALGORITHM_VERSION = "geometric-selected-actor-v1"
type SelectionStatus = Literal["waiting", "selected", "invalidated"]
_FRESH_REASONS = {"fresh_observation", "fresh_detection_hypotheses"}
_SELECTED_REASONS = {"anchor_selected", "continued_selected_actor"}
_INVALID_REASONS = {
    "source_changed",
    "anchor_missed",
    "anchor_time_mismatch",
    "anchor_rgb_mismatch",
    "anchor_missing_or_ambiguous_player",
    "anchor_competing_player",
    "future_frame",
    "stale_frame",
    "unusable_current_belief",
    "processing_clock_rollback",
    "belief_epoch_changed",
    "sequence_discontinuity",
    "nonincreasing_frame_time",
    "frame_time_gap",
    "selected_actor_missing",
    "selected_actor_low_confidence",
    "selected_actor_reacquired",
    "selected_actor_ambiguous",
    "selected_actor_not_current",
    "selected_actor_measurement_mismatch",
    "selected_actor_motion_exceeds_gate",
    "competing_player_overlap",
    "competing_player_association",
}


def _digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class ActorAnchor:
    source_sequence: int
    timestamp_ms: int
    rgb_sha256: str
    bbox: Box
    reviewer: str
    subject: str

    def __post_init__(self) -> None:
        _clock(self.source_sequence, "anchor source sequence")
        _clock(self.timestamp_ms, "anchor timestamp")
        _sha(self.rgb_sha256, "anchor RGB")
        _box(self.bbox)
        _text(self.reviewer, "anchor reviewer", 128)
        _text(self.subject, "anchor subject", 128)
        object.__setattr__(self, "bbox", tuple(float(value) for value in self.bbox))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_sequence": self.source_sequence,
            "timestamp_ms": self.timestamp_ms,
            "rgb_sha256": self.rgb_sha256,
            "bbox": list(self.bbox),
            "reviewer": self.reviewer,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActorAnchor":
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "source_sequence",
                "timestamp_ms",
                "rgb_sha256",
                "bbox",
                "reviewer",
                "subject",
            }
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not isinstance(value["bbox"], list)
        ):
            raise ValueError("invalid actor anchor schema or fields")
        return cls(
            source_sequence=value["source_sequence"],
            timestamp_ms=value["timestamp_ms"],
            rgb_sha256=value["rgb_sha256"],
            bbox=tuple(value["bbox"]),
            reviewer=value["reviewer"],
            subject=value["subject"],
        )


@dataclass(frozen=True)
class ActorSelectionConfig:
    anchor_iou_threshold: float = 0.5
    min_confidence: float = 0.7
    max_age_ms: int = 250
    max_gap_ms: int = 250

    def __post_init__(self) -> None:
        threshold = _number(self.anchor_iou_threshold, "anchor_iou_threshold", 0, 1)
        if threshold == 0:
            raise ValueError("anchor_iou_threshold must be positive")
        object.__setattr__(self, "anchor_iou_threshold", threshold)
        object.__setattr__(
            self, "min_confidence", _number(self.min_confidence, "min_confidence", 0, 1)
        )
        _integer(self.max_age_ms, "max_age_ms", 0, 60_000)
        _integer(self.max_gap_ms, "max_gap_ms", 1, 60_000)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "anchor_iou_threshold": self.anchor_iou_threshold,
            "min_confidence": self.min_confidence,
            "max_age_ms": self.max_age_ms,
            "max_gap_ms": self.max_gap_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActorSelectionConfig":
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema_version",
                "anchor_iou_threshold",
                "min_confidence",
                "max_age_ms",
                "max_gap_ms",
            }
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("invalid actor selection configuration schema or fields")
        return cls(
            anchor_iou_threshold=value["anchor_iou_threshold"],
            min_confidence=value["min_confidence"],
            max_age_ms=value["max_age_ms"],
            max_gap_ms=value["max_gap_ms"],
        )


@dataclass(frozen=True)
class ActorSelection:
    status: SelectionStatus
    actor_track_id: str | None
    reason: str
    source_id: str
    source_sequence: int
    timestamp_ms: int
    now_ms: int
    epoch: int
    selection_id: str
    belief_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in {
            "waiting",
            "selected",
            "invalidated",
        }:
            raise ValueError("invalid actor selection status")
        _text(self.reason, "selection reason", 128)
        if (
            (self.status == "waiting" and self.reason != "waiting_for_anchor")
            or (self.status == "selected" and self.reason not in _SELECTED_REASONS)
            or (self.status == "invalidated" and self.reason not in _INVALID_REASONS)
        ):
            raise ValueError("actor selection status and reason disagree")
        if self.status == "selected":
            if not isinstance(self.actor_track_id, str) or not _TRACK_ID.fullmatch(
                self.actor_track_id
            ):
                raise ValueError("selected actor requires a valid track identity")
        elif self.actor_track_id is not None:
            raise ValueError("unselected actor must not retain a current track identity")
        _text(self.source_id, "selection source", 256)
        _clock(self.source_sequence, "selection source sequence")
        _clock(self.timestamp_ms, "selection timestamp")
        _clock(self.now_ms, "selection clock")
        _clock(self.epoch, "selection epoch")
        _sha(self.selection_id, "selection identity")
        _sha(self.belief_sha256, "selection belief identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "actor_track_id": self.actor_track_id,
            "reason": self.reason,
            "source_id": self.source_id,
            "source_sequence": self.source_sequence,
            "timestamp_ms": self.timestamp_ms,
            "now_ms": self.now_ms,
            "epoch": self.epoch,
            "selection_id": self.selection_id,
            "belief_sha256": self.belief_sha256,
            "read_only": True,
            "actionable": False,
            "owner_identity_verified": False,
            "action_execution": "none",
        }


def _checked_belief(belief: BeliefState) -> None:
    if type(belief) is not BeliefState:
        raise ValueError("actor selection input must be BeliefState")
    belief.__post_init__()
    observation = belief.observation
    observation.__post_init__()
    _text(observation.source_id, "observation source", 256)
    _clock(observation.sequence, "observation sequence")
    _clock(observation.timestamp_ms, "observation timestamp")
    if len(observation.detections) > 512:
        raise ValueError("actor selection exceeds the detection budget")
    for detection in observation.detections:
        detection.__post_init__()
        _text(detection.class_id, "detection class", 64)
        _box(detection.bbox)
    for track in belief.tracks:
        track.__post_init__()
    for field in belief.fields:
        field.__post_init__()


def _measurement(track: TrackBelief, detection: Detection) -> bool:
    return (
        detection.class_id == track.class_id
        and detection.bbox == track.bbox
        and detection.confidence == track.confidence
    )


def _track_issue(track: TrackBelief, belief: BeliefState) -> str:
    observation = belief.observation
    if (
        track.class_id != "player"
        or not track.observed_this_frame
        or track.source_id != observation.source_id
        or track.source_sequence != observation.sequence
        or track.observed_at_ms != observation.timestamp_ms
        or track.expires_at_ms <= belief.now_ms
        or track.association not in {"new", "matched", "reacquired", "ambiguous"}
    ):
        return "selected_actor_not_current"
    if track.identity_uncertain or track.candidate_ids or track.association == "ambiguous":
        return "selected_actor_ambiguous"
    if track.association == "reacquired":
        return "selected_actor_reacquired"
    if sum(_measurement(track, detection) for detection in observation.detections) != 1:
        return "selected_actor_measurement_mismatch"
    return ""


def validate_actor_selection(selection: ActorSelection, belief: BeliefState) -> TrackBelief | None:
    """Reject a result from another belief before returning a current actor track.

    This is an in-process evidence binding, not an authenticated owner identity.
    Reproducing a selection requires replaying its anchor and entire frame chain.
    """
    if type(selection) is not ActorSelection:
        raise ValueError("actor selection must be an ActorSelection")
    selection.__post_init__()
    _checked_belief(belief)
    observation = belief.observation
    if (
        selection.source_id != observation.source_id
        or selection.source_sequence != observation.sequence
        or selection.timestamp_ms != observation.timestamp_ms
        or selection.now_ms != belief.now_ms
        or selection.epoch != belief.epoch
        or selection.belief_sha256 != _digest(belief.to_dict())
    ):
        raise ValueError("actor selection does not match the current belief")
    if selection.status != "selected":
        return None
    if (
        observation.timestamp_ms > belief.now_ms
        or observation.status is Status.STALE
        or belief.reason not in _FRESH_REASONS
    ):
        raise ValueError("selected actor needs a fresh current belief")
    candidates = tuple(
        track for track in belief.tracks if track.track_id == selection.actor_track_id
    )
    if len(candidates) != 1 or _track_issue(candidates[0], belief):
        raise ValueError("selected actor does not match an unambiguous current player measurement")
    actor = candidates[0]
    if selection.reason == "continued_selected_actor" and actor.association != "matched":
        raise ValueError("continued selected actor must retain a matched current identity")
    return actor


class ActorSelector:
    """Select at exactly one reviewed anchor; any later ambiguity ends the chain."""

    def __init__(
        self,
        anchor: ActorAnchor,
        config: ActorSelectionConfig | None = None,
        *,
        state_config: StateConfig | None = None,
    ) -> None:
        if type(anchor) is not ActorAnchor:
            raise ValueError("anchor must be an ActorAnchor")
        if config is not None and type(config) is not ActorSelectionConfig:
            raise ValueError("selection config must be ActorSelectionConfig")
        if state_config is not None and type(state_config) is not StateConfig:
            raise ValueError("selection state config must be StateConfig")
        self.anchor = ActorAnchor.from_dict(anchor.to_dict())
        self.config = ActorSelectionConfig.from_dict((config or ActorSelectionConfig()).to_dict())
        self.state_config = StateConfig.from_dict((state_config or StateConfig()).to_dict())
        self.selection_id = _digest(
            {
                "algorithm_version": ACTOR_SELECTION_ALGORITHM_VERSION,
                "anchor": self.anchor.to_dict(),
                "config": self.config.to_dict(),
                "state_config": self.state_config.to_dict(),
            }
        )
        self._source_id: str | None = None
        self._selected_epoch: int | None = None
        self._previous: TrackBelief | None = None
        self._last_now_ms: int | None = None
        self._invalid_reason: str | None = None

    def _result(
        self,
        belief: BeliefState,
        status: SelectionStatus,
        reason: str,
        actor: TrackBelief | None = None,
    ) -> ActorSelection:
        observation = belief.observation
        return ActorSelection(
            status=status,
            actor_track_id=actor.track_id if actor is not None else None,
            reason=reason,
            source_id=observation.source_id,
            source_sequence=observation.sequence,
            timestamp_ms=observation.timestamp_ms,
            now_ms=belief.now_ms,
            epoch=belief.epoch,
            selection_id=self.selection_id,
            belief_sha256=_digest(belief.to_dict()),
        )

    def _invalidate(self, belief: BeliefState, reason: str) -> ActorSelection:
        if self._invalid_reason is None:
            self._invalid_reason = reason
        self._previous = None
        return self._result(belief, "invalidated", self._invalid_reason)

    def _freshness_issue(self, belief: BeliefState) -> str:
        if belief.observation.timestamp_ms > belief.now_ms:
            return "future_frame"
        if belief.now_ms - belief.observation.timestamp_ms > self.config.max_age_ms:
            return "stale_frame"
        if belief.observation.status is Status.STALE or belief.reason not in _FRESH_REASONS:
            return "unusable_current_belief"
        return ""

    def _other_players(self, actor: TrackBelief, belief: BeliefState) -> tuple[Detection, ...]:
        return tuple(
            detection
            for detection in belief.observation.detections
            if detection.class_id == "player" and not _measurement(actor, detection)
        )

    def _at_anchor(self, belief: BeliefState, rgb_sha256: str) -> ActorSelection:
        observation = belief.observation
        if observation.timestamp_ms != self.anchor.timestamp_ms:
            return self._invalidate(belief, "anchor_time_mismatch")
        if rgb_sha256 != self.anchor.rgb_sha256:
            return self._invalidate(belief, "anchor_rgb_mismatch")
        issue = self._freshness_issue(belief)
        if issue:
            return self._invalidate(belief, issue)
        candidates = tuple(
            track
            for track in belief.tracks
            if track.class_id == "player"
            and track.confidence >= self.config.min_confidence
            and not _track_issue(track, belief)
            and _iou(track.bbox, self.anchor.bbox) >= self.config.anchor_iou_threshold
        )
        if len(candidates) != 1:
            return self._invalidate(belief, "anchor_missing_or_ambiguous_player")
        actor = candidates[0]
        if any(
            _iou(other.bbox, self.anchor.bbox) > 0 or _iou(other.bbox, actor.bbox) > 0
            for other in self._other_players(actor, belief)
        ):
            return self._invalidate(belief, "anchor_competing_player")
        self._previous = actor
        self._selected_epoch = belief.epoch
        self._last_now_ms = belief.now_ms
        return self._result(belief, "selected", "anchor_selected", actor)

    def _association_cost(self, predicted: Box, current: Box) -> float | None:
        px, py = _center(predicted)
        cx, cy = _center(current)
        distance = hypot(cx - px, cy - py)
        if distance > self.state_config.max_match_distance:
            return None
        return distance / self.state_config.max_match_distance + 0.25 * (
            1 - _iou(predicted, current)
        )

    def update(self, belief: BeliefState, *, rgb_sha256: str) -> ActorSelection:
        _checked_belief(belief)
        _sha(rgb_sha256, "current RGB")
        observation = belief.observation
        if self._invalid_reason is not None:
            return self._result(belief, "invalidated", self._invalid_reason)
        if self._source_id is None:
            self._source_id = observation.source_id
        elif observation.source_id != self._source_id:
            return self._invalidate(belief, "source_changed")
        previous = self._previous
        if previous is None:
            if observation.sequence < self.anchor.source_sequence:
                return self._result(belief, "waiting", "waiting_for_anchor")
            if observation.sequence > self.anchor.source_sequence:
                return self._invalidate(belief, "anchor_missed")
            return self._at_anchor(belief, rgb_sha256)
        if self._last_now_ms is not None and belief.now_ms < self._last_now_ms:
            return self._invalidate(belief, "processing_clock_rollback")
        if belief.epoch != self._selected_epoch:
            return self._invalidate(belief, "belief_epoch_changed")
        if observation.sequence != previous.source_sequence + 1:
            return self._invalidate(belief, "sequence_discontinuity")
        elapsed = observation.timestamp_ms - previous.observed_at_ms
        if elapsed <= 0:
            return self._invalidate(belief, "nonincreasing_frame_time")
        if elapsed > self.config.max_gap_ms:
            return self._invalidate(belief, "frame_time_gap")
        issue = self._freshness_issue(belief)
        if issue:
            return self._invalidate(belief, issue)
        candidates = tuple(track for track in belief.tracks if track.track_id == previous.track_id)
        if len(candidates) != 1:
            return self._invalidate(belief, "selected_actor_missing")
        actor = candidates[0]
        issue = _track_issue(actor, belief)
        if issue:
            return self._invalidate(belief, issue)
        if actor.association != "matched":
            return self._invalidate(belief, "selected_actor_not_current")
        if actor.confidence < self.config.min_confidence:
            return self._invalidate(belief, "selected_actor_low_confidence")
        others = self._other_players(actor, belief)
        if any(_iou(other.bbox, actor.bbox) > 0 for other in others):
            return self._invalidate(belief, "competing_player_overlap")
        predicted = _predicted_box(previous, observation.timestamp_ms)
        selected_cost = self._association_cost(predicted, actor.bbox)
        if selected_cost is None:
            return self._invalidate(belief, "selected_actor_motion_exceeds_gate")
        for other in others:
            cost = self._association_cost(predicted, other.bbox)
            if cost is not None and cost <= selected_cost + self.state_config.ambiguity_margin:
                return self._invalidate(belief, "competing_player_association")
        self._previous = actor
        self._last_now_ms = belief.now_ms
        return self._result(belief, "selected", "continued_selected_actor", actor)
