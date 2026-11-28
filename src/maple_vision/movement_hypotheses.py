"""Bounded geometric movement candidates from current observation-only beliefs.

Neither track identity nor these pixel relationships establish character ownership,
rope grip, game physics, or a successful action. This module never executes inputs
or upgrades the actionability of the supplied belief.
"""

from collections import deque
from dataclasses import dataclass, fields
from math import fsum, hypot, isfinite
from typing import Literal

from .contracts import Detection, Status
from .state_estimation import BeliefState, Box, TrackBelief

MOVEMENT_ALGORITHM_VERSION = "geometric-movement-v1"
type MovementState = Literal[
    "unknown", "near_rope", "climbing_candidate", "platform_arrival_candidate"
]
type Pair = tuple[float, float]
_TERRAIN = frozenset({"platform", "rope", "ladder"})
_ROPES = frozenset({"rope", "ladder"})
_LIMITATIONS = (
    "Track IDs are geometric hypotheses; character identity and ownership are unverified.",
    "Near-rope means spatial proximity only; climbing and arrival are geometric candidates.",
    "An aligned jump may also be a climbing candidate; rope grip and contact are unverified.",
    "Camera translation requires mutually consistent current terrain landmarks; "
    "zoom, parallax and detection errors are not resolved.",
    "Changing actor box sizes reset motion evidence; animation geometry is not inferred.",
    "Pixel and time thresholds are engineering defaults, not calibrated real-game accuracy.",
    "Read-only evidence never authorizes or executes an action.",
)


def _integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _number(value: object, name: str, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


@dataclass(frozen=True)
class MovementConfig:
    max_age_ms: int = 250
    max_gap_ms: int = 250
    history_size: int = 64
    min_actor_confidence: float = 0.7
    min_terrain_confidence: float = 0.7
    near_rope_distance_px: float = 12.0
    rope_alignment_tolerance_px: float = 10.0
    min_climb_samples: int = 3
    min_climb_duration_ms: int = 100
    min_climb_distance_px: float = 6.0
    min_vertical_step_px: float = 1.0
    max_horizontal_drift_px: float = 6.0
    max_vertical_speed_px_per_second: float = 500.0
    min_arrival_samples: int = 3
    min_arrival_duration_ms: int = 100
    max_climb_to_arrival_ms: int = 500
    platform_top_tolerance_px: float = 4.0
    platform_min_horizontal_overlap_ratio: float = 0.5
    stability_tolerance_px: float = 2.0
    camera_min_landmarks: int = 2
    camera_delta_tolerance_px: float = 1.0
    max_camera_translation_px: float = 64.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("max_age_ms", 0, 60_000),
            ("max_gap_ms", 1, 60_000),
            ("history_size", 3, 256),
            ("min_climb_samples", 3, 256),
            ("min_climb_duration_ms", 1, 60_000),
            ("min_arrival_samples", 3, 256),
            ("min_arrival_duration_ms", 1, 60_000),
            ("max_climb_to_arrival_ms", 1, 60_000),
            ("camera_min_landmarks", 2, 512),
        ):
            _integer(getattr(self, name), name, lower, upper)
        for name, low, high in (
            ("min_actor_confidence", 0.0, 1.0),
            ("min_terrain_confidence", 0.0, 1.0),
            ("near_rope_distance_px", 0.0, 8192.0),
            ("rope_alignment_tolerance_px", 0.0, 8192.0),
            ("min_climb_distance_px", 0.01, 8192.0),
            ("min_vertical_step_px", 0.01, 8192.0),
            ("max_horizontal_drift_px", 0.0, 8192.0),
            ("max_vertical_speed_px_per_second", 0.01, 100_000.0),
            ("platform_top_tolerance_px", 0.0, 8192.0),
            ("platform_min_horizontal_overlap_ratio", 0.01, 1.0),
            ("stability_tolerance_px", 0.0, 8192.0),
            ("camera_delta_tolerance_px", 0.0, 8192.0),
            ("max_camera_translation_px", 0.0, 8192.0),
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        if max(self.min_climb_samples, self.min_arrival_samples) > self.history_size:
            raise ValueError("history_size must cover the climb and arrival sample thresholds")
        if self.max_climb_to_arrival_ms < self.min_arrival_duration_ms:
            raise ValueError("climb-to-arrival expiry must cover the arrival duration")
        if self.min_vertical_step_px > self.min_climb_distance_px:
            raise ValueError("minimum vertical step must not exceed minimum climb distance")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    @classmethod
    def from_dict(cls, value: object) -> "MovementConfig":
        keys = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) != keys | {"schema_version"}:
            raise ValueError("invalid movement configuration fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported movement configuration schema")
        return cls(**{name: value[name] for name in keys})


@dataclass(frozen=True)
class _Frame:
    source_id: str
    source_sequence: int
    timestamp_ms: int
    epoch: int

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class _Landmark:
    track_id: str
    class_id: str
    previous_bbox: Box
    current_bbox: Box
    delta: Pair

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "previous_bbox": list(self.previous_bbox),
            "current_bbox": list(self.current_bbox),
            "delta_px": list(self.delta),
        }


@dataclass(frozen=True)
class _Camera:
    status: str = "unknown"
    reason: str = "not_evaluated"
    delta: Pair | None = None
    landmarks: tuple[_Landmark, ...] = ()
    rejected_landmark_ids: tuple[str, ...] = ()
    previous_frame: _Frame | None = None
    current_frame: _Frame | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "delta_px": list(self.delta) if self.delta is not None else None,
            "landmark_ids": [landmark.track_id for landmark in self.landmarks],
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "rejected_landmark_ids": list(self.rejected_landmark_ids),
            "previous_frame": self.previous_frame.to_dict() if self.previous_frame else None,
            "current_frame": self.current_frame.to_dict() if self.current_frame else None,
        }


@dataclass(frozen=True)
class _Motion:
    elapsed_ms: int
    actor_screen_delta: Pair
    actor_camera_relative_delta: Pair
    actor_rope_relative_delta: Pair
    direction: str
    sample_count: int
    duration_ms: int
    vertical_distance_px: float
    horizontal_drift_px: float

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "actor_screen_delta_px": list(self.actor_screen_delta),
            "actor_camera_relative_delta_px": list(self.actor_camera_relative_delta),
            "actor_rope_relative_delta_px": list(self.actor_rope_relative_delta),
            "actor_reference": "feet_midpoint",
            "rope_reference": "bbox_center",
            "direction": self.direction,
            "sample_count": self.sample_count,
            "duration_ms": self.duration_ms,
            "vertical_distance_px": self.vertical_distance_px,
            "horizontal_drift_px": self.horizontal_drift_px,
        }


@dataclass(frozen=True)
class MovementHypothesis:
    state: MovementState
    frame: _Frame
    now_ms: int
    actor_track_id: str | None
    rope_track_id: str | None
    platform_track_id: str | None
    camera: _Camera
    relative_motion: _Motion | None
    evidence_frames: tuple[_Frame, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "state": self.state,
            "read_only": True,
            "actionable": False,
            "action_execution": "none",
            **self.frame.to_dict(),
            "now_ms": self.now_ms,
            "actor_track_id": self.actor_track_id,
            "rope_track_id": self.rope_track_id,
            "platform_track_id": self.platform_track_id,
            "camera": self.camera.to_dict(),
            "relative_motion": self.relative_motion.to_dict() if self.relative_motion else None,
            "evidence_frames": [frame.to_dict() for frame in self.evidence_frames],
            "reasons": list(self.reasons),
            "limitations": list(_LIMITATIONS),
        }


def _center(box: Box) -> Pair:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _feet(box: Box) -> Pair:
    return (box[0] + box[2]) / 2, box[3]


def _subtract(first: Pair, second: Pair) -> Pair:
    return first[0] - second[0], first[1] - second[1]


def _size(box: Box) -> Pair:
    return box[2] - box[0], box[3] - box[1]


def _same_measurement(track: TrackBelief, detection: Detection) -> bool:
    return (
        track.class_id == detection.class_id
        and track.bbox == detection.bbox
        and track.confidence == detection.confidence
    )


@dataclass(frozen=True)
class _Sample:
    frame: _Frame
    actor: TrackBelief
    terrain: tuple[TrackBelief, ...]
    rope: TrackBelief | None


@dataclass(frozen=True)
class _Pose:
    frame: _Frame
    relative: Pair


class MovementHypothesisEstimator:
    def __init__(self, config: MovementConfig | None = None) -> None:
        if config is not None and type(config) is not MovementConfig:
            raise ValueError("movement config must be MovementConfig")
        self.config = config or MovementConfig()
        self._last_frame: _Frame | None = None
        self._last_now_ms: int | None = None
        self._previous: _Sample | None = None
        self._motion: deque[_Pose] = deque(maxlen=self.config.history_size)
        self._arrival: deque[_Pose] = deque(maxlen=self.config.history_size)
        self._direction = 0
        self._climb_frames: tuple[_Frame, ...] = ()
        self._last_climb_ms: int | None = None
        self._last_climb_relative: Pair | None = None
        self._climb_direction = 0
        self._arrival_platform_id: str | None = None

    def _reset_chain(self) -> None:
        self._motion.clear()
        self._direction = 0
        self._climb_frames = ()
        self._last_climb_ms = None
        self._last_climb_relative = None
        self._climb_direction = 0
        self._arrival.clear()
        self._arrival_platform_id = None

    def _fresh(self, track: TrackBelief, belief: BeliefState) -> bool:
        observation = belief.observation
        return (
            track.observed_this_frame
            and track.source_id == observation.source_id
            and track.source_sequence == observation.sequence
            and track.observed_at_ms == observation.timestamp_ms
            and track.expires_at_ms > belief.now_ms
            and any(_same_measurement(track, detection) for detection in observation.detections)
        )

    def _actor(self, belief: BeliefState) -> tuple[TrackBelief | None, str]:
        observation = belief.observation
        players = tuple(d for d in observation.detections if d.class_id == "player")
        if not players:
            return None, "missing_player"
        if len(players) != 1:
            return None, "conflicting_players"
        if players[0].confidence < self.config.min_actor_confidence:
            return None, "low_confidence_player"
        field = belief.field("player_bbox")
        if (
            field.status is not Status.KNOWN
            or not field.observed_this_frame
            or field.source_id != observation.source_id
            or field.source_sequence != observation.sequence
            or field.observed_at_ms != observation.timestamp_ms
            or field.expires_at_ms is None
            or field.expires_at_ms <= belief.now_ms
            or field.confidence < self.config.min_actor_confidence
            or field.value != players[0].bbox
        ):
            return None, "missing_current_player_bbox"
        actors = tuple(
            track
            for track in belief.tracks
            if track.class_id == "player" and self._fresh(track, belief)
        )
        if len(actors) != 1 or actors[0].bbox != field.value:
            return None, "missing_or_conflicting_actor_track"
        actor = actors[0]
        if actor.confidence < self.config.min_actor_confidence:
            return None, "low_confidence_actor_track"
        if actor.identity_uncertain or actor.candidate_ids or actor.association == "ambiguous":
            return None, "uncertain_actor_identity"
        if actor.association == "reacquired":
            return None, "reacquired_actor"
        if actor.association not in {"new", "matched"}:
            return None, "unobserved_actor"
        return actor, ""

    def _terrain(self, belief: BeliefState) -> tuple[TrackBelief, ...]:
        return tuple(
            sorted(
                (
                    track
                    for track in belief.tracks
                    if track.class_id in _TERRAIN
                    and track.confidence >= self.config.min_terrain_confidence
                    and self._fresh(track, belief)
                    and track.association in {"new", "matched"}
                    and not track.identity_uncertain
                    and not track.candidate_ids
                ),
                key=lambda track: track.track_id,
            )
        )

    def _near_rope(self, actor: Box, rope: Box) -> bool:
        dx = max(0.0, actor[0] - rope[2], rope[0] - actor[2])
        dy = max(0.0, actor[1] - rope[3], rope[1] - actor[3])
        return hypot(dx, dy) <= self.config.near_rope_distance_px

    def _support(self, actor: Box, platform: Box) -> bool:
        overlap = max(0.0, min(actor[2], platform[2]) - max(actor[0], platform[0]))
        return (
            abs(actor[3] - platform[1]) <= self.config.platform_top_tolerance_px
            and overlap / (actor[2] - actor[0]) >= self.config.platform_min_horizontal_overlap_ratio
        )

    def _geometry(
        self,
        belief: BeliefState,
        actor: TrackBelief,
        terrain: tuple[TrackBelief, ...],
        *,
        rope: bool,
    ) -> tuple[TrackBelief | None, str]:
        candidates = tuple(
            detection
            for detection in belief.observation.detections
            if (
                detection.class_id in _ROPES and self._near_rope(actor.bbox, detection.bbox)
                if rope
                else detection.class_id == "platform" and self._support(actor.bbox, detection.bbox)
            )
        )
        label = "rope" if rope else "platform"
        if not candidates:
            return None, f"missing_current_{label}_geometry"
        if len(candidates) != 1:
            return None, f"ambiguous_{label}_geometry"
        if candidates[0].confidence < self.config.min_terrain_confidence:
            return None, f"low_confidence_{label}_geometry"
        matched = tuple(track for track in terrain if _same_measurement(track, candidates[0]))
        if len(matched) != 1:
            return None, f"uncertain_{label}_track"
        return matched[0], ""

    def _camera(self, previous: _Sample, current: _Sample) -> _Camera:
        old = {track.track_id: track for track in previous.terrain}
        landmarks: list[_Landmark] = []
        rejected: list[str] = []
        for track in current.terrain:
            before = old.get(track.track_id)
            if before is None or track.association != "matched":
                rejected.append(track.track_id)
                continue
            if before.class_id != track.class_id or _size(before.bbox) != _size(track.bbox):
                rejected.append(track.track_id)
                continue
            landmarks.append(
                _Landmark(
                    track.track_id,
                    track.class_id,
                    before.bbox,
                    track.bbox,
                    _subtract(_center(track.bbox), _center(before.bbox)),
                )
            )
        reason = "consistent_terrain_translation"
        delta: Pair | None = None
        if len(landmarks) < self.config.camera_min_landmarks:
            reason = "insufficient_camera_landmarks"
        elif any(
            hypot(*_subtract(first.delta, second.delta)) > self.config.camera_delta_tolerance_px
            for index, first in enumerate(landmarks)
            for second in landmarks[index + 1 :]
        ):
            reason = "inconsistent_camera_landmarks"
        else:
            delta = (
                fsum(landmark.delta[0] for landmark in landmarks) / len(landmarks),
                fsum(landmark.delta[1] for landmark in landmarks) / len(landmarks),
            )
            if hypot(*delta) > self.config.max_camera_translation_px:
                reason, delta = "camera_translation_exceeds_budget", None
        return _Camera(
            "supported_translation" if delta is not None else "unknown",
            reason,
            delta,
            tuple(landmarks),
            tuple(rejected),
            previous.frame,
            current.frame,
        )

    def _discontinuity(self, belief: BeliefState, frame: _Frame) -> str:
        if self._last_now_ms is not None and belief.now_ms < self._last_now_ms:
            return "processing_clock_rollback"
        previous = self._last_frame
        if previous is None:
            return ""
        if frame.source_id != previous.source_id:
            return "source_changed"
        if frame.epoch != previous.epoch:
            return "belief_epoch_changed"
        if frame.source_sequence != previous.source_sequence + 1:
            return "sequence_discontinuity"
        elapsed = frame.timestamp_ms - previous.timestamp_ms
        if elapsed <= 0:
            return "nonincreasing_frame_time"
        if elapsed > self.config.max_gap_ms:
            return "frame_time_gap"
        return ""

    def update(self, belief: BeliefState) -> MovementHypothesis:
        if type(belief) is not BeliefState:
            raise ValueError("movement input must be BeliefState")
        belief.__post_init__()
        observation = belief.observation
        if len(observation.detections) > 512:
            raise ValueError("movement input exceeds the detection budget")
        frame = _Frame(
            observation.source_id, observation.sequence, observation.timestamp_ms, belief.epoch
        )
        discontinuity = self._discontinuity(belief, frame)
        self._last_frame, self._last_now_ms = frame, belief.now_ms
        actor: TrackBelief | None = None
        rope: TrackBelief | None = None
        platform: TrackBelief | None = None
        camera = _Camera()
        motion: _Motion | None = None

        def result(
            state: MovementState, reasons: tuple[str, ...], evidence: tuple[_Frame, ...] = ()
        ) -> MovementHypothesis:
            return MovementHypothesis(
                state,
                frame,
                belief.now_ms,
                actor.track_id if actor else None,
                rope.track_id if rope else None,
                platform.track_id if platform else None,
                camera,
                motion,
                evidence or (frame,),
                reasons,
            )

        invalid = ""
        if observation.timestamp_ms > belief.now_ms:
            invalid = "future_frame"
        elif belief.now_ms - observation.timestamp_ms > self.config.max_age_ms:
            invalid = "stale_frame"
        elif observation.status is Status.STALE:
            invalid = "stale_observation"
        elif belief.reason not in {"fresh_observation", "fresh_detection_hypotheses"}:
            invalid = "unusable_current_belief"
        if invalid:
            self._reset_chain()
            self._previous = None
            return result("unknown", (invalid,))
        actor, actor_reason = self._actor(belief)
        if actor is None:
            self._reset_chain()
            self._previous = None
            return result("unknown", (actor_reason,))
        terrain = self._terrain(belief)
        platform, platform_reason = self._geometry(belief, actor, terrain, rope=False)
        rope, rope_reason = self._geometry(belief, actor, terrain, rope=True)
        previous = self._previous
        # A short step off the rope can finish a supported arrival. The same rope
        # must still be freshly visible; disappearance does not preserve a token.
        if (
            rope is None
            and rope_reason == "missing_current_rope_geometry"
            and platform is not None
            and self._last_climb_ms is not None
            and previous is not None
            and previous.rope is not None
        ):
            rope = next(
                (track for track in terrain if track.track_id == previous.rope.track_id), None
            )
        current = _Sample(frame, actor, terrain, rope)
        self._previous = current
        if (
            previous is None
            or discontinuity
            or actor.association == "new"
            or previous.actor.track_id != actor.track_id
        ):
            self._reset_chain()
            return result("unknown", (discontinuity or "new_actor_requires_buildup",))
        if _size(previous.actor.bbox) != _size(actor.bbox):
            self._reset_chain()
            return result("unknown", ("actor_box_size_changed",))
        camera = self._camera(previous, current)
        if camera.delta is None:
            self._reset_chain()
            return result("unknown", (camera.reason,))
        if rope is None:
            self._reset_chain()
            return result("unknown", (rope_reason,))
        if (
            previous.rope is None
            or previous.rope.track_id != rope.track_id
            or rope.association != "matched"
            or _size(previous.rope.bbox) != _size(rope.bbox)
        ):
            self._reset_chain()
            return result("unknown", ("rope_changed_or_requires_buildup",))
        before = _Pose(
            previous.frame, _subtract(_feet(previous.actor.bbox), _center(previous.rope.bbox))
        )
        pose = _Pose(frame, _subtract(_feet(actor.bbox), _center(rope.bbox)))
        elapsed = frame.timestamp_ms - previous.frame.timestamp_ms
        screen_delta = _subtract(_feet(actor.bbox), _feet(previous.actor.bbox))
        corrected_delta = _subtract(screen_delta, camera.delta)
        relative_delta = _subtract(pose.relative, before.relative)
        if (
            hypot(*_subtract(corrected_delta, relative_delta))
            > self.config.camera_delta_tolerance_px
        ):
            self._reset_chain()
            return result("unknown", ("rope_motion_conflicts_with_camera",))
        if abs(relative_delta[1]) * 1000 / elapsed > self.config.max_vertical_speed_px_per_second:
            self._reset_chain()
            return result("unknown", ("relative_vertical_speed_exceeds_budget",))
        near = self._near_rope(actor.bbox, rope.bbox)
        aligned = (
            near
            and abs(pose.relative[0]) <= self.config.rope_alignment_tolerance_px
            and min(actor.bbox[3], rope.bbox[3]) > max(actor.bbox[1], rope.bbox[1])
        )
        if self._last_climb_ms is not None and (
            frame.timestamp_ms - self._last_climb_ms > self.config.max_climb_to_arrival_ms
        ):
            self._reset_chain()
        direction = 1 if relative_delta[1] > 0 else -1
        moving = abs(relative_delta[1]) >= self.config.min_vertical_step_px
        if moving and self._climb_direction and direction != self._climb_direction:
            self._reset_chain()
        arrival_reason = self._update_arrival(
            platform, platform_reason, actor, previous.actor, frame, camera
        )
        arrival_evidence = {
            item.source_sequence: item
            for item in self._climb_frames + tuple(p.frame for p in self._arrival)
        }
        if len(arrival_evidence) > self.config.history_size:
            self._reset_chain()
            arrival_reason = "climb_and_arrival_evidence_exceeds_history_budget"
        arrived = bool(
            self._arrival
            and self._last_climb_ms is not None
            and len(self._arrival) >= self.config.min_arrival_samples
            and frame.timestamp_ms - self._arrival[0].frame.timestamp_ms
            >= self.config.min_arrival_duration_ms
        )
        if not aligned or not moving:
            self._motion.clear()
            self._motion.append(pose)
            self._direction = 0
            if (
                not self._arrival
                and self._last_climb_relative is not None
                and (
                    abs(pose.relative[0] - self._last_climb_relative[0])
                    > self.config.max_horizontal_drift_px
                    or abs(relative_delta[0]) > self.config.stability_tolerance_px
                    or not near
                )
            ):
                self._reset_chain()
        else:
            if self._direction and direction != self._direction:
                self._reset_chain()
            if not self._motion or self._motion[-1].frame != previous.frame:
                self._motion.clear()
                self._motion.append(before)
            self._direction = direction
            self._motion.append(pose)
            horizontal = max(item.relative[0] for item in self._motion) - min(
                item.relative[0] for item in self._motion
            )
            if horizontal > self.config.max_horizontal_drift_px:
                self._reset_chain()
                self._motion.append(pose)
                arrival_reason = "horizontal_drift_exceeds_budget"
        duration = frame.timestamp_ms - self._motion[0].frame.timestamp_ms if self._motion else 0
        vertical = abs(pose.relative[1] - self._motion[0].relative[1]) if self._motion else 0.0
        horizontal = (
            max(item.relative[0] for item in self._motion)
            - min(item.relative[0] for item in self._motion)
            if self._motion
            else 0.0
        )
        motion = _Motion(
            elapsed,
            screen_delta,
            corrected_delta,
            relative_delta,
            "down" if moving and direction > 0 else "up" if moving else "stationary_or_jitter",
            len(self._motion),
            duration,
            vertical,
            horizontal,
        )
        if arrived and self._last_climb_ms is not None:
            evidence = tuple(arrival_evidence[key] for key in sorted(arrival_evidence))
            return result(
                "platform_arrival_candidate",
                ("prior_climb_and_sustained_platform_support",),
                evidence,
            )
        if (
            aligned
            and moving
            and len(self._motion) >= self.config.min_climb_samples
            and duration >= self.config.min_climb_duration_ms
            and vertical >= self.config.min_climb_distance_px
        ):
            climb_poses = tuple(self._motion)
            # Retain the shortest complete qualifying climb segment for arrival,
            # while this frame's motion report keeps its full bounded history.
            for start in range(len(climb_poses) - self.config.min_climb_samples, -1, -1):
                first = climb_poses[start]
                if (
                    frame.timestamp_ms - first.frame.timestamp_ms
                    >= self.config.min_climb_duration_ms
                    and abs(pose.relative[1] - first.relative[1])
                    >= self.config.min_climb_distance_px
                ):
                    self._climb_frames = tuple(item.frame for item in climb_poses[start:])
                    break
            self._last_climb_ms = frame.timestamp_ms
            self._last_climb_relative = pose.relative
            self._climb_direction = direction
            return result(
                "climbing_candidate",
                ("sustained_aligned_relative_vertical_motion",),
                tuple(item.frame for item in climb_poses),
            )
        if not near:
            return result("unknown", (arrival_reason or "actor_not_near_rope",))
        return result(
            "near_rope",
            (arrival_reason or "spatial_proximity_without_sustained_climb_evidence",),
            tuple(item.frame for item in self._motion) or (previous.frame, frame),
        )

    def _update_arrival(
        self,
        platform: TrackBelief | None,
        reason: str,
        actor: TrackBelief,
        previous_actor: TrackBelief,
        frame: _Frame,
        camera: _Camera,
    ) -> str:
        if self._last_climb_ms is None:
            return ""
        if platform is None:
            if self._arrival or reason != "missing_current_platform_geometry":
                self._reset_chain()
                return "platform_support_lost_or_uncertain"
            return ""
        if platform.association != "matched":
            self._reset_chain()
            return "platform_requires_buildup"
        landmark = next(
            (item for item in camera.landmarks if item.track_id == platform.track_id), None
        )
        if landmark is None:
            self._reset_chain()
            return "platform_camera_geometry_uncertain"
        if self._arrival_platform_id is not None and platform.track_id != self._arrival_platform_id:
            self._reset_chain()
            return "platform_support_identity_changed"
        relative = (
            _center(actor.bbox)[0] - _center(platform.bbox)[0],
            actor.bbox[3] - platform.bbox[1],
        )
        if not self._arrival:
            previous_relative = (
                _center(previous_actor.bbox)[0] - _center(landmark.previous_bbox)[0],
                previous_actor.bbox[3] - landmark.previous_bbox[1],
            )
            if (
                abs(relative[0] - previous_relative[0]) > self.config.stability_tolerance_px
                or abs(relative[1] - previous_relative[1]) >= self.config.min_vertical_step_px
            ):
                # Entering the top tolerance while still climbing is not settled
                # support. Preserve that climb and wait for a stable incoming step.
                return "platform_support_requires_stable_step"
        self._arrival_platform_id = platform.track_id
        self._arrival.append(_Pose(frame, relative))
        if any(
            max(item.relative[axis] for item in self._arrival)
            - min(item.relative[axis] for item in self._arrival)
            > self.config.stability_tolerance_px
            for axis in (0, 1)
        ):
            self._reset_chain()
            return "platform_relative_stability_lost"
        return "platform_support_requires_sustained_stability"
