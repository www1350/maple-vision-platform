"""Bounded, observation-only temporal state with explicit retained hypotheses.

Track IDs describe a deterministic geometric association, not a verified game
identity. Cached fields and missed tracks never enter the actionable observation.
"""

import re
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import Literal, cast

from .contracts import Status
from .vision import VisualObservation

_MAX_TIME = 2**53 - 1
_MAX_TTL = 120_000
_MAX_COORD = 1_000_000
_FIELD_NAMES = (
    "player_bbox",
    "hp_ratio",
    "mp_ratio",
    "map_id",
    "dialogue_visible",
    "inventory",
    "skills",
    "buffs",
)
_TRACK_ID = re.compile(r"track-[1-9][0-9]{0,15}-[1-9][0-9]{0,15}\Z")
type Box = tuple[float, float, float, float]
type FieldValue = str | float | int | bool | Box | None
type Association = Literal["new", "matched", "reacquired", "ambiguous", "unobserved"]


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value: object, name: str, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{name} must be bounded nonempty text without control characters")
    return value


def _box(value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError("bbox must be an immutable xyxy tuple")
    for coordinate in value:
        _number(coordinate, "bbox coordinate", 0, _MAX_COORD)
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")


def _clock(value: object, name: str) -> int:
    # Reserve exact-JSON integer headroom for bounded expiries.
    return _integer(value, name, 0, _MAX_TIME - _MAX_TTL)


@dataclass(frozen=True)
class StateConfig:
    max_age_ms: int = 250
    field_ttl_ms: int = 1000
    track_ttl_ms: int = 500
    max_tracks: int = 128
    max_detections: int = 128
    min_confidence: float = 0.5
    max_match_distance: float = 48.0
    ambiguity_margin: float = 0.15

    def __post_init__(self) -> None:
        _integer(self.max_age_ms, "max_age_ms", 0, 60_000)
        _integer(self.field_ttl_ms, "field_ttl_ms", 1, _MAX_TTL)
        _integer(self.track_ttl_ms, "track_ttl_ms", 1, _MAX_TTL)
        if min(self.field_ttl_ms, self.track_ttl_ms) <= self.max_age_ms:
            raise ValueError("field and track TTL must exceed max_age_ms")
        _integer(self.max_tracks, "max_tracks", 1, 512)
        _integer(self.max_detections, "max_detections", 1, 256)
        if self.max_detections > self.max_tracks:
            raise ValueError("max_detections must not exceed max_tracks")
        for name, low, high in (
            ("min_confidence", 0.0, 1.0),
            ("max_match_distance", 0.01, 8192.0),
            ("ambiguity_margin", 0.0, 2.0),
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "max_age_ms": self.max_age_ms,
            "field_ttl_ms": self.field_ttl_ms,
            "track_ttl_ms": self.track_ttl_ms,
            "max_tracks": self.max_tracks,
            "max_detections": self.max_detections,
            "min_confidence": self.min_confidence,
            "max_match_distance": self.max_match_distance,
            "ambiguity_margin": self.ambiguity_margin,
        }

    @classmethod
    def from_dict(cls, value: object) -> "StateConfig":
        keys = {
            "schema_version",
            "max_age_ms",
            "field_ttl_ms",
            "track_ttl_ms",
            "max_tracks",
            "max_detections",
            "min_confidence",
            "max_match_distance",
            "ambiguity_margin",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("invalid state configuration fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported state configuration schema")
        return cls(
            max_age_ms=value["max_age_ms"],
            field_ttl_ms=value["field_ttl_ms"],
            track_ttl_ms=value["track_ttl_ms"],
            max_tracks=value["max_tracks"],
            max_detections=value["max_detections"],
            min_confidence=value["min_confidence"],
            max_match_distance=value["max_match_distance"],
            ambiguity_margin=value["ambiguity_margin"],
        )


@dataclass(frozen=True)
class FieldBelief:
    name: str
    value: FieldValue = None
    status: Status = Status.UNKNOWN
    confidence: float = 0.0
    source_id: str | None = None
    source_sequence: int | None = None
    observed_at_ms: int | None = None
    expires_at_ms: int | None = None
    observed_this_frame: bool = False

    def __post_init__(self) -> None:
        if self.name not in _FIELD_NAMES or not isinstance(self.status, Status):
            raise ValueError("invalid belief field name/status")
        _number(self.confidence, "field confidence", 0, 1)
        if type(self.observed_this_frame) is not bool:
            raise ValueError("observed_this_frame must be boolean")
        metadata = (self.source_id, self.source_sequence, self.observed_at_ms, self.expires_at_ms)
        if all(item is None for item in metadata):
            if (
                self.value is not None
                or self.status is not Status.UNKNOWN
                or self.confidence != 0
                or self.observed_this_frame
            ):
                raise ValueError("unmeasured fields must remain unknown")
            return
        if any(item is None for item in metadata):
            raise ValueError("field provenance must be complete")
        _text(self.source_id, "field source", 256)
        _clock(self.source_sequence, "field source sequence")
        measured = _clock(self.observed_at_ms, "field measurement time")
        expiry = _integer(self.expires_at_ms, "field expiry", 1, _MAX_TIME)
        if not measured < expiry <= measured + _MAX_TTL:
            raise ValueError("field expiry must follow measurement within TTL budget")
        if self.status is Status.STALE:
            if self.value is not None or self.observed_this_frame or self.confidence != 0:
                raise ValueError("expired field must clear value and confidence")
            return
        if self.status is not Status.KNOWN or self.value is None:
            raise ValueError("measured field must be known or expired")
        if self.name in {"inventory", "skills", "buffs"}:
            raise ValueError("unmeasured gameplay domains cannot be invented")
        if self.name == "player_bbox":
            _box(self.value)
        elif self.name in {"hp_ratio", "mp_ratio"}:
            _number(self.value, self.name, 0, 1)
        elif self.name == "map_id":
            _text(self.value, "map_id", 80)
        elif type(self.value) is not bool:
            raise ValueError("dialogue_visible must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "source_sequence": self.source_sequence,
            "observed_at_ms": self.observed_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "observed_this_frame": self.observed_this_frame,
        }


@dataclass(frozen=True)
class TrackBelief:
    track_id: str
    class_id: str
    bbox: Box
    confidence: float
    source_id: str
    source_sequence: int
    observed_at_ms: int
    expires_at_ms: int
    observed_this_frame: bool
    association: Association
    identity_uncertain: bool = False
    candidate_ids: tuple[str, ...] = ()
    velocity_px_per_second: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or _TRACK_ID.fullmatch(self.track_id) is None:
            raise ValueError("invalid track id")
        _text(self.class_id, "track class", 64)
        _box(self.bbox)
        _number(self.confidence, "track confidence", 0, 1)
        _text(self.source_id, "track source", 256)
        _clock(self.source_sequence, "track sequence")
        measured = _clock(self.observed_at_ms, "track time")
        expiry = _integer(self.expires_at_ms, "track expiry", 1, _MAX_TIME)
        if not measured < expiry <= measured + _MAX_TTL:
            raise ValueError("invalid track expiry")
        if type(self.observed_this_frame) is not bool or type(self.identity_uncertain) is not bool:
            raise ValueError("track flags must be boolean")
        if self.association not in {"new", "matched", "reacquired", "ambiguous", "unobserved"}:
            raise ValueError("invalid track association")
        if (self.association == "unobserved") == self.observed_this_frame:
            raise ValueError("track observation and association disagree")
        if self.association == "ambiguous" and not self.identity_uncertain:
            raise ValueError("ambiguous association must remain uncertain")
        if (
            not isinstance(self.candidate_ids, tuple)
            or len(self.candidate_ids) > 512
            or any(not isinstance(v, str) or not _TRACK_ID.fullmatch(v) for v in self.candidate_ids)
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
        ):
            raise ValueError("invalid candidate track identities")
        if self.candidate_ids and not self.identity_uncertain:
            raise ValueError("identity candidates require uncertainty")
        if self.velocity_px_per_second is not None:
            if (
                not isinstance(self.velocity_px_per_second, tuple)
                or len(self.velocity_px_per_second) != 2
            ):
                raise ValueError("track velocity must be an immutable pair")
            for value in self.velocity_px_per_second:
                _number(value, "track velocity", -1e9, 1e9)

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "source_id": self.source_id,
            "source_sequence": self.source_sequence,
            "observed_at_ms": self.observed_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "observed_this_frame": self.observed_this_frame,
            "association": self.association,
            "identity_uncertain": self.identity_uncertain,
            "candidate_ids": list(self.candidate_ids),
            "velocity_px_per_second": list(self.velocity_px_per_second)
            if self.velocity_px_per_second is not None
            else None,
        }


def _observation_dict(observation: VisualObservation) -> dict[str, object]:
    return {
        "sequence": observation.sequence,
        "timestamp_ms": observation.timestamp_ms,
        "status": observation.status.value,
        "confidence": observation.confidence,
        "detections": [
            {"class_id": d.class_id, "bbox": list(d.bbox), "confidence": d.confidence}
            for d in observation.detections
        ],
        "hp_ratio": observation.hp_ratio,
        "mp_ratio": observation.mp_ratio,
        "map_id": observation.map_id,
        "dialogue_visible": observation.dialogue_visible,
        "source_id": observation.source_id,
    }


@dataclass(frozen=True)
class BeliefState:
    observation: VisualObservation
    actionable: bool
    reason: str
    now_ms: int
    epoch: int
    fields: tuple[FieldBelief, ...]
    tracks: tuple[TrackBelief, ...]
    last_accepted_sequence: int | None
    last_accepted_timestamp_ms: int | None

    def __post_init__(self) -> None:
        if type(self.observation) is not VisualObservation or type(self.actionable) is not bool:
            raise ValueError("invalid belief observation or actionable flag")
        _text(self.reason, "belief reason", 128)
        _clock(self.now_ms, "belief clock")
        _clock(self.epoch, "belief epoch")
        if (
            not isinstance(self.fields, tuple)
            or any(not isinstance(field, FieldBelief) for field in self.fields)
            or tuple(field.name for field in self.fields) != _FIELD_NAMES
        ):
            raise ValueError("belief fields must contain each domain in canonical order")
        if (
            not isinstance(self.tracks, tuple)
            or len(self.tracks) > 512
            or any(not isinstance(track, TrackBelief) for track in self.tracks)
            or len({track.track_id for track in self.tracks}) != len(self.tracks)
        ):
            raise ValueError("belief tracks must be bounded immutable unique identities")
        if (self.last_accepted_sequence is None) != (self.last_accepted_timestamp_ms is None):
            raise ValueError("accepted frame watermark must be complete")
        if self.last_accepted_sequence is not None:
            _clock(self.last_accepted_sequence, "accepted sequence")
            _clock(self.last_accepted_timestamp_ms, "accepted timestamp")
        if self.actionable and (
            self.observation.status is not Status.KNOWN
            or self.observation.timestamp_ms > self.now_ms
            or self.last_accepted_sequence != self.observation.sequence
            or self.last_accepted_timestamp_ms != self.observation.timestamp_ms
            or self.observation.map_id is None
            or self.observation.hp_ratio is None
            or self.observation.hp_ratio <= 0
            or self.observation.mp_ratio is None
            or len([d for d in self.observation.detections if d.class_id == "player"]) != 1
        ):
            raise ValueError("actionable belief must represent an accepted usable observation")

    def field(self, name: str) -> FieldBelief:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "observation": _observation_dict(self.observation),
            "actionable": self.actionable,
            "reason": self.reason,
            "now_ms": self.now_ms,
            "epoch": self.epoch,
            "fields": [field.to_dict() for field in self.fields],
            "tracks": [track.to_dict() for track in self.tracks],
            "last_accepted_sequence": self.last_accepted_sequence,
            "last_accepted_timestamp_ms": self.last_accepted_timestamp_ms,
        }


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _iou(first: Box, second: Box) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union


def _predicted_box(track: TrackBelief, timestamp_ms: int) -> Box:
    if track.velocity_px_per_second is None:
        return track.bbox
    dt = (timestamp_ms - track.observed_at_ms) / 1000
    dx, dy = (v * dt for v in track.velocity_px_per_second)
    x1, y1, x2, y2 = track.bbox
    # Prediction only helps association. It never becomes a measured detection.
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


class TemporalStateEstimator:
    def __init__(
        self,
        config: StateConfig | None = None,
        *,
        retain_detection_hypotheses: bool = False,
        retain_multiple_player_hypotheses: bool = False,
    ) -> None:
        if config is not None and type(config) is not StateConfig:
            raise ValueError("state config must be StateConfig")
        if type(retain_detection_hypotheses) is not bool:
            raise ValueError("retain_detection_hypotheses must be boolean")
        if type(retain_multiple_player_hypotheses) is not bool:
            raise ValueError("retain_multiple_player_hypotheses must be boolean")
        if retain_multiple_player_hypotheses and not retain_detection_hypotheses:
            raise ValueError(
                "retain_multiple_player_hypotheses requires retain_detection_hypotheses"
            )
        self.config = config or StateConfig()
        self.retain_detection_hypotheses = retain_detection_hypotheses
        self.retain_multiple_player_hypotheses = retain_multiple_player_hypotheses
        self._fields = {name: FieldBelief(name) for name in _FIELD_NAMES}
        self._tracks: dict[str, TrackBelief] = {}
        self._last_sequence: int | None = None
        self._last_timestamp_ms: int | None = None
        self._last_hypothesis_sequence: int | None = None
        self._last_hypothesis_timestamp_ms: int | None = None
        self._last_now_ms: int | None = None
        self._clock_failed = False
        self._context: tuple[str, str] | None = None
        self._epoch = 0
        self._next_track = 1

    def update(self, observation: VisualObservation, now_ms: int) -> BeliefState:
        self._validate_input(observation, now_ms)
        if self._last_now_ms is not None and now_ms < self._last_now_ms:
            self._clock_failed = True
            self._tracks.clear()
            self._fields = {name: FieldBelief(name) for name in _FIELD_NAMES}
        if self._clock_failed:
            return self._result(observation, now_ms, "clock_rollback", Status.UNKNOWN)
        self._last_now_ms = now_ms
        self._age(now_ms)
        reason, status = self._rejection(observation, now_ms)
        if reason:
            if reason == "unknown_observation" and self.retain_multiple_player_hypotheses:
                # This mode retains every visible actor, not a previously assumed
                # self. Only a unique current raw player may remeasure this field.
                # Clear it even if this frame has no usable detections at all.
                self._fields["player_bbox"] = FieldBelief("player_bbox")
            if reason == "unknown_observation" and self._has_detection_hypotheses(observation):
                self._measure_detection_hypotheses(observation)
                return self._result(
                    observation, now_ms, "fresh_detection_hypotheses", Status.UNKNOWN
                )
            return self._result(observation, now_ms, reason, status)
        assert observation.map_id is not None
        context = (observation.source_id, observation.map_id)
        if context != self._context:
            self._epoch += 1
            _clock(self._epoch, "state epoch")
            self._context = context
            self._tracks.clear()
            self._fields = {name: FieldBelief(name) for name in _FIELD_NAMES}
            self._next_track = 1
        self._measure_fields(observation)
        self._associate(observation)
        self._last_sequence = observation.sequence
        self._last_timestamp_ms = observation.timestamp_ms
        self._last_hypothesis_sequence = observation.sequence
        self._last_hypothesis_timestamp_ms = observation.timestamp_ms
        return self._result(observation, now_ms, "fresh_observation", None)

    def _validate_input(self, observation: VisualObservation, now_ms: int) -> None:
        _clock(now_ms, "state clock")
        if type(observation) is not VisualObservation:
            raise ValueError("state input must be VisualObservation")
        if len(observation.detections) > self.config.max_detections:
            raise ValueError("observation exceeds state detection budget")
        observation.__post_init__()
        _clock(observation.sequence, "observation sequence")
        _clock(observation.timestamp_ms, "observation timestamp")
        _text(observation.source_id, "observation source", 256)
        if observation.map_id is not None:
            _text(observation.map_id, "observation map", 80)
        for detection in observation.detections:
            _text(detection.class_id, "detection class", 64)
            _box(detection.bbox)
            _number(detection.confidence, "detection confidence", 0, 1)

    def _rejection(self, observation: VisualObservation, now_ms: int) -> tuple[str, Status]:
        if observation.timestamp_ms > now_ms:
            return "future_observation", Status.UNKNOWN
        if now_ms - observation.timestamp_ms > self.config.max_age_ms:
            return "stale_observation", Status.STALE
        if (
            self._last_hypothesis_sequence is not None
            and observation.sequence <= self._last_hypothesis_sequence
        ):
            return "duplicate_or_reordered_observation", Status.STALE
        if (
            self._last_hypothesis_timestamp_ms is not None
            and observation.timestamp_ms < self._last_hypothesis_timestamp_ms
        ):
            return "observation_timestamp_rollback", Status.STALE
        if observation.status is not Status.KNOWN:
            return "unknown_observation", Status.UNKNOWN
        if observation.confidence < self.config.min_confidence:
            return "low_observation_confidence", Status.UNKNOWN
        players = [d for d in observation.detections if d.class_id == "player"]
        if len(players) != 1:
            return "ambiguous_player", Status.UNKNOWN
        if players[0].confidence < self.config.min_confidence:
            return "low_player_confidence", Status.UNKNOWN
        if observation.map_id is None:
            return "unknown_map", Status.UNKNOWN
        if observation.hp_ratio is None or observation.mp_ratio is None:
            return "unknown_resources", Status.UNKNOWN
        if observation.hp_ratio <= 0:
            return "player_dead", Status.UNKNOWN
        return "", Status.KNOWN

    def _has_detection_hypotheses(self, observation: VisualObservation) -> bool:
        if not self.retain_detection_hypotheses:
            return False
        usable = tuple(
            detection
            for detection in observation.detections
            if detection.confidence >= self.config.min_confidence
        )
        players = tuple(item for item in usable if item.class_id == "player")
        return bool(usable) and (self.retain_multiple_player_hypotheses or len(players) <= 1)

    def _measure_detection_hypotheses(self, observation: VisualObservation) -> None:
        """Retain visible boxes without upgrading an incomplete frame to actionable state."""
        context = (observation.source_id, "<unknown-map>")
        if context != self._context:
            self._epoch += 1
            _clock(self._epoch, "state epoch")
            self._context = context
            self._tracks.clear()
            self._fields = {name: FieldBelief(name) for name in _FIELD_NAMES}
            self._next_track = 1
        raw_players = tuple(
            detection for detection in observation.detections if detection.class_id == "player"
        )
        players = tuple(
            detection
            for detection in raw_players
            if detection.confidence >= self.config.min_confidence
        )
        if len(players) == 1 and (
            not self.retain_multiple_player_hypotheses or len(raw_players) == 1
        ):
            player = players[0]
            self._fields["player_bbox"] = FieldBelief(
                name="player_bbox",
                value=player.bbox,
                status=Status.KNOWN,
                confidence=player.confidence,
                source_id=observation.source_id,
                source_sequence=observation.sequence,
                observed_at_ms=observation.timestamp_ms,
                expires_at_ms=observation.timestamp_ms + self.config.field_ttl_ms,
                observed_this_frame=True,
            )
        self._associate(observation)
        self._last_hypothesis_sequence = observation.sequence
        self._last_hypothesis_timestamp_ms = observation.timestamp_ms

    def _age(self, now_ms: int) -> None:
        for name, field in self._fields.items():
            if field.expires_at_ms is None:
                continue
            self._fields[name] = (
                replace(
                    field,
                    value=None,
                    status=Status.STALE,
                    confidence=0.0,
                    observed_this_frame=False,
                )
                if now_ms >= field.expires_at_ms
                else replace(field, observed_this_frame=False)
            )
        self._tracks = {
            key: replace(track, observed_this_frame=False, association="unobserved")
            for key, track in self._tracks.items()
            if now_ms < track.expires_at_ms
        }

    def _measure_fields(self, observation: VisualObservation) -> None:
        player = next(d for d in observation.detections if d.class_id == "player")
        values: dict[str, FieldValue] = {
            "player_bbox": player.bbox,
            "hp_ratio": observation.hp_ratio,
            "mp_ratio": observation.mp_ratio,
            "map_id": observation.map_id,
            "dialogue_visible": observation.dialogue_visible,
        }
        for name, value in values.items():
            self._fields[name] = FieldBelief(
                name=name,
                value=value,
                status=Status.KNOWN,
                confidence=player.confidence if name == "player_bbox" else observation.confidence,
                source_id=observation.source_id,
                source_sequence=observation.sequence,
                observed_at_ms=observation.timestamp_ms,
                expires_at_ms=observation.timestamp_ms + self.config.field_ttl_ms,
                observed_this_frame=True,
            )

    def _associate(self, observation: VisualObservation) -> None:
        detections = sorted(
            (d for d in observation.detections if d.confidence >= self.config.min_confidence),
            key=lambda d: (d.class_id, d.bbox, d.confidence),
        )
        old = self._tracks
        candidates: list[list[tuple[float, str]]] = [[] for _ in detections]
        reverse: dict[str, list[tuple[float, int]]] = {key: [] for key in old}
        for index, detection in enumerate(detections):
            cx, cy = _center(detection.bbox)
            for track_id, previous_track in old.items():
                if previous_track.class_id != detection.class_id:
                    continue
                prediction = _predicted_box(previous_track, observation.timestamp_ms)
                px, py = _center(prediction)
                distance = hypot(cx - px, cy - py)
                if distance > self.config.max_match_distance:
                    continue
                cost = distance / self.config.max_match_distance + 0.25 * (
                    1.0 - _iou(prediction, detection.bbox)
                )
                candidates[index].append((cost, track_id))
                reverse[track_id].append((cost, index))
        for options in candidates:
            options.sort()
        for options_by_track in reverse.values():
            options_by_track.sort()
        current_overlap: set[int] = set()
        for first, detection in enumerate(detections):
            for second in range(first + 1, len(detections)):
                other = detections[second]
                if detection.class_id == other.class_id and _iou(detection.bbox, other.bbox) >= 0.5:
                    current_overlap.update((first, second))
        observed: dict[str, TrackBelief] = {}
        contested: set[str] = set()
        for index, detection in enumerate(detections):
            options = candidates[index]
            track: TrackBelief | None = None
            if options and index not in current_overlap:
                cost, track_id = options[0]
                alternatives = reverse[track_id]
                if (
                    (len(options) == 1 or options[1][0] - cost > self.config.ambiguity_margin)
                    and alternatives[0][1] == index
                    and (
                        len(alternatives) == 1
                        or alternatives[1][0] - cost > self.config.ambiguity_margin
                    )
                ):
                    track = old[track_id]
            if track is None:
                ambiguous = bool(options) or index in current_overlap
                identities = tuple(sorted(track_id for _, track_id in options))
                contested.update(identities)
                track_id = f"track-{self._epoch}-{self._next_track}"
                self._next_track += 1
                _clock(self._next_track, "track identity budget")
                association: Association = "ambiguous" if ambiguous else "new"
                velocity = None
                uncertain = ambiguous
            else:
                track_id = track.track_id
                # Ageing marks hypotheses unobserved. Consecutive accepted source
                # sequences distinguish a normal update from missing observations.
                association = (
                    "matched" if track.source_sequence + 1 == observation.sequence else "reacquired"
                )
                uncertain, identities = track.identity_uncertain, track.candidate_ids
                elapsed_ms = observation.timestamp_ms - track.observed_at_ms
                previous, current = _center(track.bbox), _center(detection.bbox)
                velocity = (
                    tuple(
                        (new - prior) * 1000 / elapsed_ms for prior, new in zip(previous, current)
                    )
                    if elapsed_ms > 0
                    else None
                )
            observed[track_id] = TrackBelief(
                track_id=track_id,
                class_id=detection.class_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                source_id=observation.source_id,
                source_sequence=observation.sequence,
                observed_at_ms=observation.timestamp_ms,
                expires_at_ms=observation.timestamp_ms + self.config.track_ttl_ms,
                observed_this_frame=True,
                association=association,
                identity_uncertain=uncertain,
                candidate_ids=identities,
                velocity_px_per_second=cast(tuple[float, float] | None, velocity),
            )
        missing = [
            replace(track, identity_uncertain=track.identity_uncertain or track_id in contested)
            for track_id, track in old.items()
            if track_id not in observed
        ]
        missing.sort(key=lambda track: (-track.observed_at_ms, track.track_id))
        kept = list(observed.values()) + missing[: self.config.max_tracks - len(observed)]
        self._tracks = {track.track_id: track for track in sorted(kept, key=lambda t: t.track_id)}

    def _result(
        self, observation: VisualObservation, now_ms: int, reason: str, status: Status | None
    ) -> BeliefState:
        normalized = observation if status is None else replace(observation, status=status)
        return BeliefState(
            observation=normalized,
            actionable=status is None,
            reason=reason,
            now_ms=now_ms,
            epoch=self._epoch,
            fields=tuple(self._fields[name] for name in _FIELD_NAMES),
            tracks=tuple(self._tracks.values()),
            last_accepted_sequence=self._last_sequence,
            last_accepted_timestamp_ms=self._last_timestamp_ms,
        )
