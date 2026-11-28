"""Read-only, window-local tracking of imported pixels with independent reinference.

Track identities are hypotheses within one imported window. A replay proves
reproducibility against those pixels, not detection accuracy or action success.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pillow_version

from .actor_selection import (
    ACTOR_SELECTION_ALGORITHM_VERSION,
    ActorAnchor,
    ActorSelectionConfig,
    ActorSelector,
)
from .catalog import canonical_json, strict_json
from .desktop_observation import DetectorOnlyPerception, FramePerception
from .evaluation import _detector_metadata
from .local_files import publish_new
from .movement_hypotheses import (
    MOVEMENT_ALGORITHM_VERSION,
    MovementConfig,
    MovementHypothesisEstimator,
)
from .onnx_detector import OnnxDetector
from .recordings import (
    _file_hash,
    _integer,
    _load_frame,
    _new_output,
    _sha,
    _timestamp,
    _validated_manifest,
    _write_json,
)
from .scene import ImageFrame
from .state_estimation import BeliefState, StateConfig, TemporalStateEstimator
from .vision import Detector, PixelPerception

MAX_TRACK_BYTES = 256 * 1024 * 1024
MAX_TRACK_LINE_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_PREVIEW_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_FRAMES = 120
PREVIEW_WIDTH = 720
PREVIEW_IMAGE_HEIGHT = 480
PREVIEW_CAPTION_HEIGHT = 176
_OUTPUT_FILES = {"tracks.jsonl", "report.json", "preview.gif"}
_LIMITATIONS = [
    "identities are window-local geometric hypotheses; windows are never stitched",
    "screen-pixel velocity does not establish world velocity or climbing/platform state",
    "state timing uses imported integer milliseconds; raw source PTS and clock are retained",
    "retained boxes are last measurements, not detections in the current frame",
    "fixture palette and HUD parsing are synthetic-fixture behavior only",
    "detector-only map, HP, MP and actionability remain unknown/unavailable",
    "no ground-truth labels: reproducible inference is not measured detection accuracy",
    "fixed recordings cannot verify counterfactual action outcomes",
    "custom detector identity binds implementation and reproduced outputs, not arbitrary code",
    "hashes establish bundle consistency, not an authenticated source or permission grant",
]


def _canonical_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def _identity(
    manifest: dict[str, Any],
    config: StateConfig,
    detector: Detector | None,
    movement_config: MovementConfig | None = None,
    actor_selection: dict[str, Any] | None = None,
) -> tuple[FramePerception, dict[str, object], str]:
    implementation = PixelPerception() if detector is None else detector
    detector_only = detector is not None or actor_selection is not None
    metadata = _detector_metadata(implementation, include_runtime=True)
    metadata["observation_adapter"] = (
        "DetectorOnlyPerception.observe" if detector_only else "PixelPerception.observe"
    )
    metadata["max_detections"] = config.max_detections
    metadata["model_identity_bound"] = isinstance(implementation, OnnxDetector)
    if isinstance(implementation, PixelPerception):
        metadata["map_markers"] = dict(implementation.map_markers)
    # Normalize tuples to their JSON representation before comparing fresh reports.
    encoded = _canonical_bytes(metadata)
    if len(encoded) > 64 * 1024:
        raise ValueError("detector metadata exceeds its bounded identity budget")
    metadata = strict_json(encoded.decode("utf-8"))
    identity: dict[str, object] = {
        "recording_manifest": manifest,
        "state_config": config.to_dict(),
        "detector_metadata": metadata,
        "retain_detection_hypotheses": detector_only,
    }
    if movement_config is not None:
        identity["movement_hypotheses"] = {
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "config": movement_config.to_dict(),
        }
    if actor_selection is not None:
        identity["actor_selection"] = {
            "algorithm_version": ACTOR_SELECTION_ALGORITHM_VERSION,
            "specification": actor_selection,
            "retain_multiple_player_hypotheses": True,
        }
    window_id = _sha(_canonical_bytes(identity))
    perception: FramePerception = (
        implementation
        if isinstance(implementation, PixelPerception) and not detector_only
        else DetectorOnlyPerception(implementation, max_detections=config.max_detections)
    )
    return perception, metadata, window_id


def _original_time(manifest: dict[str, Any], item: dict[str, Any]) -> int:
    if manifest["timebase"] == "video_presentation_ticks":
        clock = manifest["video_clock"]
        return round(
            Fraction(item["pts"]) * Fraction(clock["numerator"], clock["denominator"]) * 1000
        )
    sequence = manifest.get("source_start_frame", 0) + item["sequence"]
    return _timestamp(sequence, manifest["fps"])


def _source_identity(manifest: dict[str, Any]) -> dict[str, object]:
    return {
        **{
            key: manifest[key]
            for key in (
                "session_id",
                "source_kind",
                "source_name",
                "source_uri",
                "permission",
                "license",
                "source_sha256",
                "source_hash_basis",
                "frame_count",
                "width",
                "height",
                "fps",
                "timebase",
                "truncated",
                "completion_reason",
            )
        },
        "manifest_sha256": _sha(_canonical_bytes(manifest)),
        "source_start_frame": manifest.get("source_start_frame", 0),
        "source_end_frame": manifest.get("source_start_frame", 0) + manifest["frame_count"] - 1,
        "source_start_timestamp_ms": _original_time(manifest, manifest["frames"][0]),
        "video_clock": manifest.get("video_clock"),
    }


def _assert_source_unchanged(root: Path, manifest: dict[str, Any]) -> None:
    # A detector can take time or consume an externally changing directory.
    # Recheck earlier frames too, not only the most recently loaded PNG.
    if _canonical_bytes(_validated_manifest(root)) != _canonical_bytes(manifest):
        raise ValueError("recording source identity changed during tracking or verification")


def _actor_selection_spec(manifest: dict[str, Any], value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "kind", "source", "anchor", "config"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != "recording_actor_selection"
        or _canonical_bytes(value["source"]) != _canonical_bytes(_source_identity(manifest))
    ):
        raise ValueError("actor selection source or format differs from the recording")
    anchor = ActorAnchor.from_dict(value["anchor"])
    config = ActorSelectionConfig.from_dict(value["config"])
    local = anchor.source_sequence - manifest.get("source_start_frame", 0)
    if not 0 <= local < manifest["frame_count"]:
        raise ValueError("actor selection anchor is outside the recording window")
    item = manifest["frames"][local]
    if anchor.timestamp_ms != item["timestamp_ms"] or anchor.rgb_sha256 != item["rgb_sha256"]:
        raise ValueError("actor selection anchor frame identity differs from source pixels")
    if anchor.bbox[2] > manifest["width"] or anchor.bbox[3] > manifest["height"]:
        raise ValueError("actor selection bbox exceeds source dimensions")
    return {
        "schema_version": 1,
        "kind": "recording_actor_selection",
        "source": _source_identity(manifest),
        "anchor": anchor.to_dict(),
        "config": config.to_dict(),
    }


def create_recording_actor_selection(
    recording: Path,
    output: Path,
    *,
    source_sequence: int,
    bbox: tuple[float, float, float, float],
    reviewer: str,
    subject: str,
    config: ActorSelectionConfig | None = None,
) -> dict[str, Any]:
    """Bind an explicitly reviewed target box to one actual source frame."""
    recording, output = Path(recording), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"actor selection already exists: {output}")
    if output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write actor selection outside the recording")
    manifest = _validated_manifest(recording)
    _integer(source_sequence, "source_sequence", 0, 2**53)
    local = source_sequence - manifest.get("source_start_frame", 0)
    if not 0 <= local < manifest["frame_count"]:
        raise ValueError("actor selection anchor is outside the recording window")
    if config is not None and type(config) is not ActorSelectionConfig:
        raise ValueError("actor selection config must be ActorSelectionConfig")
    item = manifest["frames"][local]
    anchor = ActorAnchor(
        source_sequence, item["timestamp_ms"], item["rgb_sha256"], bbox, reviewer, subject
    )
    specification = _actor_selection_spec(
        manifest,
        {
            "schema_version": 1,
            "kind": "recording_actor_selection",
            "source": _source_identity(manifest),
            "anchor": anchor.to_dict(),
            "config": (config or ActorSelectionConfig()).to_dict(),
        },
    )
    _assert_source_unchanged(recording, manifest)
    publish_new(output, _canonical_bytes(specification) + b"\n", 64 * 1024, "actor selection")
    return specification


def _tracked_frames(
    root: Path,
    manifest: dict[str, Any],
    config: StateConfig,
    perception: FramePerception,
    window_id: str,
    *,
    detector_only: bool,
    movement_config: MovementConfig | None = None,
    actor_selection: dict[str, Any] | None = None,
) -> Iterator[tuple[dict[str, object], ImageFrame, BeliefState]]:
    estimator = TemporalStateEstimator(
        config,
        retain_detection_hypotheses=detector_only,
        retain_multiple_player_hypotheses=actor_selection is not None,
    )
    movement = MovementHypothesisEstimator(movement_config) if movement_config is not None else None
    selector = (
        ActorSelector(
            ActorAnchor.from_dict(actor_selection["anchor"]),
            ActorSelectionConfig.from_dict(actor_selection["config"]),
            state_config=config,
        )
        if actor_selection is not None
        else None
    )
    source_start = manifest.get("source_start_frame", 0)
    for item in manifest["frames"]:
        frame = _load_frame(root, item)
        observation = replace(
            perception.observe(frame),
            sequence=source_start + frame.sequence,
            source_id=f"recording-window:{window_id}",
        )
        belief = estimator.update(observation, now_ms=frame.timestamp_ms)
        row: dict[str, object] = {
            "schema_version": 1,
            "session_id": manifest["session_id"],
            "tracking_window_id": window_id,
            "sequence": frame.sequence,
            "source_sequence": source_start + frame.sequence,
            "source_start_frame": source_start,
            "timestamp_ms": frame.timestamp_ms,
            "source_timestamp_ms": _original_time(manifest, item),
            "pts": item.get("pts"),
            "video_clock": manifest.get("video_clock"),
            "timebase": manifest["timebase"],
            "frame_sha256": item["sha256"],
            "frame_rgb_sha256": item["rgb_sha256"],
            "source_name": item["source_name"],
            "source_sha256": manifest["source_sha256"],
            "source_frame_sha256": item["source_sha256"],
            "belief": belief.to_dict(),
        }
        selected = selector.update(belief, rgb_sha256=item["rgb_sha256"]) if selector else None
        if selected is not None:
            row["actor_selection"] = selected.to_dict()
        if movement is not None:
            row["movement"] = movement.update(belief, actor_selection=selected).to_dict()
        yield row, frame, belief


def _sampling(count: int, stride: int, maximum: int) -> tuple[int, list[int]]:
    _integer(stride, "preview_stride", 1, 10000)
    _integer(maximum, "preview_max_frames", 2, MAX_PREVIEW_FRAMES)
    effective = max(stride, (count - 2 + maximum - 1) // (maximum - 1))
    sequences = list(range(0, count, effective))
    if sequences[-1] != count - 1:
        sequences.append(count - 1)
    return effective, sequences


def _short(text: str, maximum: int = 104) -> str:
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def _overlay(
    frame: ImageFrame, belief: BeliefState, row: dict[str, object], *, detector_only: bool
) -> Image.Image:
    scale = min(PREVIEW_WIDTH / frame.width, PREVIEW_IMAGE_HEIGHT / frame.height)
    size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
    canvas = Image.new(
        "RGB", (PREVIEW_WIDTH, PREVIEW_IMAGE_HEIGHT + PREVIEW_CAPTION_HEIGHT), "#101620"
    )
    with Image.frombytes("RGB", (frame.width, frame.height), frame.pixels) as original:
        resized = original.resize(size, Image.Resampling.BILINEAR)
    offset = ((PREVIEW_WIDTH - size[0]) // 2, (PREVIEW_IMAGE_HEIGHT - size[1]) // 2)
    canvas.paste(resized, offset)
    resized.close()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=10)
    selection = row.get("actor_selection")
    selected_id = (
        selection.get("actor_track_id")
        if isinstance(selection, dict) and selection.get("status") == "selected"
        else None
    )
    measured = {
        (track.class_id, track.bbox) for track in belief.tracks if track.observed_this_frame
    }
    for detection in belief.observation.detections:
        if (detection.class_id, detection.bbox) in measured:
            continue
        x1, y1, x2, y2 = detection.bbox
        raw_box = (
            offset[0] + x1 * scale,
            offset[1] + y1 * scale,
            offset[0] + x2 * scale,
            offset[1] + y2 * scale,
        )
        draw.rectangle(raw_box, outline="#ffffff", width=1)
        draw.text(
            (raw_box[0] + 2, max(0, raw_box[1] - 11)),
            _short(f"raw {detection.class_id} (unassigned)", 56),
            fill="#ffffff",
            font=small,
        )
    for track in belief.tracks:
        color = (
            "#66bfff"
            if track.track_id == selected_id
            else "#f1b255"
            if track.identity_uncertain
            else "#53e3a6"
            if track.observed_this_frame
            else "#a0aec0"
        )
        x1, y1, x2, y2 = track.bbox
        box = (
            offset[0] + x1 * scale,
            offset[1] + y1 * scale,
            offset[0] + x2 * scale,
            offset[1] + y2 * scale,
        )
        draw.rectangle(box, outline=color, width=2)
        state = "current" if track.observed_this_frame else "retained"
        uncertainty = " ambiguous" if track.identity_uncertain else ""
        label = _short(f"{track.track_id} {track.association} {state}{uncertainty}", 72)
        if track.track_id == selected_id:
            label = "TARGET " + label
        label_y = (
            max(0, box[1] - 11)
            if track.observed_this_frame
            else min(PREVIEW_IMAGE_HEIGHT - 12, box[3] + 2)
        )
        draw.text((box[0] + 2, label_y), label, fill=color, font=small)
    mode = "DETECTOR HYPOTHESES" if detector_only else "SYNTHETIC FIXTURE"
    players = sum(item.class_id == "player" for item in belief.observation.detections)
    player_note = f" | {players} player boxes: identity ambiguous" if players > 1 else ""
    if isinstance(selection, dict):
        player_note = f" | {players} visible player boxes"
    movement = row.get("movement")
    detail_limit = 6 - int(isinstance(movement, dict)) - int(isinstance(selection, dict))
    lines = [
        f"{mode} | READ ONLY | window {str(row['tracking_window_id'])[:12]} | local IDs",
        f"source frame {row['source_sequence']} | local {frame.sequence} | "
        f"local {frame.timestamp_ms} ms | source {row['source_timestamp_ms']} ms",
        f"PTS {row['pts']} | {belief.observation.status.value}: {belief.reason}{player_note}",
        f"{len(belief.tracks)} tracks (details <={detail_limit}) | green=current, gray=retained, "
        "amber=ambiguous, white=raw",
    ]
    if isinstance(selection, dict):
        lines[3] = (
            f"{len(belief.tracks)} tracks (details <={detail_limit}) | blue=target, "
            "green=current, gray=retained, amber=ambiguous"
        )
        lines.append(
            f"Target: {selection['status']} | {selection['actor_track_id']} | {selection['reason']}"
        )
    if isinstance(movement, dict):
        reasons = movement.get("reasons", [])
        reason = ", ".join(str(value) for value in reasons) if isinstance(reasons, list) else ""
        lines.append(f"Movement: {movement['state']} | {reason}")
    # Keep the preview readable and bounded; the JSONL contains every full track.
    detailed = sorted(
        belief.tracks,
        key=lambda track: (
            track.class_id != "player",
            not track.observed_this_frame,
            track.track_id,
        ),
    )
    for track in detailed[:detail_limit]:
        velocity = track.velocity_px_per_second
        motion = "unknown" if velocity is None else f"({velocity[0]:.1f}, {velocity[1]:.1f})"
        lines.append(
            f"{track.track_id} {track.class_id} | {track.association} | "
            f"{'current' if track.observed_this_frame else 'retained'} | "
            f"ambiguous={track.identity_uncertain} | v={motion} px/s"
        )
    if not belief.tracks:
        lines.append("No track hypotheses; missing detections do not establish absence or death.")
    if isinstance(movement, dict):
        lines.append(
            "Geometry candidates only | Owner identity, rope grip and action success unverified"
        )
    else:
        lines.append(
            "Full hypotheses: tracks.jsonl | Screen motion only; "
            "climbing/platform occupancy unclassified"
        )
    for index, text in enumerate(lines):
        draw.text(
            (8, PREVIEW_IMAGE_HEIGHT + 5 + index * 15), _short(text), fill="#edf2f7", font=font
        )
    palette = canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    canvas.close()
    return palette


@dataclass
class _Summary:
    frames_processed: int = 0
    frames_with_detections: int = 0
    frames_with_tracks: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    classes: Counter[str] = field(default_factory=Counter)
    associations: Counter[str] = field(default_factory=Counter)
    movement_states: Counter[str] = field(default_factory=Counter)
    selection_statuses: Counter[str] = field(default_factory=Counter)
    selection_reasons: Counter[str] = field(default_factory=Counter)
    preview_images: list[Image.Image] = field(default_factory=list)
    preview_mapping: list[dict[str, object]] = field(default_factory=list)

    def add(
        self,
        row: dict[str, object],
        frame: ImageFrame,
        belief: BeliefState,
        *,
        sample: bool,
        detector_only: bool,
    ) -> None:
        self.frames_processed += 1
        self.frames_with_detections += bool(belief.observation.detections)
        self.frames_with_tracks += bool(belief.tracks)
        self.statuses[belief.observation.status.value] += 1
        self.classes.update(item.class_id for item in belief.observation.detections)
        self.associations.update(item.association for item in belief.tracks)
        movement = row.get("movement")
        if isinstance(movement, dict):
            self.movement_states[str(movement["state"])] += 1
        selection = row.get("actor_selection")
        if isinstance(selection, dict):
            self.selection_statuses[str(selection["status"])] += 1
            self.selection_reasons[str(selection["reason"])] += 1
        if sample:
            image = _overlay(frame, belief, row, detector_only=detector_only)
            self.preview_images.append(image)
            with image.convert("RGB") as rgb:
                rgb_sha256 = _sha(rgb.tobytes())
            self.preview_mapping.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "sequence",
                            "source_sequence",
                            "timestamp_ms",
                            "source_timestamp_ms",
                            "pts",
                            "frame_rgb_sha256",
                        )
                    },
                    "rendered_rgb_sha256": rgb_sha256,
                }
            )

    def preview(self, manifest: dict[str, Any]) -> bytes:
        durations: list[int] = []
        sequences = [
            _integer(item["sequence"], "preview sequence", 0, 9999) for item in self.preview_mapping
        ]
        frames = manifest["frames"]
        for index, sequence in enumerate(sequences):
            if index + 1 < len(sequences):
                elapsed = (
                    frames[sequences[index + 1]]["timestamp_ms"] - frames[sequence]["timestamp_ms"]
                )
            elif sequence > 0:
                elapsed = frames[sequence]["timestamp_ms"] - frames[sequence - 1]["timestamp_ms"]
            else:
                elapsed = round(1000 / manifest["fps"]) if manifest["fps"] is not None else 100
            duration = min(655350, max(10, round(elapsed / 10) * 10))
            durations.append(duration)
            self.preview_mapping[index]["duration_ms"] = duration
        stream = io.BytesIO()
        self.preview_images[0].save(
            stream,
            format="GIF",
            save_all=True,
            append_images=self.preview_images[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
        if stream.tell() > MAX_PREVIEW_BYTES:
            raise ValueError("tracking preview exceeds its encoded-byte budget")
        return stream.getvalue()

    def close(self) -> None:
        for image in self.preview_images:
            image.close()


def _report(
    manifest: dict[str, Any],
    config: StateConfig,
    metadata: dict[str, object],
    window_id: str,
    summary: _Summary,
    *,
    detector_only: bool,
    stride: int,
    maximum: int,
    effective: int,
    sequences: list[int],
    tracks_sha256: str,
    preview_sha256: str,
    movement_config: MovementConfig | None = None,
    actor_selection: dict[str, Any] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "recording_tracking_replay",
        "mode": "read_only",
        "scope": "single_imported_recording_window",
        "perception_mode": "detector_only" if detector_only else "synthetic_fixture",
        "session_id": manifest["session_id"],
        "source_sha256": manifest["source_sha256"],
        "source": _source_identity(manifest),
        "tracking_window_id": window_id,
        "identity_scope": "window_local_only",
        "state_config": config.to_dict(),
        "detector_metadata": metadata,
        "retain_detection_hypotheses": detector_only,
        "model_identity_bound": metadata["model_identity_bound"],
        "frames_processed": summary.frames_processed,
        "frames_with_detections": summary.frames_with_detections,
        "frames_with_tracks": summary.frames_with_tracks,
        "status_counts": dict(sorted(summary.statuses.items())),
        "detections_by_class": dict(sorted(summary.classes.items())),
        "association_counts": dict(sorted(summary.associations.items())),
        "preview_frames": len(summary.preview_mapping),
        "preview": {
            "requested_stride": stride,
            "max_frames": maximum,
            "effective_stride": effective,
            "sequences": sequences,
            "mapping": summary.preview_mapping,
            "width": PREVIEW_WIDTH,
            "height": PREVIEW_IMAGE_HEIGHT + PREVIEW_CAPTION_HEIGHT,
            "timing": "source_intervals_centisecond_quantized_capped_10_to_655350_ms",
            "renderer": (
                "recording_selected_actor_overlay_v1"
                if actor_selection is not None
                else "recording_movement_overlay_v1"
                if movement_config is not None
                else "recording_tracking_overlay_v1"
            ),
            "pillow_version": pillow_version,
        },
        "tracks_sha256": tracks_sha256,
        "preview_sha256": preview_sha256,
        "files": {"tracks.jsonl": tracks_sha256, "preview.gif": preview_sha256},
        "action_execution": "none",
        "accuracy_metrics_available": False,
        "effect_success_rate": None,
        "timing_accuracy": (
            "source_pts"
            if manifest["timebase"] == "video_presentation_ticks"
            else "nominal_fps_only"
        ),
        "limitations": list(_LIMITATIONS),
    }
    if movement_config is not None:
        result["movement_analysis"] = {
            "schema_version": 1,
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "config": movement_config.to_dict(),
            "frames_processed": sum(summary.movement_states.values()),
            "state_counts": dict(sorted(summary.movement_states.items())),
            "scope": "read_only_geometric_candidates",
            "owner_identity_verified": False,
            "rope_grip_verified": False,
            "real_domain_accuracy_available": False,
            "actionable": False,
        }
    if actor_selection is not None:
        result["retain_multiple_player_hypotheses"] = True
        result["actor_selection_analysis"] = {
            "schema_version": 1,
            "algorithm_version": ACTOR_SELECTION_ALGORITHM_VERSION,
            "specification": actor_selection,
            "frames_processed": sum(summary.selection_statuses.values()),
            "status_counts": dict(sorted(summary.selection_statuses.items())),
            "reason_counts": dict(sorted(summary.selection_reasons.items())),
            "identity_scope": "explicit_anchor_then_uninterrupted_geometric_track",
            "owner_identity_verified": False,
            "actionable": False,
        }
    return result


def _line(row: dict[str, object]) -> bytes:
    encoded = _canonical_bytes(row) + b"\n"
    if len(encoded) > MAX_TRACK_LINE_BYTES:
        raise ValueError("tracking row exceeds its encoded-byte budget")
    return encoded


def track_recording(
    recording: Path,
    output: Path,
    detector: Detector | None = None,
    *,
    state_config: StateConfig | None = None,
    movement_config: MovementConfig | None = None,
    actor_selection: dict[str, Any] | None = None,
    preview_stride: int = 3,
    preview_max_frames: int = 120,
) -> dict[str, object]:
    """Track all imported frames; sample only the bounded presentation GIF."""
    root = Path(recording)
    manifest = _validated_manifest(root)
    if state_config is not None and not isinstance(state_config, StateConfig):
        raise ValueError("state_config must be a StateConfig")
    if movement_config is not None and not isinstance(movement_config, MovementConfig):
        raise ValueError("movement_config must be a MovementConfig")
    movement_config = (
        MovementConfig.from_dict(movement_config.to_dict()) if movement_config is not None else None
    )
    actor_selection = (
        _actor_selection_spec(manifest, actor_selection) if actor_selection is not None else None
    )
    detector_only = detector is not None or actor_selection is not None
    config = StateConfig.from_dict((state_config or StateConfig()).to_dict())
    effective, sequences = _sampling(manifest["frame_count"], preview_stride, preview_max_frames)
    perception, metadata, window_id = _identity(
        manifest, config, detector, movement_config, actor_selection
    )
    samples = set(sequences)
    summary = _Summary()
    try:
        with _new_output(output, root) as (stage, destination):
            size = 0
            with (stage / "tracks.jsonl").open("xb") as stream:
                for row, frame, belief in _tracked_frames(
                    root,
                    manifest,
                    config,
                    perception,
                    window_id,
                    detector_only=detector_only,
                    movement_config=movement_config,
                    actor_selection=actor_selection,
                ):
                    encoded = _line(row)
                    size += len(encoded)
                    if size > MAX_TRACK_BYTES:
                        raise ValueError("tracking trace exceeds its encoded-byte budget")
                    stream.write(encoded)
                    summary.add(
                        row,
                        frame,
                        belief,
                        sample=frame.sequence in samples,
                        detector_only=detector_only,
                    )
            if (
                _identity(manifest, config, detector, movement_config, actor_selection)[1]
                != metadata
            ):
                raise ValueError("detector identity changed during tracking")
            preview = summary.preview(manifest)
            (stage / "preview.gif").write_bytes(preview)
            report = _report(
                manifest,
                config,
                metadata,
                window_id,
                summary,
                detector_only=detector_only,
                stride=preview_stride,
                maximum=preview_max_frames,
                effective=effective,
                sequences=sequences,
                tracks_sha256=_file_hash(stage / "tracks.jsonl"),
                preview_sha256=_sha(preview),
                movement_config=movement_config,
                actor_selection=actor_selection,
            )
            if len(_canonical_bytes(report)) + 1 > MAX_REPORT_BYTES:
                raise ValueError("tracking report exceeds its encoded-byte budget")
            _write_json(stage / "report.json", report)
            _assert_source_unchanged(root, manifest)
    finally:
        summary.close()
    return {**report, "output": str(destination)}


def _read_report(replay: Path) -> dict[str, Any]:
    if replay.is_symlink() or not replay.is_dir():
        raise ValueError("tracking replay must be a real local directory")
    entries = list(replay.iterdir())
    if {path.name for path in entries} != _OUTPUT_FILES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("tracking replay has symlinks, extra files, or missing files")
    for name, maximum in (
        ("report.json", MAX_REPORT_BYTES),
        ("tracks.jsonl", MAX_TRACK_BYTES),
        ("preview.gif", MAX_PREVIEW_BYTES),
    ):
        if not 0 < (replay / name).stat().st_size <= maximum:
            raise ValueError(f"tracking {name} exceeds its encoded-byte budget or is empty")
    report = strict_json((replay / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("preview"), dict):
        raise ValueError("invalid tracking report")
    return report


def verify_recording_tracking(
    recording: Path, replay: Path, detector: Detector | None = None
) -> dict[str, object]:
    """Recompute the whole trajectory and GIF; never trust the stored beliefs."""
    root, replay = Path(recording), Path(replay)
    manifest = _validated_manifest(root)
    report = _read_report(replay)
    config = StateConfig.from_dict(report.get("state_config"))
    movement_config = None
    if "movement_analysis" in report:
        movement_spec = report["movement_analysis"]
        if not isinstance(movement_spec, dict):
            raise ValueError("invalid movement analysis specification")
        movement_config = MovementConfig.from_dict(movement_spec.get("config"))
    actor_selection = None
    if "actor_selection_analysis" in report:
        analysis = report["actor_selection_analysis"]
        if not isinstance(analysis, dict):
            raise ValueError("invalid actor selection analysis")
        actor_selection = _actor_selection_spec(manifest, analysis.get("specification"))
    detector_only = detector is not None or actor_selection is not None
    preview_spec = report["preview"]
    stride = _integer(preview_spec.get("requested_stride"), "preview_stride", 1, 10000)
    maximum = _integer(preview_spec.get("max_frames"), "preview_max_frames", 2, MAX_PREVIEW_FRAMES)
    effective, sequences = _sampling(manifest["frame_count"], stride, maximum)
    perception, metadata, window_id = _identity(
        manifest, config, detector, movement_config, actor_selection
    )
    if (
        _canonical_bytes(report.get("detector_metadata")) != _canonical_bytes(metadata)
        or report.get("tracking_window_id") != window_id
        or _canonical_bytes(report.get("source")) != _canonical_bytes(_source_identity(manifest))
    ):
        raise ValueError("tracking source, detector or configuration identity mismatch")
    for name, key in (("tracks.jsonl", "tracks_sha256"), ("preview.gif", "preview_sha256")):
        if _file_hash(replay / name) != report.get(key):
            raise ValueError(f"tracking {name} SHA256 mismatch")
    samples = set(sequences)
    summary = _Summary()
    digest = hashlib.sha256()
    try:
        with (replay / "tracks.jsonl").open("rb") as stream:
            for row, frame, belief in _tracked_frames(
                root,
                manifest,
                config,
                perception,
                window_id,
                detector_only=detector_only,
                movement_config=movement_config,
                actor_selection=actor_selection,
            ):
                actual = stream.readline(MAX_TRACK_LINE_BYTES + 1)
                if len(actual) > MAX_TRACK_LINE_BYTES or not actual:
                    raise ValueError("tracking trajectory has an oversized or missing row")
                try:
                    stored = strict_json(actual.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    raise ValueError("invalid tracking trajectory JSON") from exc
                encoded = _line(row)
                if _canonical_bytes(stored) != _canonical_bytes(row):
                    raise ValueError(
                        f"tracking trajectory mismatch at local frame {frame.sequence}"
                    )
                digest.update(encoded)
                summary.add(
                    row,
                    frame,
                    belief,
                    sample=frame.sequence in samples,
                    detector_only=detector_only,
                )
            if stream.read(1):
                raise ValueError("tracking trajectory contains extra rows")
        if _identity(manifest, config, detector, movement_config, actor_selection)[1] != metadata:
            raise ValueError("detector identity changed during verification")
        preview = summary.preview(manifest)
        if _sha(preview) != _file_hash(replay / "preview.gif"):
            raise ValueError("tracking preview differs from independently reproduced pixels")
        expected = _report(
            manifest,
            config,
            metadata,
            window_id,
            summary,
            detector_only=detector_only,
            stride=stride,
            maximum=maximum,
            effective=effective,
            sequences=sequences,
            tracks_sha256=digest.hexdigest(),
            preview_sha256=_sha(preview),
            movement_config=movement_config,
            actor_selection=actor_selection,
        )
        if _canonical_bytes(report) != _canonical_bytes(expected):
            raise ValueError("tracking report differs from independently recomputed evidence")
        _assert_source_unchanged(root, manifest)
        if _canonical_bytes(_read_report(replay)) != _canonical_bytes(report) or any(
            _file_hash(replay / name) != expected[key]
            for name, key in (("tracks.jsonl", "tracks_sha256"), ("preview.gif", "preview_sha256"))
        ):
            raise ValueError("tracking replay changed during verification")
    finally:
        summary.close()
    result: dict[str, object] = {
        "verified": True,
        "scope": "single_imported_recording_window",
        "identity_scope": "window_local_only",
        "session_id": manifest["session_id"],
        "tracking_window_id": window_id,
        "frames_verified": summary.frames_processed,
        "preview_frames": len(summary.preview_mapping),
        "source_identity_verified": True,
        "original_source_file_redecoded": False,
        "detector_metadata_verified": True,
        "model_identity_bound": metadata["model_identity_bound"],
        "trajectory_recomputed": True,
        "preview_verified": True,
        "preview_recomputed": True,
        "mode": "read_only",
        "action_execution": "none",
        "accuracy_metrics_available": False,
        "effect_success_rate": None,
    }
    if movement_config is not None:
        result["movement_recomputed"] = True
        result["movement_state_counts"] = dict(sorted(summary.movement_states.items()))
    if actor_selection is not None:
        result["actor_selection_recomputed"] = True
        result["actor_selection_status_counts"] = dict(sorted(summary.selection_statuses.items()))
    return result
