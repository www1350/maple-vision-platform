"""Score selected actors against independent, partial visible-identity reviews.

References are used only after a replay has been independently re-inferred. They
do not enter detection or selection, and do not authenticate character ownership.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import canonical_json
from .contracts import Detection
from .evaluation import _iou, _threshold
from .local_files import publish_new, read_regular
from .recording_movement_evaluation import (
    MAX_BENCHMARK_BYTES,
    MAX_LABEL_BYTES,
    MAX_MANIFEST_BYTES,
    _chunks,
    _document,
    _file_digest,
    _frame_identity,
    _manifest_hash,
    _manifest_snapshot,
    _Snapshot,
    _unchanged,
)
from .recording_tracking import (
    MAX_PREVIEW_BYTES,
    MAX_REPORT_BYTES,
    MAX_TRACK_BYTES,
    verify_recording_tracking,
)
from .recordings import _validated_manifest
from .vision import Detector

ACTOR_EVALUATION_ALGORITHM_VERSION = "selected-actor-reference-evaluation-v1"
MAX_OTHER_ACTORS = 16
_ACTOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z", re.ASCII)
_LABEL_FIELDS = {
    "schema_version",
    "kind",
    "session_id",
    "source_sha256",
    "manifest_sha256",
    "domain",
    "reviewer",
    "subject",
    "frames",
}
_REVIEW_FIELDS = {"visibility", "actor_bbox", "other_actors", "note"}
_VISIBILITIES = ("visible", "not_visible", "uncertain")
_SELECTION_STATES = ("waiting", "selected", "invalidated")
_OUTCOMES = (
    "correct_target",
    "wrong_other_actor",
    "localization_mismatch",
    "ambiguous_reference_match",
    "missed_visible_target",
    "correct_abstention_not_visible",
    "selected_when_not_visible",
    "unreviewed",
    "reference_uncertain",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text(value: object, label: str, *, empty: bool = False, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or (not empty and not value.strip())
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid actor review {label}")
    return value


def _destination(output: Path, *inputs: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"actor evidence already exists: {output}")
    resolved = output.resolve()
    if any(resolved.is_relative_to(path.resolve()) for path in inputs):
        raise ValueError("write actor evidence outside input recordings, replays and labels")


def _box(value: object, manifest: dict[str, Any]) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
        or not 0 <= value[0] < value[2] <= manifest["width"]
        or not 0 <= value[1] < value[3] <= manifest["height"]
    ):
        raise ValueError("actor review needs a bounded integer bbox")


def _review(value: object, manifest: dict[str, Any]) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _REVIEW_FIELDS:
        raise ValueError("invalid actor frame review fields")
    visibility = value["visibility"]
    if visibility not in _VISIBILITIES:
        raise ValueError("invalid actor review visibility")
    _text(value["note"], "note", empty=visibility == "visible", maximum=512)
    if visibility == "visible":
        _box(value["actor_bbox"], manifest)
    elif value["actor_bbox"] is not None:
        raise ValueError("non-visible or uncertain actor references require a null target bbox")
    others = value["other_actors"]
    if not isinstance(others, list) or len(others) > MAX_OTHER_ACTORS:
        raise ValueError("actor review allows at most 16 other actors")
    seen: set[str] = set()
    for other in others:
        if not isinstance(other, dict) or set(other) != {"actor_id", "bbox"}:
            raise ValueError("invalid other actor reference fields")
        identity = other["actor_id"]
        if (
            not isinstance(identity, str)
            or _ACTOR_ID.fullmatch(identity) is None
            or identity == "target"
            or identity in seen
        ):
            raise ValueError("other actor IDs must be unique short ASCII IDs excluding target")
        seen.add(identity)
        _box(other["bbox"], manifest)


def _labels(raw: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
    document = _document(raw, "actor labels")
    if (
        set(document) != _LABEL_FIELDS
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document["kind"] != "recording_actor_labels"
        or document["session_id"] != manifest["session_id"]
        or document["source_sha256"] != manifest["source_sha256"]
        or document["manifest_sha256"] != _manifest_hash(manifest)
    ):
        raise ValueError("actor label format or recording identity mismatch")
    if document["domain"] not in ("authored_fixture", "game_recording"):
        raise ValueError("declare the actor review domain")
    _text(document["reviewer"], "reviewer")
    _text(document["subject"], "subject")
    frames = document["frames"]
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("actor labels must inventory every imported frame")
    for entry, frame in zip(frames, manifest["frames"], strict=True):
        identity = _frame_identity(manifest, frame)
        if (
            not isinstance(entry, dict)
            or set(entry) != {*identity, "review"}
            or canonical_json({key: entry[key] for key in identity}) != canonical_json(identity)
        ):
            raise ValueError("actor label frame identity differs from the recording")
        _review(entry["review"], manifest)
    return document


def create_actor_label_template(recording: Path, output: Path) -> dict[str, Any]:
    """Inventory every source frame without deriving any labels from predictions."""
    recording, output = Path(recording), Path(output)
    _destination(output, recording)
    manifest, raw = _manifest_snapshot(recording)
    document = {
        "schema_version": 1,
        "kind": "recording_actor_labels",
        "session_id": manifest["session_id"],
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": _manifest_hash(manifest),
        "domain": None,
        "reviewer": None,
        "subject": None,
        "frames": [
            {**_frame_identity(manifest, frame), "review": None} for frame in manifest["frames"]
        ],
    }
    if canonical_json(_validated_manifest(recording)) != canonical_json(manifest):
        raise ValueError("recording source changed during actor template creation")
    if read_regular(recording / "manifest.json", MAX_MANIFEST_BYTES, "recording manifest") != raw:
        raise ValueError("recording manifest changed during actor template creation")
    publish_new(
        output,
        (canonical_json(document) + "\n").encode("utf-8"),
        MAX_LABEL_BYTES,
        "actor label template",
    )
    return document


def _selection(row: dict[str, Any]) -> dict[str, Any]:
    selection = row.get("actor_selection")
    if not isinstance(selection, dict) or selection.get("status") not in _SELECTION_STATES:
        raise ValueError("tracking row lacks a supported actor selection")
    status, track_id = selection["status"], selection.get("actor_track_id")
    reason = _text(selection.get("reason"), "selection reason")
    box = None
    if status == "selected":
        _text(track_id, "selected track ID")
        tracks = [
            track
            for track in row["belief"]["tracks"]
            if track["track_id"] == track_id
            and track["class_id"] == "player"
            and track["observed_this_frame"] is True
        ]
        if len(tracks) != 1:
            raise ValueError("selected actor must have one current player track")
        box = tracks[0]["bbox"]
    elif track_id is not None:
        raise ValueError("unselected actor cannot declare a selected track")
    return {"status": status, "reason": reason, "track_id": track_id, "bbox": box}


def _reference_match(
    selection: dict[str, Any], review: dict[str, Any] | None, threshold: float
) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    matches: list[str] = []
    if review is not None:
        actors = list(review["other_actors"])
        if review["visibility"] == "visible":
            actors.insert(0, {"actor_id": "target", "bbox": review["actor_bbox"]})
        for actor in actors:
            measured = (
                _iou(
                    Detection("player", tuple(selection["bbox"]), 1.0),
                    Detection("player", tuple(actor["bbox"]), 1.0),
                )
                if selection["bbox"] is not None
                else None
            )
            matched = measured is not None and measured >= threshold
            references.append(
                {
                    "actor_id": actor["actor_id"],
                    "bbox": actor["bbox"],
                    "iou": measured,
                    "meets_threshold": matched,
                }
            )
            if matched:
                matches.append(actor["actor_id"])
    if review is None:
        status = "unreviewed"
    elif review["visibility"] == "uncertain":
        status = "reference_uncertain"
    else:
        status = "unique" if len(matches) == 1 else "ambiguous" if matches else "none"
    return {
        "status": status,
        "actor_id": matches[0] if status == "unique" else None,
        "matched_actor_ids": matches,
        "references": references,
    }


def _outcome(
    selection: dict[str, Any], review: dict[str, Any] | None, match: dict[str, Any]
) -> str:
    if review is None:
        return "unreviewed"
    if review["visibility"] == "uncertain":
        return "reference_uncertain"
    selected = selection["status"] == "selected"
    if review["visibility"] == "not_visible":
        return "selected_when_not_visible" if selected else "correct_abstention_not_visible"
    if not selected:
        return "missed_visible_target"
    if match["status"] == "ambiguous":
        return "ambiguous_reference_match"
    if match["status"] == "unique":
        return "correct_target" if match["actor_id"] == "target" else "wrong_other_actor"
    return "localization_mismatch"


def _metrics(
    coverage: Counter[str], outcomes: Counter[str], selected: Counter[str]
) -> dict[str, int | float | None]:
    visible, not_visible = coverage["visible"], coverage["not_visible"]
    decidable = visible + not_visible
    selected_decidable = selected["visible"] + selected["not_visible"]
    return {
        "visible_target_frames": visible,
        "correct_target_frames": outcomes["correct_target"],
        "selected_visible_target_frames": selected["visible"],
        "target_recall": outcomes["correct_target"] / visible if visible else None,
        "visible_selection_coverage": selected["visible"] / visible if visible else None,
        "not_visible_frames": not_visible,
        "selected_when_not_visible_frames": outcomes["selected_when_not_visible"],
        "not_visible_selection_rate": (
            outcomes["selected_when_not_visible"] / not_visible if not_visible else None
        ),
        "reviewed_decidable_frames": decidable,
        "selected_decidable_frames": selected_decidable,
        "selection_coverage": selected_decidable / decidable if decidable else None,
        "correct_abstention_not_visible_frames": outcomes["correct_abstention_not_visible"],
    }


def _start_interval(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_sequence": frame["sequence"],
        "end_sequence": frame["sequence"],
        "start_source_sequence": frame["source_sequence"],
        "end_source_sequence": frame["source_sequence"],
        "start_source_timestamp_ms": frame["source_timestamp_ms"],
        "end_source_timestamp_ms": frame["source_timestamp_ms"],
        "frames": 1,
    }


def benchmark_actor_recording(
    recording: Path,
    replay: Path,
    labels: Path,
    output: Path,
    detector: Detector | None = None,
    *,
    actor_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate actor selection independently of motion states or action results."""
    recording, replay, labels, output = map(Path, (recording, replay, labels, output))
    threshold = _threshold(actor_iou_threshold, "actor_iou_threshold", positive=True)
    _destination(output, recording, replay, labels)
    manifest, manifest_raw = _manifest_snapshot(recording)
    labels_raw = read_regular(labels, MAX_LABEL_BYTES, "actor labels")
    document = _labels(labels_raw, manifest)
    report_raw = read_regular(replay / "report.json", MAX_REPORT_BYTES, "tracking report")
    tracking_report = _document(report_raw, "tracking report")
    if not isinstance(tracking_report.get("actor_selection_analysis"), dict):
        raise ValueError("actor evaluation requires a replay with actor selection")
    tracks_hash = _file_digest(replay / "tracks.jsonl", MAX_TRACK_BYTES)
    preview_hash = _file_digest(replay / "preview.gif", MAX_PREVIEW_BYTES)
    snapshot: _Snapshot = {
        "manifest_raw": manifest_raw,
        "report_raw": report_raw,
        "labels_raw": labels_raw,
        "tracks_hash": tracks_hash,
        "preview_hash": preview_hash,
    }
    # Human identities and boxes are never supplied to this inference call.
    verification = verify_recording_tracking(recording, replay, detector)
    _unchanged(recording, replay, labels, **snapshot)
    if verification.get("actor_selection_recomputed") is not True:
        raise ValueError("actor evaluation requires independently recomputed actor selection")

    coverage: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    diagnostics = []
    scored_hash = hashlib.sha256()
    previous_identity: tuple[int, str] | None = None
    eligible_pairs = identity_changes = 0
    intervals: list[dict[str, Any]] = []
    current_interval: dict[str, Any] | None = None
    for entry, raw_row in zip(
        document["frames"],
        _chunks(replay / "tracks.jsonl", MAX_TRACK_BYTES, lines=True),
        strict=True,
    ):
        scored_hash.update(raw_row)
        row = _document(raw_row, "tracking row")
        identity = {key: entry[key] for key in entry if key != "review"}
        row_identity = {
            key: row.get("frame_rgb_sha256" if key == "rgb_sha256" else key) for key in identity
        }
        if canonical_json(identity) != canonical_json(row_identity):
            raise ValueError("actor review and trajectory frame identities differ")
        review = entry["review"]
        coverage["reviewed" if review is not None else "unreviewed"] += 1
        if review is not None:
            coverage[review["visibility"]] += 1
        selection = _selection(row)
        statuses[selection["status"]] += 1
        if review is not None and selection["status"] == "selected":
            selected[review["visibility"]] += 1
        match = _reference_match(selection, review, threshold)
        outcome = _outcome(selection, review, match)
        outcomes[outcome] += 1

        if match["status"] == "unique":
            current_identity = (entry["source_sequence"], match["actor_id"])
            if previous_identity is not None and previous_identity[0] + 1 == current_identity[0]:
                eligible_pairs += 1
                identity_changes += previous_identity[1] != current_identity[1]
            previous_identity = current_identity
        else:
            previous_identity = None
        if outcome == "correct_target":
            if (
                current_interval is not None
                and current_interval["end_source_sequence"] + 1 == entry["source_sequence"]
            ):
                current_interval["end_sequence"] = entry["sequence"]
                current_interval["end_source_sequence"] = entry["source_sequence"]
                current_interval["end_source_timestamp_ms"] = entry["source_timestamp_ms"]
                current_interval["frames"] += 1
            else:
                if current_interval is not None:
                    intervals.append(current_interval)
                current_interval = _start_interval(entry)
        elif current_interval is not None:
            intervals.append(current_interval)
            current_interval = None
        diagnostics.append(
            {
                **identity,
                "review_status": "reviewed" if review is not None else "unreviewed",
                "review_note": review["note"] if review is not None else None,
                "reference_visibility": review["visibility"] if review is not None else None,
                "review_actor_bbox": review["actor_bbox"] if review is not None else None,
                "review_other_actors": review["other_actors"] if review is not None else None,
                "selection": selection,
                "reference_match": match,
                "outcome": outcome,
            }
        )
    if current_interval is not None:
        intervals.append(current_interval)
    longest = max(intervals, key=lambda interval: interval["frames"], default=None)
    if scored_hash.hexdigest() != tracks_hash:
        raise ValueError("tracking trajectory changed between verification and actor scoring")
    if canonical_json(_validated_manifest(recording)) != canonical_json(manifest):
        raise ValueError("recording source changed during actor scoring")
    _unchanged(recording, replay, labels, **snapshot)
    configuration = {
        "algorithm_version": ACTOR_EVALUATION_ALGORITHM_VERSION,
        "detector_metadata": tracking_report["detector_metadata"],
        "state_config": tracking_report["state_config"],
        "actor_selection_analysis": tracking_report["actor_selection_analysis"],
        "actor_iou_threshold": threshold,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "recording_actor_benchmark",
        "algorithm_version": ACTOR_EVALUATION_ALGORITHM_VERSION,
        "scope": "single_imported_recording_window",
        "declared_domain": document["domain"],
        "reviewer": document["reviewer"],
        "subject": document["subject"],
        "frames": manifest["frame_count"],
        "evaluation_status": (
            "evaluated" if coverage["visible"] + coverage["not_visible"] else "no_scorable_labels"
        ),
        "actor_iou_threshold": threshold,
        "coverage": {name: coverage[name] for name in ("reviewed", "unreviewed", *_VISIBILITIES)},
        "selection_status_counts": {name: statuses[name] for name in _SELECTION_STATES},
        "outcomes": {name: outcomes[name] for name in _OUTCOMES},
        "metrics": _metrics(coverage, outcomes, selected),
        "continuity": {
            "eligible_adjacent_identity_pairs": eligible_pairs,
            "reference_identity_changes": identity_changes,
            "longest_correct_target_run_frames": longest["frames"] if longest is not None else 0,
            "longest_correct_target_interval": longest,
            "correct_target_intervals": intervals,
        },
        "diagnostics": diagnostics,
        "source": {
            "session_id": manifest["session_id"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": _manifest_hash(manifest),
            "timebase": manifest["timebase"],
            "video_clock": manifest.get("video_clock"),
        },
        "bindings": {
            "manifest_raw_sha256": _sha(manifest_raw),
            "labels_sha256": _sha(labels_raw),
            "tracking_report_sha256": _sha(report_raw),
            "tracks_sha256": tracks_hash,
            "preview_sha256": preview_hash,
            "tracking_window_id": tracking_report["tracking_window_id"],
            **configuration,
            "configuration_sha256": _sha(canonical_json(configuration).encode("utf-8")),
        },
        "verification": verification,
        "read_only": True,
        "actionable": False,
        "action_execution": "none",
        "owner_identity_verified": False,
        "action_success_evaluated": False,
        "limitations": [
            "independent review identities describe visible subjects, not authenticated ownership",
            "reference IoU matches geometry and cannot authenticate continuous real-world identity",
            "target recall includes visible targets with waiting or invalidated selections",
            "selection coverage includes wrong and ambiguous choices; it is not identity accuracy",
            "unreviewed and uncertain references are not negatives or successful abstentions",
            "identity changes require adjacent reviewed frames with unique reference matches",
            "reference identity changes are not standard multi-object-tracking ID switch metrics",
            "unreviewed, uncertain, ambiguous, unmatched and unselected frames break continuity",
            "other references may be partial; no match cannot identify a particular other actor",
            "partial labels and one recording do not establish held-out real-domain generalization",
            "hashes establish input consistency, not permission or an authenticated review",
        ],
    }
    publish_new(
        output,
        (canonical_json(report) + "\n").encode("utf-8"),
        MAX_BENCHMARK_BYTES,
        "actor benchmark",
    )
    return report
