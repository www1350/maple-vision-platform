"""Compare verified motion candidates with independent, partial human reviews.

Labels select a geometric subject for scoring only. They never enter perception,
tracking or movement inference, and do not establish character ownership.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

from .catalog import canonical_json, strict_json
from .contracts import Detection
from .evaluation import _iou, _threshold
from .local_files import publish_new, read_regular
from .recording_tracking import (
    MAX_PREVIEW_BYTES,
    MAX_REPORT_BYTES,
    MAX_TRACK_BYTES,
    MAX_TRACK_LINE_BYTES,
    _original_time,
    verify_recording_tracking,
)
from .recordings import _validated_manifest
from .vision import Detector

MAX_FRAMES = 5000
MAX_LABEL_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_BENCHMARK_BYTES = 32 * 1024 * 1024
_PREDICATES = ("rope_climb_visible", "arrival_after_climb_visible")
_STATES = {
    "unknown": (None, None),
    "near_rope": (False, False),
    "climbing_candidate": (True, False),
    "platform_arrival_candidate": (False, True),
}
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
_REVIEW_FIELDS = {"actor_bbox", *_PREDICATES, "note"}
_ALIGNMENTS = (
    "matched",
    "mismatched",
    "missing_prediction",
    "unreviewed",
    "reference_unknown",
)


class _Snapshot(TypedDict):
    manifest_raw: bytes
    report_raw: bytes
    labels_raw: bytes
    tracks_hash: str
    preview_hash: str


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str, *, empty: bool = False, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or (not empty and not value.strip())
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid movement review {label}")
    return value


def _destination(output: Path, *inputs: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"movement evidence already exists: {output}")
    resolved = output.resolve()
    if any(resolved.is_relative_to(path.resolve()) for path in inputs):
        raise ValueError("write movement evidence outside input recordings, replays and labels")


def _manifest_snapshot(recording: Path) -> tuple[dict[str, Any], bytes]:
    path = recording / "manifest.json"
    raw = read_regular(path, MAX_MANIFEST_BYTES, "recording manifest")
    raw_document = _document(raw, "recording manifest")
    count = raw_document.get("frame_count")
    if type(count) is not int or not 1 <= count <= MAX_FRAMES:
        raise ValueError("movement evaluation needs 1..5000 imported frames")
    manifest = _validated_manifest(recording)
    if read_regular(path, MAX_MANIFEST_BYTES, "recording manifest") != raw or canonical_json(
        raw_document
    ) != canonical_json(manifest):
        raise ValueError("recording manifest changed during movement evaluation")
    return manifest, raw


def _manifest_hash(manifest: dict[str, Any]) -> str:
    # Match the canonical identity already recorded by recording_tracking.
    return _sha(canonical_json(manifest).encode("utf-8"))


def _frame_identity(manifest: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": frame["sequence"],
        "source_sequence": manifest.get("source_start_frame", 0) + frame["sequence"],
        "timestamp_ms": frame["timestamp_ms"],
        "source_timestamp_ms": _original_time(manifest, frame),
        "pts": frame.get("pts"),
        "rgb_sha256": frame["rgb_sha256"],
    }


def create_movement_label_template(recording: Path, output: Path) -> dict[str, Any]:
    """Create an unreviewed frame inventory; never infer human labels from pixels."""
    recording, output = Path(recording), Path(output)
    _destination(output, recording)
    manifest, raw = _manifest_snapshot(recording)
    document = {
        "schema_version": 1,
        "kind": "recording_movement_labels",
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
    if read_regular(recording / "manifest.json", MAX_MANIFEST_BYTES, "recording manifest") != raw:
        raise ValueError("recording manifest changed during template creation")
    publish_new(
        output,
        (canonical_json(document) + "\n").encode("utf-8"),
        MAX_LABEL_BYTES,
        "movement label template",
    )
    return document


def _review(value: object, manifest: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _REVIEW_FIELDS:
        raise ValueError("invalid movement frame review fields")
    states = tuple(value[name] for name in _PREDICATES)
    if any(state is not None and type(state) is not bool for state in states):
        raise ValueError("movement review predicates must be boolean or null")
    if states == (True, True):
        raise ValueError("climbing and arrival cannot both be marked visible in one frame")
    uncertain = all(state is None for state in states)
    _text(value["note"], "note", empty=not uncertain, maximum=512)
    box = value["actor_bbox"]
    if box is None and uncertain:
        return value
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(coordinate) is not int for coordinate in box)
        or not 0 <= box[0] < box[2] <= manifest["width"]
        or not 0 <= box[1] < box[3] <= manifest["height"]
    ):
        raise ValueError("known movement labels require a valid bounded integer actor_bbox")
    return value


def _labels(raw: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
    document = _document(raw, "movement labels")
    if (
        set(document) != _LABEL_FIELDS
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document["kind"] != "recording_movement_labels"
        or document["session_id"] != manifest["session_id"]
        or document["source_sha256"] != manifest["source_sha256"]
        or document["manifest_sha256"] != _manifest_hash(manifest)
    ):
        raise ValueError("movement label format or recording identity mismatch")
    if document["domain"] not in ("authored_fixture", "game_recording"):
        raise ValueError("declare the movement review domain")
    _text(document["reviewer"], "reviewer")
    _text(document["subject"], "subject")
    frames = document["frames"]
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("movement labels must inventory every imported frame")
    for entry, frame in zip(frames, manifest["frames"], strict=True):
        identity = _frame_identity(manifest, frame)
        if (
            not isinstance(entry, dict)
            or set(entry) != {*identity, "review"}
            or canonical_json({key: entry[key] for key in identity}) != canonical_json(identity)
        ):
            raise ValueError("movement label frame identity differs from the recording")
        _review(entry["review"], manifest)
    return document


def _chunks(path: Path, limit: int, *, lines: bool = False) -> Iterator[bytes]:
    """Bound file descriptors and reads while keeping the trajectory streaming."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("movement replay inputs must be regular files, not symlinks")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
            raise ValueError("movement replay input exceeds its bounded size or is empty")
        total = 0
        while block := (
            stream.readline(MAX_TRACK_LINE_BYTES + 1) if lines else stream.read(1024 * 1024)
        ):
            total += len(block)
            if total > limit or (lines and len(block) > MAX_TRACK_LINE_BYTES):
                raise ValueError("movement replay input exceeds its bounded read budget")
            yield block


def _file_digest(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    for block in _chunks(path, limit):
        digest.update(block)
    return digest.hexdigest()


def _unchanged(
    recording: Path,
    replay: Path,
    labels: Path,
    *,
    manifest_raw: bytes,
    report_raw: bytes,
    labels_raw: bytes,
    tracks_hash: str,
    preview_hash: str,
) -> None:
    for path, expected, limit, label in (
        (recording / "manifest.json", manifest_raw, MAX_MANIFEST_BYTES, "recording manifest"),
        (replay / "report.json", report_raw, MAX_REPORT_BYTES, "tracking report"),
        (labels, labels_raw, MAX_LABEL_BYTES, "movement labels"),
    ):
        if read_regular(path, limit, label) != expected:
            raise ValueError(f"{label} changed during movement evaluation")
    if (
        _file_digest(replay / "tracks.jsonl", MAX_TRACK_BYTES) != tracks_hash
        or _file_digest(replay / "preview.gif", MAX_PREVIEW_BYTES) != preview_hash
    ):
        raise ValueError("tracking replay changed during movement evaluation")


def _actor_match(
    row: dict[str, Any], review: dict[str, Any] | None, threshold: float
) -> dict[str, Any]:
    movement = row["movement"]
    track_id = movement["actor_track_id"]
    tracks = [
        track
        for track in row["belief"]["tracks"]
        if track["track_id"] == track_id
        and track["class_id"] == "player"
        and track["observed_this_frame"] is True
    ]
    box = tracks[0]["bbox"] if len(tracks) == 1 else None
    measured = None
    if review is None:
        status = "unreviewed"
    elif review["actor_bbox"] is None:
        status = "reference_unknown"
    elif box is None:
        status = "missing_prediction"
    else:
        measured = _iou(
            Detection("player", tuple(box), 1.0),
            Detection("player", tuple(review["actor_bbox"]), 1.0),
        )
        status = "matched" if measured >= threshold else "mismatched"
    return {"status": status, "track_id": track_id, "bbox": box, "iou": measured}


def _outcome(expected: bool | None, predicted: bool | None, reviewed: bool) -> str:
    if not reviewed:
        return "unreviewed"
    if expected is None:
        return "uncertain"
    if predicted is None:
        return "abstained_positive" if expected else "abstained_negative"
    if expected:
        return "tp" if predicted else "fn"
    return "fp" if predicted else "tn"


def _metrics(counts: Counter[str]) -> dict[str, int | float | None]:
    positive = counts["tp"] + counts["fn"] + counts["abstained_positive"]
    negative = counts["fp"] + counts["tn"] + counts["abstained_negative"]
    decided = sum(counts[key] for key in ("tp", "fp", "tn", "fn"))
    predicted_positive = counts["tp"] + counts["fp"]
    return {
        "positives": positive,
        "negatives": negative,
        "true_positives": counts["tp"],
        "false_positives": counts["fp"],
        "true_negatives": counts["tn"],
        "false_negatives": counts["fn"] + counts["abstained_positive"],
        "abstained_positive": counts["abstained_positive"],
        "abstained_negative": counts["abstained_negative"],
        "unreviewed": counts["unreviewed"],
        "uncertain": counts["uncertain"],
        "decision_coverage": decided / (positive + negative) if positive + negative else None,
        "precision": counts["tp"] / predicted_positive if predicted_positive else None,
        "recall": counts["tp"] / positive if positive else None,
    }


def benchmark_movement_recording(
    recording: Path,
    replay: Path,
    labels: Path,
    output: Path,
    detector: Detector | None = None,
    *,
    actor_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score a fully re-inferred replay without exposing human reviews to inference."""
    recording, replay, labels, output = map(Path, (recording, replay, labels, output))
    threshold = _threshold(actor_iou_threshold, "actor_iou_threshold", positive=True)
    _destination(output, recording, replay, labels)
    manifest, manifest_raw = _manifest_snapshot(recording)
    labels_raw = read_regular(labels, MAX_LABEL_BYTES, "movement labels")
    document = _labels(labels_raw, manifest)
    report_raw = read_regular(replay / "report.json", MAX_REPORT_BYTES, "tracking report")
    tracking_report = _document(report_raw, "tracking report")
    if not isinstance(tracking_report.get("movement_analysis"), dict):
        raise ValueError("movement evaluation requires a replay with movement analysis")
    tracks_hash = _file_digest(replay / "tracks.jsonl", MAX_TRACK_BYTES)
    preview_hash = _file_digest(replay / "preview.gif", MAX_PREVIEW_BYTES)
    snapshot: _Snapshot = {
        "manifest_raw": manifest_raw,
        "report_raw": report_raw,
        "labels_raw": labels_raw,
        "tracks_hash": tracks_hash,
        "preview_hash": preview_hash,
    }
    # This is the existing full pixel, inference, trajectory and GIF verification.
    # No labels, subject boxes or review predicates are supplied to this call.
    verification = verify_recording_tracking(recording, replay, detector)
    _unchanged(recording, replay, labels, **snapshot)
    if verification.get("movement_recomputed") is not True:
        raise ValueError("movement evaluation requires independently recomputed movement")

    counters: dict[str, Counter[str]] = {name: Counter() for name in _PREDICATES}
    alignment: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    diagnostics = []
    scored_hash = hashlib.sha256()
    for entry, raw_row in zip(
        document["frames"],
        _chunks(replay / "tracks.jsonl", MAX_TRACK_BYTES, lines=True),
        strict=True,
    ):
        scored_hash.update(raw_row)
        row = _document(raw_row, "tracking row")
        movement = row.get("movement")
        if not isinstance(movement, dict) or movement.get("state") not in _STATES:
            raise ValueError("tracking row lacks supported movement candidates")
        identity = {key: entry[key] for key in entry if key != "review"}
        row_identity = {
            key: row.get("frame_rgb_sha256" if key == "rgb_sha256" else key) for key in identity
        }
        if canonical_json(identity) != canonical_json(row_identity):
            raise ValueError("movement review and trajectory frame identities differ")
        review = entry["review"]
        is_reviewed = review is not None
        coverage["reviewed" if is_reviewed else "unreviewed"] += 1
        expected = {name: review[name] if is_reviewed else None for name in _PREDICATES}
        if is_reviewed:
            coverage[
                "reviewed_uncertain"
                if all(value is None for value in expected.values())
                else "reviewed_with_decidable_predicate"
            ] += 1
        match = _actor_match(row, review, threshold)
        alignment[match["status"]] += 1
        if match["status"] == "mismatched" and movement["state"] in (
            "climbing_candidate",
            "platform_arrival_candidate",
        ):
            alignment["wrong_actor_positive_candidates"] += 1
        predicted = dict(
            zip(
                _PREDICATES,
                _STATES[movement["state"]] if match["status"] == "matched" else (None, None),
                strict=True,
            )
        )
        outcomes = {
            name: _outcome(expected[name], predicted[name], is_reviewed) for name in _PREDICATES
        }
        for name, outcome in outcomes.items():
            counters[name][outcome] += 1
        diagnostics.append(
            {
                **identity,
                "review_status": "reviewed" if is_reviewed else "unreviewed",
                "review_note": review["note"] if is_reviewed else None,
                "review_actor_bbox": review["actor_bbox"] if is_reviewed else None,
                "expected": expected,
                "predicted": predicted,
                "predicted_state": movement["state"],
                "predicted_reasons": movement["reasons"],
                "actor_match": match,
                "outcomes": outcomes,
            }
        )
    if scored_hash.hexdigest() != tracks_hash:
        raise ValueError("tracking trajectory changed between verification and scoring")
    # Revalidate all source pixels after scoring, then bind the exact bytes used.
    if canonical_json(_validated_manifest(recording)) != canonical_json(manifest):
        raise ValueError("recording source changed during movement scoring")
    _unchanged(recording, replay, labels, **snapshot)
    metrics = {name: _metrics(counter) for name, counter in counters.items()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "recording_movement_benchmark",
        "scope": "single_imported_recording_window",
        "declared_domain": document["domain"],
        "reviewer": document["reviewer"],
        "subject": document["subject"],
        "frames": manifest["frame_count"],
        "evaluation_status": (
            "evaluated" if coverage["reviewed_with_decidable_predicate"] else "no_scorable_labels"
        ),
        "actor_iou_threshold": threshold,
        "coverage": {
            name: coverage[name]
            for name in (
                "reviewed",
                "unreviewed",
                "reviewed_uncertain",
                "reviewed_with_decidable_predicate",
            )
        },
        "actor_alignment": {
            name: alignment[name] for name in (*_ALIGNMENTS, "wrong_actor_positive_candidates")
        },
        "predicates": metrics,
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
            "detector_metadata": tracking_report["detector_metadata"],
            "state_config": tracking_report["state_config"],
            "movement_analysis": tracking_report["movement_analysis"],
            "actor_iou_threshold": threshold,
        },
        "verification": verification,
        "read_only": True,
        "actionable": False,
        "action_execution": "none",
        "owner_identity_verified": False,
        "grip_verified": False,
        "action_success_evaluated": False,
        "limitations": [
            "human labels are independent visible-state judgments, not authenticated ground truth",
            "actor IoU matches a geometric reference, not ownership or continuous identity",
            "unknown or wrong-actor states abstain; abstained negatives are not true negatives",
            "false_negatives include abstained_positive; recall keeps every reviewed positive",
            "precision covers aligned decisions; wrong-actor positives are reported separately",
            "unreviewed and uncertain labels are excluded from decisions, with separate counts",
            "frame agreement does not measure event boundaries, rope grip or action success",
            "partial labels and one recording do not establish held-out real-domain generalization",
            "hashes establish input consistency, not permission or an authenticated review",
        ],
    }
    if "actor_selection_analysis" in tracking_report:
        report["bindings"]["actor_selection_analysis"] = tracking_report["actor_selection_analysis"]
    publish_new(
        output,
        (canonical_json(report) + "\n").encode("utf-8"),
        MAX_BENCHMARK_BYTES,
        "movement benchmark",
    )
    return report
