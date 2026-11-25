"""Score frozen, human-reviewed frame labels without exposing them to inference."""

import hashlib
import io
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import canonical_json, strict_json
from .contracts import Detection
from .dataset import _contained, validate_dataset
from .evaluation import (
    EvaluationSample,
    _bounded_manifest_bytes,
    _detector_metadata,
    _environment,
    _iou,
    evaluate_predictions,
)
from .input_bindings import InputBindingProfile, normalize_key
from .local_files import publish_new, read_regular
from .recordings import _load_frame, _validated_manifest
from .scene import CLASS_NAMES
from .vision import Detector, PixelPerception

MAX_FRAMES = 5000
MAX_OBJECTS = 50_000
MAX_OBJECTS_PER_FRAME = 128
MAX_LABEL_BYTES = 16 * 1024 * 1024
MAX_STRIKE_REPORT_BYTES = 1024 * 1024
MAX_EVENT_REPORT_BYTES = 4 * 1024 * 1024
MAX_REVIEWED_EVENTS = 256
MAX_WINDOW_REPORT_BYTES = 4 * 1024 * 1024
MAX_RECORDING_WINDOWS = 256
REVIEWED_EVENT_KINDS = frozenset(
    {
        "attack_motion_visible",
        "projectile_visible",
        "damage_feedback_visible",
        "airborne_visible",
        "rope_contact_visible",
        "layer_change_visible",
        "camera_motion_visible",
    }
)


@dataclass(frozen=True)
class Case:
    recording: Path
    manifest: dict[str, Any]
    label_sha256: str
    reviewer: str
    domain: str
    annotations: tuple[tuple[str, tuple[Detection, ...], tuple[bool, ...]], ...]


def _write(path: Path, value: object) -> None:
    # Reuse atomic create-only publication. Never overwrite reviewed labels.
    publish_new(
        path, (canonical_json(value) + "\n").encode(), 64 * 1024 * 1024, "recording evidence"
    )


def create_label_template(recording: Path, output: Path) -> dict[str, object]:
    if output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write the label template outside the captured recording")
    manifest = _validated_manifest(recording)
    if manifest["frame_count"] > MAX_FRAMES:
        raise ValueError("recording exceeds the scored frame budget")
    value = {
        "schema_version": 1,
        "kind": "recording_object_labels",
        "session_id": manifest["session_id"],
        "domain": None,
        "reviewer": None,
        "frames": [
            {
                "sequence": frame["sequence"],
                "rgb_sha256": frame["rgb_sha256"],
                "map_id": None,
                "objects": None,
            }
            for frame in manifest["frames"]
        ],
    }
    _write(output, value)
    return {
        "session_id": manifest["session_id"],
        "frames": len(value["frames"]),
        "output": str(output),
    }


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"invalid annotation {field}")
    return value


def _frame_identity(frame: dict[str, Any]) -> dict[str, Any]:
    return {key: frame[key] for key in ("sequence", "timestamp_ms", "rgb_sha256")}


def _reviewed_event_document(
    manifest: dict[str, Any], reviewer: str, events: Sequence[tuple[str, int, int]]
) -> dict[str, Any]:
    if not isinstance(events, (list, tuple)) or not 1 <= len(events) <= MAX_REVIEWED_EVENTS:
        raise ValueError("supply 1..256 reviewed visible events")
    normalized = []
    for kind, start, end in events:
        if (
            kind not in REVIEWED_EVENT_KINDS
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < manifest["frame_count"]
        ):
            raise ValueError("invalid reviewed visible event")
        normalized.append((kind, start, end))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate reviewed visible event")
    normalized.sort(key=lambda item: (item[1], item[2], item[0]))
    frames = manifest["frames"]
    reviewed = []
    for kind, start, end in normalized:
        first, last = frames[start], frames[end]
        reviewed.append(
            {
                "kind": kind,
                "start": _frame_identity(first),
                "end": _frame_identity(last),
                "duration_ms": last["timestamp_ms"] - first["timestamp_ms"],
            }
        )
    source = {
        "session_id": manifest["session_id"],
        "source_sha256": manifest["source_sha256"],
        "source_hash_basis": manifest["source_hash_basis"],
        "source_name": manifest["source_name"],
        "source_uri": manifest["source_uri"],
        "permission": manifest["permission"],
        "license": manifest["license"],
        "frame_count": manifest["frame_count"],
        "source_frame_size": [manifest["width"], manifest["height"]],
        "truncated": manifest["truncated"],
        "timebase": manifest["timebase"],
        "video_clock": manifest.get("video_clock"),
    }
    if "source_start_frame" in manifest:
        source["source_start_frame"] = manifest["source_start_frame"]
        source["source_end_frame"] = manifest["source_start_frame"] + manifest["frame_count"] - 1
    return {
        "schema_version": 1,
        "kind": "reviewed_recording_events",
        "reviewer": _text(reviewer, "event reviewer"),
        "review_scope": "visible_pixels_only",
        "source": source,
        "events": reviewed,
        "limitations": [
            "event names record reviewer-selected visible intervals, not keyboard input",
            "temporal overlap does not establish that one visible event caused another",
            "attack and projectile observations do not identify a skill without separate evidence",
            "a truncated imported session does not describe unimported source frames",
        ]
        + (
            ["sequence and timestamp intervals are local to this imported source-frame window"]
            if "source_start_frame" in manifest
            else []
        ),
    }


def review_events(
    recording: Path,
    output: Path | None,
    *,
    reviewer: str,
    events: Sequence[tuple[str, int, int]],
) -> dict[str, Any]:
    """Bind sparse, reviewer-selected visible intervals to immutable source frames."""
    if output is not None and output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write evidence outside the recording")
    report = _reviewed_event_document(_validated_manifest(recording), reviewer, events)
    if output is not None:
        _write(output, report)
    return report


def _event_report(path: Path) -> dict[str, Any]:
    raw = strict_json(read_regular(path, MAX_EVENT_REPORT_BYTES, "reviewed event report").decode())
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {
            "schema_version",
            "kind",
            "reviewer",
            "review_scope",
            "source",
            "events",
            "limitations",
        }
        or raw["schema_version"] != 1
        or raw["kind"] != "reviewed_recording_events"
        or raw["review_scope"] != "visible_pixels_only"
        or not isinstance(raw["source"], dict)
        or not isinstance(raw["events"], list)
        or not isinstance(raw["limitations"], list)
    ):
        raise ValueError("invalid reviewed event report")
    return raw


def verify_events(recording: Path, report: Path) -> dict[str, Any]:
    """Rebuild every source-derived event identity from the imported recording."""
    raw = _event_report(report)
    selected = []
    for event in raw["events"]:
        if (
            not isinstance(event, dict)
            or set(event) != {"kind", "start", "end", "duration_ms"}
            or not isinstance(event["start"], dict)
            or not isinstance(event["end"], dict)
        ):
            raise ValueError("invalid reviewed visible event")
        kind = event["kind"]
        start = event["start"].get("sequence")
        end = event["end"].get("sequence")
        if not isinstance(kind, str) or type(start) is not int or type(end) is not int:
            raise ValueError("invalid reviewed visible event")
        selected.append((kind, start, end))
    measured = _reviewed_event_document(_validated_manifest(recording), raw["reviewer"], selected)
    if canonical_json(raw) != canonical_json(measured):
        raise ValueError("reviewed event report differs from the recording")
    return {
        "verified": True,
        "scope": "source_frame_identity_and_reviewed_visible_intervals",
        "session_id": measured["source"]["session_id"],
        "events": len(measured["events"]),
    }


def _recording_window_document(recordings: Sequence[Path]) -> dict[str, Any]:
    if (
        not isinstance(recordings, (list, tuple))
        or not 2 <= len(recordings) <= MAX_RECORDING_WINDOWS
    ):
        raise ValueError("supply 2..256 recording windows")
    manifests = [_validated_manifest(Path(recording)) for recording in recordings]
    if any(
        manifest["source_kind"] != "mp4"
        or manifest["source_hash_basis"] != "mp4_file"
        or manifest["timebase"] != "video_presentation_ticks"
        for manifest in manifests
    ):
        raise ValueError("window coverage requires source-PTS MP4 sessions")
    shared_fields = (
        "source_sha256",
        "source_name",
        "source_uri",
        "permission",
        "license",
        "timebase",
        "video_clock",
        "width",
        "height",
    )
    reference = manifests[0]
    if any(
        any(manifest[field] != reference[field] for field in shared_fields)
        for manifest in manifests[1:]
    ):
        raise ValueError("recording windows have inconsistent source identity or provenance")
    ordered = sorted(manifests, key=lambda manifest: manifest.get("source_start_frame", 0))
    if len({manifest["session_id"] for manifest in ordered}) != len(ordered):
        raise ValueError("recording window sessions must be distinct")
    rows = []
    previous_end = -1
    previous_pts: int | None = None
    for index, manifest in enumerate(ordered):
        start = manifest.get("source_start_frame", 0)
        end = start + manifest["frame_count"] - 1
        if type(start) is not int or start != previous_end + 1:
            raise ValueError("recording windows must start at zero and be contiguous")
        if index < len(ordered) - 1:
            if not manifest["truncated"] or manifest["completion_reason"] != "frame_limit":
                raise ValueError("every non-final recording window must end at its frame limit")
        elif manifest["truncated"] or manifest["completion_reason"] != "decoder_stopped":
            raise ValueError("the final recording window must reach the decoder stop")
        first, last = manifest["frames"][0], manifest["frames"][-1]
        first_pts, last_pts = first["pts"], last["pts"]
        if previous_pts is not None and first_pts <= previous_pts:
            raise ValueError("presentation timestamps must increase across recording windows")
        rows.append(
            {
                "session_id": manifest["session_id"],
                "schema_version": manifest["schema_version"],
                "source_start_frame": start,
                "source_end_frame": end,
                "frame_count": manifest["frame_count"],
                "truncated": manifest["truncated"],
                "completion_reason": manifest["completion_reason"],
                "first": {
                    "source_name": first["source_name"],
                    "pts": first_pts,
                    "rgb_sha256": first["rgb_sha256"],
                },
                "last": {
                    "source_name": last["source_name"],
                    "pts": last_pts,
                    "rgb_sha256": last["rgb_sha256"],
                },
            }
        )
        previous_end = end
        previous_pts = last_pts
    document = {
        "schema_version": 1,
        "kind": "recording_window_coverage",
        "source": {field: reference[field] for field in shared_fields},
        "windows": rows,
        "window_count": len(rows),
        "decoded_frame_count": previous_end + 1,
        "decoded_source_range": [0, previous_end],
        "contiguous": True,
        "decoder_stop_reached": True,
        "limitations": [
            "decoder stop does not independently prove that the MP4 container is undamaged",
            "coverage verifies decoded frame identity and timing, not exhaustive gameplay labels",
            "visible events do not identify skills, actors, targets, keyboard input, or causality",
        ],
    }
    document["coverage_sha256"] = hashlib.sha256(canonical_json(document).encode()).hexdigest()
    return document


def summarize_recording_windows(recordings: Sequence[Path], output: Path | None) -> dict[str, Any]:
    """Bind a contiguous, bounded MP4 import chain into one source coverage report."""
    report = _recording_window_document(recordings)
    if output is not None:
        _write(output, report)
    return report


def verify_recording_windows(recordings: Sequence[Path], report: Path) -> dict[str, object]:
    raw = strict_json(
        read_regular(report, MAX_WINDOW_REPORT_BYTES, "recording window report").decode()
    )
    measured = _recording_window_document(recordings)
    if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(measured):
        raise ValueError("recording window report differs from its source sessions")
    return {
        "verified": True,
        "coverage_sha256": measured["coverage_sha256"],
        "windows": measured["window_count"],
        "decoded_frames": measured["decoded_frame_count"],
    }


def _case(recording: Path, labels: Path, available_frames: int) -> Case:
    manifest = _validated_manifest(recording)
    if manifest["frame_count"] > available_frames:
        raise ValueError("total recording frame budget exceeded")
    raw = read_regular(labels, MAX_LABEL_BYTES, "frame labels")
    document = strict_json(raw.decode())
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "kind", "session_id", "domain", "reviewer", "frames"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != "recording_object_labels"
        or document["session_id"] != manifest["session_id"]
    ):
        raise ValueError("label format or recording session mismatch")
    domain = document["domain"]
    if domain not in ("authored_fixture", "game_recording"):
        raise ValueError("declare the recording domain")
    reviewer = _text(document["reviewer"], "reviewer")
    frames = document["frames"]
    if not isinstance(frames, list) or len(frames) != manifest["frame_count"]:
        raise ValueError("labels must cover every imported frame")
    result = []
    for entry, frame in zip(frames, manifest["frames"], strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sequence", "rgb_sha256", "map_id", "objects"}
            or type(entry["sequence"]) is not int
            or entry["sequence"] != frame["sequence"]
            or entry["rgb_sha256"] != frame["rgb_sha256"]
        ):
            raise ValueError("label frame identity differs from captured pixels")
        map_id = _text(entry["map_id"], "map ID")
        objects = entry["objects"]
        if not isinstance(objects, list) or len(objects) > MAX_OBJECTS_PER_FRAME:
            raise ValueError("each frame needs reviewed objects, with [] for reviewed empty frames")
        boxes = []
        occlusions = []
        for item in objects:
            if (
                not isinstance(item, dict)
                or set(item) != {"class_id", "bbox", "occluded"}
                or item["class_id"] not in CLASS_NAMES
                or not isinstance(item["bbox"], list)
                or type(item["occluded"]) is not bool
            ):
                raise ValueError("invalid object label")
            box = Detection(item["class_id"], tuple(item["bbox"]), 1.0)
            if box.bbox[2] > manifest["width"] or box.bbox[3] > manifest["height"]:
                raise ValueError("label exceeds recording bounds")
            boxes.append(box)
            occlusions.append(item["occluded"])
        result.append((map_id, tuple(boxes), tuple(occlusions)))
    return Case(
        recording, manifest, hashlib.sha256(raw).hexdigest(), reviewer, domain, tuple(result)
    )


def _recall_groups(
    sample: EvaluationSample,
    flags: tuple[bool, ...],
    threshold: float,
    totals: Counter[str],
    matched: Counter[str],
) -> None:
    # Match over all instances first, so visible objects cannot be counted as hits
    # on an occluded neighbor. Ties follow the input order of checked annotations.
    hits: set[int] = set()
    for pred in sorted(sample.predictions, key=lambda d: -d.confidence):
        if pred.confidence < threshold:
            continue
        candidates = [
            (_iou(pred, obj), i)
            for i, obj in enumerate(sample.ground_truth)
            if i not in hits and obj.class_id == pred.class_id
        ]
        if candidates:
            score, i = max(candidates, key=lambda x: (x[0], -x[1]))
            if score >= 0.5:
                hits.add(i)
    for i, obj in enumerate(sample.ground_truth):
        x1, y1, x2, y2 = obj.bbox
        for group in (
            "occluded" if flags[i] else "unoccluded",
            "small" if (x2 - x1) * (y2 - y1) < 32**2 else "other_size",
        ):
            totals[group] += 1
            matched[group] += int(i in hits)


def _development_split(
    dataset: Path, recording_pixels: set[tuple[int, int, str]]
) -> dict[str, Any]:
    root = dataset.resolve()
    manifest_path = root / "manifest.json"
    original = _bounded_manifest_bytes(manifest_path)
    validate_dataset(root)
    doc = strict_json(original.decode())
    count = 0
    for row in doc["samples"]:
        if row["split"] not in ("train", "val"):
            continue
        path = _contained(root, row["image"])
        image_bytes = read_regular(path, 64 * 1024 * 1024, "development image")
        if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError("development image differs from the validated manifest")
        with Image.open(io.BytesIO(image_bytes)) as image:
            key = (
                image.width,
                image.height,
                hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
            )
        if key in recording_pixels:
            raise ValueError("recording pixels overlap the declared train/val data")
        count += 1
    if original != _bounded_manifest_bytes(manifest_path):
        raise ValueError("development manifest changed during evaluation preflight")
    return {"manifest_sha256": hashlib.sha256(original).hexdigest(), "train_val_frames": count}


def benchmark_recordings(
    pairs: Sequence[tuple[Path, Path]],
    output: Path,
    detector: Detector | None = None,
    *,
    confidence: float = 0.25,
    development_datasets: Sequence[Path] = (),
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"report already exists: {output}")
    if not 1 <= len(pairs) <= 32:
        raise ValueError("a benchmark needs 1..32 labeled recording sessions")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be in [0,1]")
    cases: list[Case] = []
    sources: set[str] = set()
    frame_count = truth_count = 0
    for recording, labels in pairs:
        case = _case(recording, labels, MAX_FRAMES - frame_count)
        if case.manifest["source_sha256"] in sources:
            raise ValueError("duplicate source recording")
        if cases and case.domain != cases[0].domain:
            raise ValueError("score fixtures and game recordings in separate reports")
        sources.add(case.manifest["source_sha256"])
        cases.append(case)
        frame_count += len(case.annotations)
        truth_count += sum(len(row[1]) for row in case.annotations)
        if truth_count > MAX_OBJECTS:
            raise ValueError("ground-truth object budget exceeded")

    if len(development_datasets) > 16:
        raise ValueError("up to 16 development datasets can be checked")
    roots = [case.recording for case in cases] + list(development_datasets)
    if any(output.resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError("write benchmark results outside input bundles")
    recording_pixels = {
        (case.manifest["width"], case.manifest["height"], frame["rgb_sha256"])
        for case in cases
        for frame in case.manifest["frames"]
    }
    development = [_development_split(path, recording_pixels) for path in development_datasets]
    engine: Detector = PixelPerception() if detector is None else detector
    metadata = _detector_metadata(engine, include_runtime=True)
    samples = []
    by_map: dict[str, list[EvaluationSample]] = {}
    sessions = []
    times = []
    seen_rgb: set[str] = set()
    cross_session_duplicates: set[str] = set()
    predictions = 0
    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for case in cases:
        session_samples = []
        rgb = {frame["rgb_sha256"] for frame in case.manifest["frames"]}
        cross_session_duplicates.update(seen_rgb & rgb)
        seen_rgb.update(rgb)
        for item, (map_id, truth, flags) in zip(
            case.manifest["frames"], case.annotations, strict=True
        ):
            frame = _load_frame(case.recording, item)
            started = time.perf_counter()
            result = engine.detect(frame)
            times.append((time.perf_counter() - started) * 1000)
            if not isinstance(result, tuple) or len(result) > MAX_OBJECTS_PER_FRAME:
                raise ValueError("prediction count exceeds the per-frame budget")
            sample = EvaluationSample(truth, result)
            if any(d.bbox[2] > frame.width or d.bbox[3] > frame.height for d in result):
                raise ValueError("prediction exceeds recording bounds")
            predictions += len(result)
            if predictions > MAX_OBJECTS:
                raise ValueError("prediction budget exceeded")
            _recall_groups(sample, flags, confidence, totals, matched)
            samples.append(sample)
            session_samples.append(sample)
            by_map.setdefault(map_id, []).append(sample)
        sessions.append(
            {
                "session_id": case.manifest["session_id"],
                "source_sha256": case.manifest["source_sha256"],
                "labels_sha256": case.label_sha256,
                "reviewer": case.reviewer,
                "frames": len(session_samples),
                "truncated": case.manifest["truncated"],
                "timebase": case.manifest["timebase"],
                "source_uri": case.manifest["source_uri"],
                "permission": case.manifest["permission"],
                "metrics": evaluate_predictions(session_samples, confidence_threshold=confidence),
            }
        )
    if _detector_metadata(engine, include_runtime=True) != metadata:
        raise ValueError("detector identity changed during the benchmark")
    levels = [
        evaluate_predictions(samples, confidence_threshold=confidence, iou_threshold=i / 100)
        for i in range(50, 96, 5)
    ]
    measured = [float(v) for level in levels if isinstance(v := level["map_at_iou"], (float, int))]
    report = {
        "schema_version": 1,
        "kind": "recording_detection_benchmark",
        "declared_domain": cases[0].domain,
        "frames": frame_count,
        "session_count": len(cases),
        "ground_truth_objects": truth_count,
        "predicted_objects": predictions,
        "detector": metadata,
        "environment": _environment(),
        "sessions": sessions,
        "metrics_at_50": levels[0],
        "mean_ap_50_95": statistics.mean(measured) if measured else None,
        "ap_by_iou": [
            {"iou": level["iou_threshold"], "mean_ap": level["map_at_iou"]} for level in levels
        ],
        "ap_definition": "AP101 after confidence filtering, macro over labeled classes",
        "by_map": {
            k: evaluate_predictions(v, confidence_threshold=confidence)
            for k, v in sorted(by_map.items())
        },
        "recall_at_50": {
            group: {
                "matched": matched[group],
                "total": totals[group],
                "recall": matched[group] / totals[group] if totals[group] else None,
            }
            for group in ("small", "other_size", "occluded", "unoccluded")
        },
        "exact_rgb_hashes_shared_across_sessions": len(cross_session_duplicates),
        "training_overlap_verified": False,
        "declared_development_exact_rgb_checks": development,
        "action_outcomes_evaluated": False,
        "latency_ms": {
            "scope": "detector.detect including first call",
            "median": statistics.median(times),
            "p95": sorted(times)[(len(times) * 95 + 99) // 100 - 1],
        },
    }
    _write(output, report)
    return report


def audit_route(recording: Path, labels: Path, graph_path: Path, output: Path) -> dict[str, Any]:
    """Compare reviewed map changes between adjacent captured frames with a static graph."""
    from .navigation_graph import load_world_graph

    if output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write the audit outside the captured recording")
    case = _case(recording, labels, MAX_FRAMES)
    graph = load_world_graph(graph_path)
    frames = case.manifest["frames"]
    edges = {(p.source, p.target) for p in graph.portals}
    changes: list[dict[str, Any]] = []
    previous = case.annotations[0][0]
    outside_graph = set()
    for index, (current, _, _) in enumerate(case.annotations):
        if current not in graph.nodes:
            outside_graph.add(current)
        if current != previous:
            changes.append(
                {
                    "source": previous,
                    "target": current,
                    "before_sequence": frames[index - 1]["sequence"],
                    "after_sequence": frames[index]["sequence"],
                    "before_ms": frames[index - 1]["timestamp_ms"],
                    "after_ms": frames[index]["timestamp_ms"],
                    "listed_direct_edge": (previous, current) in edges,
                }
            )
        previous = current
    report = {
        "kind": "reviewed_map_transition_audit",
        "schema_version": 1,
        "session_id": case.manifest["session_id"],
        "domain": case.domain,
        "reviewer": case.reviewer,
        "label_sha256": case.label_sha256,
        "graph_sha256": graph.content_sha256,
        "timebase": case.manifest["timebase"],
        "truncated": case.manifest["truncated"],
        "frames": len(frames),
        "observed_changes": len(changes),
        "listed_direct_changes": sum(change["listed_direct_edge"] for change in changes),
        "map_ids_outside_graph": sorted(outside_graph),
        "changes": changes,
    }
    _write(output, report)
    return report


def measure_strike(
    recording: Path,
    labels: Path,
    output: Path | None,
    *,
    start: int,
    end: int,
    player_index: int,
    monster_index: int,
    level: int,
    skill_id: str,
    weapon: str,
    reviewer: str,
    facing: str,
    feedback: str,
    feedback_frames: Sequence[int] = (),
    movement_frame: int | None = None,
    weapon_speed_grade: int | None = None,
    speed_effects: Sequence[str] = (),
    input_profile: InputBindingProfile | None = None,
    input_key: str | None = None,
) -> dict[str, Any]:
    """Measure a reviewer-selected cast window in the recording's native pixel space."""
    if output is not None and output.resolve().is_relative_to(recording.resolve()):
        raise ValueError("write evidence outside the recording")
    report = _measure_strike_window(
        _case(recording, labels, MAX_FRAMES),
        start=start,
        end=end,
        player_index=player_index,
        monster_index=monster_index,
        level=level,
        skill_id=skill_id,
        weapon=weapon,
        reviewer=reviewer,
        facing=facing,
        feedback=feedback,
        feedback_frames=feedback_frames,
        movement_frame=movement_frame,
        weapon_speed_grade=weapon_speed_grade,
        speed_effects=speed_effects,
        input_profile=input_profile,
        input_key=input_key,
    )
    if output is not None:
        _write(output, report)
    return report


def _measure_strike_window(
    case: Case,
    *,
    start: int,
    end: int,
    player_index: int,
    monster_index: int,
    level: int,
    skill_id: str,
    weapon: str,
    reviewer: str,
    facing: str,
    feedback: str,
    feedback_frames: Sequence[int] = (),
    movement_frame: int | None = None,
    weapon_speed_grade: int | None = None,
    speed_effects: Sequence[str] = (),
    input_profile: InputBindingProfile | None = None,
    input_key: str | None = None,
) -> dict[str, Any]:
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end < len(case.annotations)
    ):
        raise ValueError("invalid strike frame interval")
    if type(level) is not int or level < 1 or facing not in ("left", "right"):
        raise ValueError("invalid skill level or facing")
    if feedback not in ("damage_visible", "miss_text_visible", "no_feedback"):
        raise ValueError("invalid feedback category")
    if weapon_speed_grade is not None and (
        type(weapon_speed_grade) is not int or not 2 <= weapon_speed_grade <= 9
    ):
        raise ValueError("weapon speed grade must be an integer from 2 through 9")
    if not isinstance(speed_effects, (list, tuple)) or len(speed_effects) > 8:
        raise ValueError("invalid reviewed speed effect sequence")
    reviewed_effects = [_text(effect, "speed effect") for effect in speed_effects]
    if (
        any(effect != effect.strip() for effect in reviewed_effects)
        or len(set(reviewed_effects)) != len(reviewed_effects)
        or (reviewed_effects and weapon_speed_grade is None)
    ):
        raise ValueError("invalid reviewed speed effect sequence")
    reviewed_effects.sort()
    if (input_profile is None) != (input_key is None):
        raise ValueError("input profile and input key must be supplied together")
    reviewed_binding = None
    if input_profile is not None:
        if not isinstance(input_profile, InputBindingProfile):
            raise ValueError("input profile must be a validated binding profile")
        assert input_key is not None
        binding = input_profile.binding_for_key(input_key)
        if binding is None:
            raise ValueError("input key is absent from the reviewed binding profile")
        manifest = input_profile.manifest()
        reviewed_binding = {
            "scope": "reviewed_key_layout_context",
            "profile": manifest,
            "profile_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
            "key": binding.key,
            "normalized_key": normalize_key(binding.key),
            "configured_skill": binding.skill,
        }
    if (
        not isinstance(feedback_frames, (list, tuple))
        or len(feedback_frames) > 16
        or any(type(index) is not int or not start <= index <= end for index in feedback_frames)
        or list(feedback_frames) != sorted(set(feedback_frames))
        or (feedback == "no_feedback" and feedback_frames)
    ):
        raise ValueError("invalid reviewed feedback frame sequence")
    if movement_frame is not None and (
        type(movement_frame) is not int
        or not feedback_frames
        or not feedback_frames[-1] <= movement_frame <= end
    ):
        raise ValueError("movement frame must follow or coincide with final reviewed feedback")
    map_id, objects, occluded = case.annotations[start]
    if any(entry[0] != map_id for entry in case.annotations[start : end + 1]):
        raise ValueError("strike interval crosses a reviewed map change")
    for index, category in ((player_index, "player"), (monster_index, "monster")):
        if (
            type(index) is not int
            or not 0 <= index < len(objects)
            or objects[index].class_id != category
        ):
            raise ValueError(f"select a reviewed {category} object in the start frame")
    p, m = objects[player_index].bbox, objects[monster_index].bbox
    first, last = (case.manifest["frames"][i] for i in (start, end))
    sign = 1 if facing == "right" else -1
    report = {
        "schema_version": 2 if feedback_frames else 1,
        "kind": "reviewed_strike_window",
        "session_id": case.manifest["session_id"],
        "domain": case.domain,
        "labels_sha256": case.label_sha256,
        "label_reviewer": case.reviewer,
        "strike_reviewer": _text(reviewer, "strike reviewer"),
        "skill_id": _text(skill_id, "skill ID"),
        "level": level,
        "weapon": _text(weapon, "weapon"),
        "facing": facing,
        "map_id": map_id,
        "reviewed_feedback": feedback,
        "timebase": case.manifest["timebase"],
        "start": {k: first[k] for k in ("sequence", "timestamp_ms", "rgb_sha256")},
        "end": {k: last[k] for k in ("sequence", "timestamp_ms", "rgb_sha256")},
        "window_ms": last["timestamp_ms"] - first["timestamp_ms"],
        "source_frame_size": [case.manifest["width"], case.manifest["height"]],
        "start_geometry": {
            "player_index": player_index,
            "monster_index": monster_index,
            "player_bbox": list(p),
            "monster_bbox": list(m),
            "player_occluded": occluded[player_index],
            "monster_occluded": occluded[monster_index],
            "forward_center_delta_px": sign * (m[0] + m[2] - p[0] - p[2]) / 2,
            "vertical_center_delta_px": (m[1] + m[3] - p[1] - p[3]) / 2,
            "horizontal_edge_gap_px": max(0, p[0] - m[2], m[0] - p[2]),
        },
    }
    if feedback_frames:
        previous_ms: int | None = None
        report["feedback_frames"] = []
        for index in feedback_frames:
            frame = case.manifest["frames"][index]
            timestamp_ms = frame["timestamp_ms"]
            report["feedback_frames"].append(
                {
                    **{key: frame[key] for key in ("sequence", "timestamp_ms", "rgb_sha256")},
                    "offset_ms": timestamp_ms - first["timestamp_ms"],
                    "gap_ms": None if previous_ms is None else timestamp_ms - previous_ms,
                }
            )
            previous_ms = timestamp_ms
    if movement_frame is not None:
        movement = case.manifest["frames"][movement_frame]
        report["schema_version"] = 3
        report["reviewed_movement"] = {
            **{k: movement[k] for k in ("sequence", "timestamp_ms", "rgb_sha256")},
            "after_last_feedback_ms": movement["timestamp_ms"]
            - case.manifest["frames"][feedback_frames[-1]]["timestamp_ms"],
        }
    if weapon_speed_grade is not None:
        report["schema_version"] = 4
        report["attack_context"] = {
            "weapon_speed_grade": weapon_speed_grade,
            "speed_effects": reviewed_effects,
        }
    if reviewed_binding is not None:
        report["schema_version"] = 5
        report["input_binding"] = reviewed_binding
    return report


def verify_strike(recording: Path, labels: Path, report_path: Path) -> dict[str, Any]:
    """Recompute source-derived fields while retaining the reviewer's selections."""
    raw, _ = _strike_report(report_path)
    _verify_strike_source(_case(recording, labels, MAX_FRAMES), raw)
    return {
        "source_fields_verified": True,
        "session_id": raw["session_id"],
        "domain": raw["domain"],
        "labels_sha256": raw["labels_sha256"],
    }


def _verify_strike_source(case: Case, raw: dict[str, Any]) -> None:
    geometry = raw["start_geometry"]
    measured = _measure_strike_window(
        case,
        start=raw["start"]["sequence"],
        end=raw["end"]["sequence"],
        player_index=geometry["player_index"],
        monster_index=geometry["monster_index"],
        level=raw["level"],
        skill_id=raw["skill_id"],
        weapon=raw["weapon"],
        reviewer=raw["strike_reviewer"],
        facing=raw["facing"],
        feedback=raw["reviewed_feedback"],
        movement_frame=raw.get("reviewed_movement", {}).get("sequence"),
        feedback_frames=[frame["sequence"] for frame in raw.get("feedback_frames", [])],
        weapon_speed_grade=raw.get("attack_context", {}).get("weapon_speed_grade"),
        speed_effects=raw.get("attack_context", {}).get("speed_effects", []),
        input_profile=(
            InputBindingProfile.from_manifest(raw["input_binding"]["profile"])
            if "input_binding" in raw
            else None
        ),
        input_key=raw.get("input_binding", {}).get("key"),
    )
    if canonical_json(raw) != canonical_json(measured):
        raise ValueError("strike report differs from recording or reviewed labels")


def _strike_report(path: Path) -> tuple[dict[str, Any], str]:
    contents = read_regular(path, MAX_STRIKE_REPORT_BYTES, "strike report")
    raw = strict_json(contents.decode())
    required = {
        "schema_version",
        "kind",
        "session_id",
        "domain",
        "labels_sha256",
        "label_reviewer",
        "strike_reviewer",
        "skill_id",
        "level",
        "weapon",
        "facing",
        "map_id",
        "reviewed_feedback",
        "timebase",
        "start",
        "end",
        "window_ms",
        "source_frame_size",
        "start_geometry",
    }
    if isinstance(raw, dict) and raw.get("schema_version") in (2, 3):
        required.add("feedback_frames")
        if raw["schema_version"] == 3:
            required.add("reviewed_movement")
    if isinstance(raw, dict) and raw.get("schema_version") == 4:
        required.add("attack_context")
        if "feedback_frames" in raw:
            required.add("feedback_frames")
        if "reviewed_movement" in raw:
            required.add("feedback_frames")
            required.add("reviewed_movement")
    if isinstance(raw, dict) and raw.get("schema_version") == 5:
        required.add("input_binding")
        if "attack_context" in raw:
            required.add("attack_context")
        if "feedback_frames" in raw:
            required.add("feedback_frames")
        if "reviewed_movement" in raw:
            required.add("feedback_frames")
            required.add("reviewed_movement")
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid strike report schema")
    if raw["schema_version"] not in (1, 2, 3, 4, 5) or raw["kind"] != "reviewed_strike_window":
        raise ValueError("unsupported strike report")
    if raw["domain"] not in ("authored_fixture", "game_recording"):
        raise ValueError("invalid strike report domain")
    if raw["facing"] not in ("left", "right") or raw["reviewed_feedback"] not in (
        "damage_visible",
        "miss_text_visible",
        "no_feedback",
    ):
        raise ValueError("invalid strike review category")
    if type(raw["level"]) is not int or raw["level"] < 1:
        raise ValueError("invalid strike skill level")
    if type(raw["window_ms"]) is not int or raw["window_ms"] < 0:
        raise ValueError("invalid strike window")
    if (
        not isinstance(raw["source_frame_size"], list)
        or len(raw["source_frame_size"]) != 2
        or any(type(value) is not int or value < 1 for value in raw["source_frame_size"])
    ):
        raise ValueError("invalid strike frame size")
    geometry = raw["start_geometry"]
    geometry_fields = {
        "player_index",
        "monster_index",
        "player_bbox",
        "monster_bbox",
        "player_occluded",
        "monster_occluded",
        "forward_center_delta_px",
        "vertical_center_delta_px",
        "horizontal_edge_gap_px",
    }
    if not isinstance(geometry, dict) or set(geometry) != geometry_fields:
        raise ValueError("invalid strike geometry")
    for field in ("forward_center_delta_px", "vertical_center_delta_px", "horizontal_edge_gap_px"):
        if not isinstance(geometry[field], (int, float)) or isinstance(geometry[field], bool):
            raise ValueError("invalid strike distance")
    for frame in (raw["start"], raw["end"]):
        if not isinstance(frame, dict) or set(frame) != {"sequence", "timestamp_ms", "rgb_sha256"}:
            raise ValueError("invalid strike frame identity")
        if type(frame["sequence"]) is not int or type(frame["timestamp_ms"]) is not int:
            raise ValueError("invalid strike frame timing")
        if not isinstance(frame["rgb_sha256"], str) or len(frame["rgb_sha256"]) != 64:
            raise ValueError("invalid strike frame hash")
    if "feedback_frames" in raw:
        frames = raw["feedback_frames"]
        if (
            not isinstance(frames, list)
            or not 1 <= len(frames) <= 16
            or raw["reviewed_feedback"] == "no_feedback"
        ):
            raise ValueError("invalid strike feedback frames")
        previous_ms: int | None = None
        previous_sequence: int | None = None
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {
                "sequence",
                "timestamp_ms",
                "rgb_sha256",
                "offset_ms",
                "gap_ms",
            }:
                raise ValueError("invalid strike feedback frame")
            sequence, timestamp = frame["sequence"], frame["timestamp_ms"]
            if (
                type(sequence) is not int
                or type(timestamp) is not int
                or type(frame["offset_ms"]) is not int
                or not isinstance(frame["rgb_sha256"], str)
                or len(frame["rgb_sha256"]) != 64
                or not raw["start"]["sequence"] <= sequence <= raw["end"]["sequence"]
                or not raw["start"]["timestamp_ms"] <= timestamp <= raw["end"]["timestamp_ms"]
                or frame["offset_ms"] != timestamp - raw["start"]["timestamp_ms"]
                or (previous_sequence is not None and sequence <= previous_sequence)
                or (previous_ms is not None and timestamp < previous_ms)
            ):
                raise ValueError("invalid strike feedback frame identity or timing")
            expected_gap = None if previous_ms is None else timestamp - previous_ms
            if (expected_gap is None and frame["gap_ms"] is not None) or (
                expected_gap is not None
                and (type(frame["gap_ms"]) is not int or frame["gap_ms"] != expected_gap)
            ):
                raise ValueError("invalid strike feedback gap")
            previous_sequence, previous_ms = sequence, timestamp
    if "reviewed_movement" in raw:
        movement = raw["reviewed_movement"]
        final = raw["feedback_frames"][-1]
        if (
            not isinstance(movement, dict)
            or set(movement) != {"sequence", "timestamp_ms", "rgb_sha256", "after_last_feedback_ms"}
            or type(movement["sequence"]) is not int
            or type(movement["timestamp_ms"]) is not int
            or type(movement["after_last_feedback_ms"]) is not int
            or not final["sequence"] <= movement["sequence"] <= raw["end"]["sequence"]
            or not final["timestamp_ms"] <= movement["timestamp_ms"] <= raw["end"]["timestamp_ms"]
            or movement["after_last_feedback_ms"]
            != movement["timestamp_ms"] - final["timestamp_ms"]
            or not isinstance(movement["rgb_sha256"], str)
            or len(movement["rgb_sha256"]) != 64
        ):
            raise ValueError("invalid reviewed movement frame")
    if "attack_context" in raw:
        context = raw["attack_context"]
        if (
            not isinstance(context, dict)
            or set(context) != {"weapon_speed_grade", "speed_effects"}
            or type(context["weapon_speed_grade"]) is not int
            or not 2 <= context["weapon_speed_grade"] <= 9
            or not isinstance(context["speed_effects"], list)
            or len(context["speed_effects"]) > 8
            or any(
                not isinstance(effect, str)
                or not effect.strip()
                or effect != effect.strip()
                or len(effect) > 128
                or any(ord(character) < 32 for character in effect)
                for effect in context["speed_effects"]
            )
            or context["speed_effects"] != sorted(set(context["speed_effects"]))
        ):
            raise ValueError("invalid reviewed attack context")
    if raw["schema_version"] == 5:
        context = raw["input_binding"]
        if not isinstance(context, dict) or set(context) != {
            "scope",
            "profile",
            "profile_sha256",
            "key",
            "normalized_key",
            "configured_skill",
        }:
            raise ValueError("invalid reviewed input binding")
        try:
            profile = InputBindingProfile.from_manifest(context["profile"])
            binding = profile.binding_for_key(context["key"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid reviewed input binding") from exc
        digest = hashlib.sha256(canonical_json(profile.manifest()).encode()).hexdigest()
        if (
            context["scope"] != "reviewed_key_layout_context"
            or context["profile_sha256"] != digest
            or binding is None
            or context["normalized_key"] != normalize_key(binding.key)
            or context["configured_skill"] != binding.skill
        ):
            raise ValueError("invalid reviewed input binding")
    return raw, hashlib.sha256(contents).hexdigest()


def summarize_strikes(
    reports: Sequence[Path],
    output: Path,
    *,
    sources: Sequence[tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    """Aggregate reviewed observations without promoting them into a calibrated range."""
    if not reports or len(reports) > 10_000:
        raise ValueError("supply 1..10000 strike reports")
    if sources is not None and len(sources) != len(reports):
        raise ValueError("each verified strike report needs its recording and labels")
    if sources and any(output.resolve().is_relative_to(path.resolve()) for path, _ in sources):
        raise ValueError("write evidence outside the recording")
    snapshots = [_strike_report(Path(path)) for path in reports]
    documents = [document for document, _ in snapshots]
    if sources is not None:
        by_source: dict[tuple[Path, Path], list[dict[str, Any]]] = {}
        for source, document in zip(sources, documents, strict=True):
            by_source.setdefault(source, []).append(document)
        for (recording, labels), windows in by_source.items():
            case = _case(recording, labels, MAX_FRAMES)
            for window in windows:
                _verify_strike_source(case, window)
    identity = tuple(documents[0][key] for key in ("domain", "skill_id", "level", "weapon"))
    attack_context = documents[0].get("attack_context")
    input_binding = documents[0].get("input_binding")
    frame_size = documents[0]["source_frame_size"]
    if any(
        tuple(document[key] for key in ("domain", "skill_id", "level", "weapon")) != identity
        for document in documents
    ):
        raise ValueError("strike reports must describe the same domain, skill, level and weapon")
    if any(document.get("attack_context") != attack_context for document in documents):
        raise ValueError("strike reports need the same reviewed attack context")
    if any(document.get("input_binding") != input_binding for document in documents):
        raise ValueError("strike reports need the same reviewed input binding")
    if any(document["source_frame_size"] != frame_size for document in documents):
        raise ValueError("strike reports need the same source frame size")
    timebase = documents[0]["timebase"]
    if any(document["timebase"] != timebase for document in documents):
        raise ValueError("strike reports need the same recording timebase")
    # One cast onset is one sample even when reviewers choose different end frames.
    # Distinct source frames can have identical pixels.
    sample_ids = [(document["session_id"], document["start"]["sequence"]) for document in documents]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate strike observation")

    def forward_edge_gap(item: dict[str, Any]) -> float:
        geometry = item["start_geometry"]
        player = geometry["player_bbox"]
        monster = geometry["monster_bbox"]
        return (
            float(monster[0] - player[2])
            if item["facing"] == "right"
            else float(player[0] - monster[2])
        )

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        forward = [float(item["start_geometry"]["forward_center_delta_px"]) for item in items]
        vertical = [float(item["start_geometry"]["vertical_center_delta_px"]) for item in items]
        forward_edges = []
        vertical_edges = []
        vertical_overlaps = []
        for item in items:
            geometry = item["start_geometry"]
            player = geometry["player_bbox"]
            monster = geometry["monster_bbox"]
            forward_edges.append(forward_edge_gap(item))
            vertical_edges.append(float(max(0, player[1] - monster[3], monster[1] - player[3])))
            vertical_overlaps.append(
                float(max(0, min(player[3], monster[3]) - max(player[1], monster[1])))
            )
        windows = [item["window_ms"] for item in items]
        return {
            "observations": len(items),
            "forward_center_delta_px": {
                "min": min(forward) if forward else None,
                "max": max(forward) if forward else None,
            },
            "vertical_center_delta_px": {
                "min": min(vertical) if vertical else None,
                "max": max(vertical) if vertical else None,
            },
            "forward_edge_gap_px": {
                "min": min(forward_edges) if forward_edges else None,
                "max": max(forward_edges) if forward_edges else None,
            },
            "vertical_edge_gap_px": {
                "min": min(vertical_edges) if vertical_edges else None,
                "max": max(vertical_edges) if vertical_edges else None,
            },
            "vertical_overlap_px": {
                "min": min(vertical_overlaps) if vertical_overlaps else None,
                "max": max(vertical_overlaps) if vertical_overlaps else None,
            },
            "window_ms": {
                "min": min(windows) if windows else None,
                "median": statistics.median(windows) if windows else None,
                "max": max(windows) if windows else None,
            },
        }

    by_feedback = {
        feedback: metrics([item for item in documents if item["reviewed_feedback"] == feedback])
        for feedback in ("damage_visible", "miss_text_visible", "no_feedback")
    }
    sequenced = [item for item in documents if "feedback_frames" in item]
    report = {
        "schema_version": (
            5
            if input_binding is not None
            else (4 if attack_context is not None else (2 if sequenced else 1))
        ),
        "kind": "reviewed_strike_summary",
        "domain": identity[0],
        "skill_id": identity[1],
        "level": identity[2],
        "weapon": identity[3],
        "source_frame_size": frame_size,
        "observations": len(documents),
        "timebase": timebase,
        "sessions": len({item["session_id"] for item in documents}),
        "maps": sorted({item["map_id"] for item in documents}),
        "by_feedback": by_feedback,
        "by_facing": {
            facing: metrics([item for item in documents if item["facing"] == facing])
            for facing in ("left", "right")
        },
        "observed_damage_distance_lower_bound_px": max(
            (
                float(item["start_geometry"]["forward_center_delta_px"])
                for item in documents
                if item["reviewed_feedback"] == "damage_visible"
                and float(item["start_geometry"]["forward_center_delta_px"]) >= 0
            ),
            default=None,
        ),
        "observed_damage_forward_edge_gap_lower_bound_px": max(
            (
                forward_edge_gap(item)
                for item in documents
                if item["reviewed_feedback"] == "damage_visible"
                and float(item["start_geometry"]["forward_center_delta_px"]) >= 0
                and forward_edge_gap(item) >= 0
            ),
            default=None,
        ),
        "range_upper_bound_px": None,
        "calibrated_range": False,
        "limitations": [
            "damage feedback only establishes an observed hit at the measured geometry",
            "MISS text can be an accuracy outcome and does not establish an out-of-range sample",
            "no feedback can result from occlusion, timing or attribution error",
            "distances are native pixels for one fixed source frame size",
            "forward edge gaps use reviewed bounding boxes, not collision or attack hitboxes",
        ],
        "source_reports_sha256": [digest for _, digest in snapshots],
    }
    if attack_context is not None:
        report["attack_context"] = attack_context
    if input_binding is not None:
        report["input_binding"] = input_binding
        report["limitations"].append(
            "the reviewed key profile associates this cast with a configured key; "
            "recording pixels do not prove keyboard input"
        )
    if sources is not None:
        report["source_fields_verified"] = len(documents)
        ordered = sorted(
            snapshots, key=lambda item: (item[0]["session_id"], item[0]["start"]["sequence"])
        )
        intervals = []
        for (first, first_digest), (second, second_digest) in zip(ordered, ordered[1:]):
            if first["session_id"] != second["session_id"] or first["map_id"] != second["map_id"]:
                continue
            intervals.append(
                {
                    "session_id": first["session_id"],
                    "map_id": first["map_id"],
                    "first_report_sha256": first_digest,
                    "second_report_sha256": second_digest,
                    "first_sequence": first["start"]["sequence"],
                    "second_sequence": second["start"]["sequence"],
                    "elapsed_ms": second["start"]["timestamp_ms"] - first["start"]["timestamp_ms"],
                }
            )
        report["reviewed_onset_intervals"] = {"count": len(intervals), "pairs": intervals}
        report["limitations"].append(
            "reviewed onsets may omit intervening casts or include player pauses; "
            "their spacing is not a calibrated attack period"
        )
    if sequenced:
        counts = [len(item["feedback_frames"]) for item in sequenced]
        first_offsets = [item["feedback_frames"][0]["offset_ms"] for item in sequenced]
        gaps = [frame["gap_ms"] for item in sequenced for frame in item["feedback_frames"][1:]]
        report["feedback_sequences"] = {
            "observations": len(sequenced),
            "event_count_distribution": {
                str(count): counts.count(count) for count in sorted(set(counts))
            },
            "first_feedback_offset_ms": {
                "min": min(first_offsets),
                "median": statistics.median(first_offsets),
                "max": max(first_offsets),
            },
            "inter_feedback_gap_ms": {
                "observations": len(gaps),
                "min": min(gaps) if gaps else None,
                "median": statistics.median(gaps) if gaps else None,
                "max": max(gaps) if gaps else None,
            },
            "calibrated_timing": False,
        }
        report["limitations"].append(
            "reviewed feedback frames are visible event evidence, not guaranteed "
            "one-to-one hit packets"
        )
    delays = [
        item["reviewed_movement"]["after_last_feedback_ms"]
        for item in sequenced
        if "reviewed_movement" in item
    ]
    if delays:
        if attack_context is None and input_binding is None:
            report["schema_version"] = 3
        report["observed_movement_after_feedback_ms"] = {
            "observations": len(delays),
            "min": min(delays),
            "median": statistics.median(delays),
            "max": max(delays),
        }
        report["limitations"].append("visible movement depends on player input and frame sampling")
    _write(output, report)
    return report
