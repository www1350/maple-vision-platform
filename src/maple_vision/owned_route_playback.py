"""Verified playback and takeover index for the project-owned two-map route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RoutePlaybackFrame:
    source_sequence: int
    path: Path
    labels: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_route_playback(
    output: Path, review: dict[str, Any], *, stride: int = 4
) -> tuple[RoutePlaybackFrame, ...]:
    """Select regular and exact event frames after checking the route index sources."""
    if type(stride) is not int or not 1 <= stride <= 60:
        raise ValueError("route playback stride must be an integer in [1, 60]")
    if review.get("scope") not in {
        "project_owned_synthetic_route",
        "project_owned_live_synthetic_route",
    } or review.get("preview") != "preview.gif":
        raise ValueError("route playback requires one owned-route review")
    if review.get("scope") == "project_owned_synthetic_route" and review.get(
        "status"
    ) != "completed":
        raise ValueError("planned route playback requires a completed source route")
    if review.get("scope") == "project_owned_live_synthetic_route" and review.get(
        "status"
    ) not in {"completed", "taken_over"}:
        raise ValueError("live route playback has an invalid terminal status")
    expected = review.get("source_files")
    if not isinstance(expected, dict) or set(expected) != {
        "summary.json",
        "trace.jsonl",
        "preview.gif",
    }:
        raise ValueError("route review source bindings are incomplete")
    for name, digest in expected.items():
        if not isinstance(digest, str) or _sha256(output / name) != digest:
            raise ValueError(f"route review source changed: {name}")

    labels: dict[int, list[str]] = {}
    for item in review.get("navigation", []):
        sequence = item.get("sequence")
        if type(sequence) is not int:
            raise ValueError("invalid navigation source sequence")
        labels.setdefault(sequence, []).append("layer_reached")
    for item in review.get("casts", []):
        sequence = item.get("source_sequence")
        if type(sequence) is not int:
            raise ValueError("invalid cast source sequence")
        labels.setdefault(sequence, []).append(f"skill_dispatched:{item.get('key')}")

    source_frames: dict[int, Path] = {}
    for path in sorted((output / "frames").glob("*.png")):
        if not path.stem.isdecimal():
            raise ValueError("invalid route frame name")
        source_frames[int(path.stem)] = path
    if not source_frames:
        raise ValueError("route playback frames are missing")
    if any(sequence not in source_frames for sequence in labels):
        raise ValueError("route event frame is missing")
    maximum = max(source_frames)
    selected = {
        sequence
        for sequence in source_frames
        if sequence % stride == 0 or sequence in labels or sequence == maximum
    }
    return tuple(
        RoutePlaybackFrame(sequence, source_frames[sequence], tuple(labels.get(sequence, ())))
        for sequence in sorted(selected)
    )


def finalize_route_playback(
    output: Path,
    review: dict[str, Any],
    displayed: tuple[tuple[int, Image.Image], ...],
    *,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    """Write one replay and timeline for either completion or player interruption."""
    if status not in {"completed", "taken_over"}:
        raise ValueError("invalid route playback status")
    if status == "taken_over" and not isinstance(reason, str):
        raise ValueError("taken-over playback requires a reason")
    if status == "completed" and reason is not None:
        raise ValueError("completed playback cannot have a stop reason")
    if (
        not displayed
        or any(
            type(sequence) is not int or not isinstance(image, Image.Image)
            for sequence, image in displayed
        )
        or any(
            displayed[index][0] >= displayed[index + 1][0] for index in range(len(displayed) - 1)
        )
    ):
        raise ValueError("displayed route frames must be a non-empty increasing sequence")

    last_sequence = displayed[-1][0]
    execution = review.get("execution")
    if execution is not None and (
        not isinstance(execution, dict)
        or execution.get("status") != status
        or execution.get("last_source_sequence") != last_sequence
    ):
        raise ValueError("route playback does not match the stepwise execution terminal")
    timeline = [
        {
            "kind": "layer_reached",
            "source_sequence": item["sequence"],
            "map_id": item["map_id"],
        }
        for item in review["navigation"]
        if item["sequence"] <= last_sequence
    ]
    timeline.extend(
        {
            "kind": "skill_dispatched",
            "source_sequence": item["source_sequence"],
            "map_id": item["map_id"],
            "key": item["key"],
            "skill": item["skill"],
        }
        for item in review["casts"]
        if item["source_sequence"] <= last_sequence
    )
    if status == "taken_over":
        timeline.append(
            {
                "kind": "takeover",
                "source_sequence": last_sequence,
                "reason": reason,
            }
        )
    timeline.sort(key=lambda item: item["source_sequence"])

    replay_name = "route-playback.gif" if status == "completed" else "route-takeover.gif"
    first, *rest = [image for _, image in displayed]
    first.save(
        output / replay_name,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=80,
        loop=0,
        optimize=False,
    )
    updated = dict(review)
    updated["playback"] = {
        "status": status,
        "reason": reason,
        "last_source_sequence": last_sequence,
        "displayed_frames": len(displayed),
        "replay": replay_name,
        "replay_sha256": _sha256(output / replay_name),
    }
    updated["timeline"] = timeline
    (output / "player-review.json").write_text(
        json.dumps(updated, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def verify_route_playback(output: Path) -> dict[str, Any]:
    """Reopen a player route index and its GIF without trusting the live window."""
    value = json.loads((output / "player-review.json").read_text(encoding="utf-8"))
    build_route_playback(output, value)
    playback = value.get("playback")
    timeline = value.get("timeline")
    if (
        not isinstance(playback, dict)
        or playback.get("status") not in {"completed", "taken_over"}
        or type(playback.get("last_source_sequence")) is not int
        or type(playback.get("displayed_frames")) is not int
        or not isinstance(playback.get("replay"), str)
        or not isinstance(playback.get("replay_sha256"), str)
        or not isinstance(timeline, list)
        or not timeline
    ):
        raise ValueError("invalid route playback review")
    replay_path = output / playback["replay"]
    if replay_path.parent != output or _sha256(replay_path) != playback["replay_sha256"]:
        raise ValueError("route playback GIF changed")
    with Image.open(replay_path) as replay:
        if getattr(replay, "n_frames", 1) != playback["displayed_frames"]:
            raise ValueError("route playback frame count changed")
    sequences: list[int] = []
    for item in timeline:
        if not isinstance(item, dict) or type(item.get("source_sequence")) is not int:
            raise ValueError("invalid route playback timeline")
        sequences.append(item["source_sequence"])
    if sequences != sorted(sequences) or any(
        sequence > playback["last_source_sequence"] for sequence in sequences
    ):
        raise ValueError("invalid route playback timeline")
    if playback["status"] == "taken_over" and (
        timeline[-1].get("kind") != "takeover"
        or timeline[-1].get("reason") != playback.get("reason")
        or timeline[-1].get("source_sequence") != playback["last_source_sequence"]
    ):
        raise ValueError("taken-over route is missing its terminal event")
    execution_result = None
    if "execution" in value:
        if value.get("scope") == "project_owned_live_synthetic_route":
            from .owned_live_route import verify_live_route_execution

            execution_result = verify_live_route_execution(output, value)
        else:
            from .owned_route_execution import verify_route_execution

            execution_result = verify_route_execution(output, value)
        if (
            execution_result["status"] != playback["status"]
            or execution_result["last_source_sequence"] != playback["last_source_sequence"]
        ):
            raise ValueError("route playback and stepwise execution terminals disagree")
    result = {
        "verified": True,
        "scope": value["scope"],
        "status": playback["status"],
        "last_source_sequence": playback["last_source_sequence"],
        "timeline_events": len(timeline),
        "displayed_frames": playback["displayed_frames"],
        "replay_sha256": playback["replay_sha256"],
    }
    if execution_result is not None:
        result["execution"] = execution_result
    return result
