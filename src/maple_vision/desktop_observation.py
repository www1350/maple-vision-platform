"""Bind visible cross-window pixels to one read-only perception/state report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from .desktop_window import DesktopFrame, DesktopWindowSession
from .scene import ImageFrame
from .state_estimation import BeliefState, TemporalStateEstimator
from .vision import PixelPerception, VisualObservation


class FramePerception(Protocol):
    """Pixel-only perception surface accepted by the desktop bridge."""

    def observe(self, frame: ImageFrame) -> VisualObservation: ...


@dataclass(frozen=True)
class DesktopObservation:
    """One captured frame and every result derived from those exact pixels."""

    frame: DesktopFrame
    observation: VisualObservation
    belief: BeliefState
    perception_mode: str

    def manifest(self) -> dict[str, object]:
        return {
            "scope": "project_owned_external_window_observation",
            "perception_mode": self.perception_mode,
            "frame": self.frame.manifest(),
            "observation": self.belief.to_dict()["observation"],
            "belief": self.belief.to_dict(),
        }


class DesktopObservationSession:
    """Run read-only perception after the guarded desktop session captures a frame."""

    def __init__(
        self,
        desktop: DesktopWindowSession,
        perception: FramePerception,
        *,
        estimator: TemporalStateEstimator | None = None,
        perception_mode: str,
        expected_size: tuple[int, int] | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        if (
            not perception_mode
            or not perception_mode.strip()
            or len(perception_mode) > 200
            or any(ord(char) < 32 or ord(char) == 127 for char in perception_mode)
        ):
            raise ValueError("perception mode must be bounded non-empty text")
        if expected_size is not None and (
            len(expected_size) != 2
            or any(type(value) is not int or value <= 0 for value in expected_size)
        ):
            raise ValueError("expected desktop observation size must contain positive integers")
        self.desktop = desktop
        self.perception = perception
        self.estimator = estimator or TemporalStateEstimator()
        self.perception_mode = perception_mode
        self.expected_size = expected_size
        self.now_ms = now_ms or (lambda: self.desktop.clock() // 1_000_000)

    def observe(self) -> DesktopObservation:
        desktop_frame = self.desktop.observe()
        size = desktop_frame.image.size
        if self.expected_size is not None and size != self.expected_size:
            self.desktop.takeover("capture_size_changed")
            raise ValueError(
                f"captured client size {size[0]}x{size[1]} does not match "
                f"required {self.expected_size[0]}x{self.expected_size[1]}"
            )
        timestamp_ms = desktop_frame.captured_monotonic_ns // 1_000_000
        image_frame = ImageFrame(
            desktop_frame.sequence,
            timestamp_ms,
            desktop_frame.image.width,
            desktop_frame.image.height,
            desktop_frame.image.tobytes(),
        )
        raw = self.perception.observe(image_frame)
        observation = replace(raw, source_id=f"external-window:{self.perception_mode}")
        belief = self.estimator.update(observation, self.now_ms())
        result = DesktopObservation(desktop_frame, observation, belief, self.perception_mode)
        self.desktop.events.append(
            {
                "kind": "observation_derived",
                "frame_sequence": desktop_frame.sequence,
                "frame_rgb_sha256": desktop_frame.rgb_sha256,
                "perception_mode": self.perception_mode,
                "status": observation.status.value,
                "actionable": belief.actionable,
                "reason": belief.reason,
            }
        )
        return result


def fixture_desktop_observation_session(
    desktop: DesktopWindowSession,
) -> DesktopObservationSession:
    """Explicit project-fixture mode; never represents game-domain accuracy."""
    return DesktopObservationSession(
        desktop,
        PixelPerception(),
        perception_mode="synthetic_fixture_palette_v2",
        expected_size=(320, 192),
    )


def _overlay(result: DesktopObservation) -> Image.Image:
    image = result.frame.image.copy()
    draw = ImageDraw.Draw(image)
    for detection in result.observation.detections:
        draw.rectangle(detection.bbox, outline="#ffffff", width=2)
        x1, y1, _, _ = detection.bbox
        draw.text((x1 + 2, max(0, y1 - 12)), detection.class_id, fill="#ffffff")
    text = (
        f"{result.observation.status.value} | {result.belief.reason} | "
        f"frame {result.frame.sequence}"
    )
    draw.rectangle((0, image.height - 18, image.width, image.height), fill="#111724")
    draw.text((4, image.height - 16), text, fill="#ffffff")
    return image


def write_desktop_observation_evidence(
    output: Path,
    result: DesktopObservation,
    *,
    receipt: dict[str, object],
    events: list[dict[str, object]],
) -> dict[str, object]:
    """Write a new, hash-bound player-visible read-only observation bundle."""
    if output.exists():
        raise FileExistsError(f"observation output already exists: {output}")
    output.mkdir(parents=True)
    capture = output / "capture.png"
    overlay = output / "recognized-overlay.png"
    result.frame.image.save(capture)
    _overlay(result).save(overlay)
    report = {
        "schema_version": 1,
        **result.manifest(),
        "receipt": receipt,
        "files": {
            capture.name: hashlib.sha256(capture.read_bytes()).hexdigest(),
            overlay.name: hashlib.sha256(overlay.read_bytes()).hexdigest(),
        },
        "events": events,
        "limitations": [
            "visible pixels from a player-confirmed external window only",
            "fixture palette mode validates plumbing, not official-client perception accuracy",
            "no input is sent and no official-client permission is established",
        ],
    }
    (output / "evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def verify_desktop_observation_evidence(output: Path) -> dict[str, object]:
    """Reopen a read-only observation bundle and reject media substitution."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError("desktop observation root must be a real directory")
    entries = list(output.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("desktop observation evidence cannot contain symlinks")
    names = {path.name for path in entries if path.is_file()}
    if names != {"capture.png", "recognized-overlay.png", "evidence.json"}:
        raise ValueError("desktop observation evidence contains missing or undeclared files")
    evidence = output / "evidence.json"
    if evidence.stat().st_size > 512 * 1024:
        raise ValueError("desktop observation evidence is too large")
    try:
        report = json.loads(evidence.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid desktop observation JSON") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("scope") != "project_owned_external_window_observation"
        or not isinstance(report.get("files"), dict)
        or not isinstance(report.get("frame"), dict)
        or not isinstance(report.get("observation"), dict)
        or not isinstance(report.get("belief"), dict)
    ):
        raise ValueError("unsupported desktop observation evidence")
    files = report["files"]
    for name in ("capture.png", "recognized-overlay.png"):
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != files.get(name):
            raise ValueError(f"desktop observation {name} changed")
    frame = report["frame"]
    with Image.open(output / "capture.png") as source:
        source.load()
        capture = source.convert("RGB")
    if capture.size != (frame.get("width"), frame.get("height")) or hashlib.sha256(
        capture.tobytes()
    ).hexdigest() != frame.get("rgb_sha256"):
        raise ValueError("desktop observation decoded capture pixels changed")
    observation = report["observation"]
    belief = report["belief"]
    if (
        observation.get("sequence") != frame.get("sequence")
        or belief.get("observation") != observation
    ):
        raise ValueError("desktop observation results are not bound to the captured frame")
    return {
        "verified": True,
        "scope": report["scope"],
        "perception_mode": report.get("perception_mode"),
        "frame_sequence": frame.get("sequence"),
        "rgb_sha256": frame.get("rgb_sha256"),
        "status": observation.get("status"),
        "actionable": belief.get("actionable"),
    }
