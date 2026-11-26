"""Player-invoked read-only observation of one confirmed Windows window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .desktop_observation import (
    fixture_desktop_observation_session,
    onnx_desktop_observation_session,
    verify_desktop_observation_evidence,
    write_desktop_observation_evidence,
)
from .desktop_window import DesktopWindowSession, NativeWindowsBackend
from .desktop_window_check import _select
from .onnx_detector import OnnxDetector


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one confirmed visible Windows window, run an explicit fixture or "
            "hash-bound ONNX read-only perception mode, and save a player-visible overlay."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--title-exact", help="select the only window with this exact title")
    target.add_argument("--handle", type=lambda value: int(value, 0), help="select a Win32 HWND")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument(
        "--perception",
        choices=("synthetic-fixture", "onnx-detector"),
        required=True,
        help="explicit recognizer scope; ONNX mode emits boxes but leaves HUD/map unknown",
    )
    parser.add_argument("--model", type=Path, help="local ONNX model for onnx-detector mode")
    parser.add_argument("--manifest", type=Path, help="hash-binding model manifest")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        help="time to focus the target before confirmation (0..10, default 3)",
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(arguments)
    if not 0 <= args.delay_seconds <= 10:
        raise ValueError("--delay-seconds must be between 0 and 10")
    if args.output.exists():
        raise FileExistsError(f"observation output already exists: {args.output}")
    supplied_model = args.model is not None or args.manifest is not None
    if args.perception == "synthetic-fixture" and supplied_model:
        raise ValueError("fixture perception does not accept --model or --manifest")
    if args.perception == "onnx-detector" and (args.model is None or args.manifest is None):
        raise ValueError("onnx-detector requires --model and --manifest together")
    backend = NativeWindowsBackend()
    selected = _select(backend.list_windows(), title=args.title_exact, handle=args.handle)
    time.sleep(args.delay_seconds)
    desktop = DesktopWindowSession(backend, release_key=lambda _: None)
    desktop.select(selected.handle)
    receipt = desktop.confirm()
    try:
        if args.perception == "synthetic-fixture":
            observation_session = fixture_desktop_observation_session(desktop)
        else:
            assert args.model is not None and args.manifest is not None
            detector = OnnxDetector(
                args.model,
                args.manifest,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
            )
            observation_session = onnx_desktop_observation_session(desktop, detector)
        result = observation_session.observe()
    finally:
        desktop.close()
    report = write_desktop_observation_evidence(
        args.output, result, receipt=receipt, events=desktop.events
    )
    return {**report, "verification": verify_desktop_observation_evidence(args.output)}


def main() -> int:
    try:
        report = run()
    except (FileExistsError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "verification": report["verification"],
                "limitations": report["limitations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
