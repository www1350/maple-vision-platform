"""Player-invoked read-only check for one visible Windows target window."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .desktop_window import DesktopWindowSession, NativeWindowsBackend, WindowIdentity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or capture one visible Windows window without sending input."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--list", action="store_true", help="list visible candidate windows")
    target.add_argument("--title-exact", help="select the only window with this exact title")
    target.add_argument("--handle", type=lambda value: int(value, 0), help="select a Win32 HWND")
    parser.add_argument("--output", type=Path, help="new directory for capture evidence")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        help="time to focus the target before confirmation (0..10, default 3)",
    )
    return parser


def _select(
    windows: tuple[WindowIdentity, ...], *, title: str | None, handle: int | None
) -> WindowIdentity:
    matches = [
        item
        for item in windows
        if (title is not None and item.title == title)
        or (handle is not None and item.handle == handle)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"window selector matched {len(matches)} visible windows; expected exactly 1"
        )
    return matches[0]


def run(arguments: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(arguments)
    if not 0 <= args.delay_seconds <= 10:
        raise ValueError("--delay-seconds must be between 0 and 10")
    backend = NativeWindowsBackend()
    windows = backend.list_windows()
    if args.list:
        return {
            "scope": "visible_windows_read_only_inventory",
            "windows": [item.manifest() for item in windows],
        }
    if args.output is None:
        raise ValueError("--output is required when capturing a window")
    if args.output.exists():
        raise FileExistsError(f"capture output already exists: {args.output}")
    selected = _select(windows, title=args.title_exact, handle=args.handle)
    time.sleep(args.delay_seconds)
    session = DesktopWindowSession(backend, release_key=lambda _: None)
    session.select(selected.handle)
    receipt = session.confirm()
    frame = session.observe()
    authorization = session.authorize_dispatch(frame)
    args.output.mkdir(parents=True)
    capture = args.output / "capture.png"
    frame.image.save(capture)
    session.close()
    report = {
        "schema_version": 1,
        "scope": "project_owned_external_window_capture_check",
        "receipt": receipt,
        "frame": frame.manifest(),
        "capture_file_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "dispatch_gate": authorization,
        "events": session.events,
        "limitations": [
            "captures visible screen pixels only",
            "does not send input or establish official-client permission",
            "one successful frame does not establish game perception accuracy",
        ],
    }
    (args.output / "evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def verify_desktop_window_capture(output: Path) -> dict[str, object]:
    """Reopen the one-frame bundle and reject media substitution or extra files."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError("desktop capture evidence root must be a real directory")
    files = {path.name for path in output.iterdir() if path.is_file() and not path.is_symlink()}
    if files != {"capture.png", "evidence.json"} or any(
        path.is_symlink() for path in output.iterdir()
    ):
        raise ValueError("desktop capture evidence contains missing or undeclared files")
    evidence = output / "evidence.json"
    if evidence.stat().st_size > 128 * 1024:
        raise ValueError("desktop capture evidence is too large")
    try:
        report = json.loads(evidence.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid desktop capture evidence JSON") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("scope") != "project_owned_external_window_capture_check"
    ):
        raise ValueError("unsupported desktop capture evidence")
    frame = report.get("frame")
    if not isinstance(frame, dict):
        raise ValueError("desktop capture frame manifest is missing")
    capture = output / "capture.png"
    if hashlib.sha256(capture.read_bytes()).hexdigest() != report.get("capture_file_sha256"):
        raise ValueError("desktop capture PNG file changed")
    with Image.open(capture) as source:
        source.load()
        image = source.convert("RGB")
    if (
        image.width != frame.get("width")
        or image.height != frame.get("height")
        or hashlib.sha256(image.tobytes()).hexdigest() != frame.get("rgb_sha256")
    ):
        raise ValueError("desktop capture decoded pixels changed")
    return {
        "verified": True,
        "scope": report["scope"],
        "frame_sequence": frame.get("sequence"),
        "rgb_sha256": frame.get("rgb_sha256"),
        "width": image.width,
        "height": image.height,
    }


def main() -> int:
    try:
        result = run()
    except (FileExistsError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
