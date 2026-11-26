from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from maple_vision.desktop_window import WindowIdentity
from maple_vision.desktop_window_check import _select, run, verify_desktop_window_capture


def identity(handle: int, title: str) -> WindowIdentity:
    return WindowIdentity(
        handle=handle,
        process_id=handle + 100,
        thread_id=handle + 200,
        process_started_100ns=handle + 300,
        class_name="OwnedWindow",
        title=title,
        client_bounds=(0, 0, 320, 192),
        dpi=96,
    )


def test_exact_title_or_handle_never_silently_picks_an_ambiguous_window() -> None:
    windows = (identity(1, "target"), identity(2, "target"), identity(3, "other"))
    with pytest.raises(ValueError, match="matched 2"):
        _select(windows, title="target", handle=None)
    assert _select(windows, title=None, handle=3).title == "other"
    with pytest.raises(ValueError, match="matched 0"):
        _select(windows, title="missing", handle=None)


def test_invalid_delay_is_rejected_before_native_backend(monkeypatch, tmp_path: Path) -> None:
    class UnexpectedBackend:
        def __init__(self) -> None:
            raise AssertionError("invalid arguments must fail before native access")

    monkeypatch.setattr("maple_vision.desktop_window_check.NativeWindowsBackend", UnexpectedBackend)
    with pytest.raises(ValueError, match="between 0 and 10"):
        run(["--handle", "1", "--output", str(tmp_path / "out"), "--delay-seconds", "11"])


def test_capture_command_writes_one_reopenable_read_only_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    target = identity(7, "target")

    class FakeBackend:
        def list_windows(self) -> tuple[WindowIdentity, ...]:
            return (target,)

        def describe(self, handle: int) -> WindowIdentity | None:
            return target if handle == target.handle else None

        def foreground_handle(self) -> int | None:
            return target.handle

        def capture(self, selected: WindowIdentity) -> Image.Image:
            assert selected == target
            return Image.new("RGB", (320, 192), "#123456")

    monkeypatch.setattr("maple_vision.desktop_window_check.NativeWindowsBackend", FakeBackend)
    monkeypatch.setattr("maple_vision.desktop_window_check.time.sleep", lambda _: None)
    output = tmp_path / "capture"
    report = run(["--title-exact", "target", "--output", str(output), "--delay-seconds", "0"])
    assert report["dispatch_gate"]["authorized"] is True
    verified = verify_desktop_window_capture(output)
    assert verified["verified"] is True
    assert verified["width"] == 320
    Image.new("RGB", (320, 192), "#654321").save(output / "capture.png")
    with pytest.raises(ValueError, match="file changed"):
        verify_desktop_window_capture(output)
