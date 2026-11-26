from __future__ import annotations

from pathlib import Path

from PIL import Image

from maple_vision.desktop_observation_check import run
from maple_vision.desktop_window import WindowIdentity


def test_observation_command_requires_explicit_fixture_mode_and_verifies(
    monkeypatch, tmp_path: Path
) -> None:
    identity = WindowIdentity(9, 10, 11, 12, "Owned", "fixture", (0, 0, 320, 192), 96)

    class Backend:
        def list_windows(self) -> tuple[WindowIdentity, ...]:
            return (identity,)

        def describe(self, handle: int) -> WindowIdentity | None:
            return identity if handle == identity.handle else None

        def foreground_handle(self) -> int | None:
            return identity.handle

        def capture(self, selected: WindowIdentity) -> Image.Image:
            assert selected == identity
            return Image.new("RGB", (320, 192), "#123456")

    monkeypatch.setattr("maple_vision.desktop_observation_check.NativeWindowsBackend", Backend)
    monkeypatch.setattr("maple_vision.desktop_observation_check.time.sleep", lambda _: None)
    output = tmp_path / "result"
    report = run(
        [
            "--title-exact",
            "fixture",
            "--output",
            str(output),
            "--perception",
            "synthetic-fixture",
            "--delay-seconds",
            "0",
        ]
    )
    assert report["verification"]["verified"] is True
    assert report["verification"]["status"] == "unknown"
    assert (output / "recognized-overlay.png").is_file()
