from __future__ import annotations

import sys
from dataclasses import replace

import pytest
from PIL import Image

from maple_vision.desktop_window import (
    DesktopWindowSession,
    NativeWindowsBackend,
    WindowIdentity,
)

IDENTITY = WindowIdentity(
    handle=41,
    process_id=9001,
    thread_id=19,
    process_started_100ns=123456,
    class_name="OwnedTargetWindow",
    title="Maple Vision Capture Target",
    client_bounds=(10, 20, 330, 212),
    dpi=96,
)


class FakeDesktop:
    def __init__(self) -> None:
        self.identity: WindowIdentity | None = IDENTITY
        self.foreground: int | None = IDENTITY.handle
        self.image = Image.new("RGB", (320, 192), (17, 34, 51))
        self.captures = 0

    def list_windows(self) -> tuple[WindowIdentity, ...]:
        return () if self.identity is None else (self.identity,)

    def describe(self, handle: int) -> WindowIdentity | None:
        if self.identity is None or self.identity.handle != handle:
            return None
        return self.identity

    def foreground_handle(self) -> int | None:
        return self.foreground

    def capture(self, identity: WindowIdentity) -> Image.Image:
        assert identity == self.identity
        self.captures += 1
        return self.image.copy()


def test_confirmed_external_window_frame_can_gate_one_recorded_key() -> None:
    desktop = FakeDesktop()
    released: list[str] = []
    now = 1_000_000_000
    session = DesktopWindowSession(
        desktop,
        release_key=released.append,
        clock=lambda: now,
    )
    assert session.select(41) == IDENTITY
    receipt = session.confirm()
    assert receipt["scope"] == "project_owned_external_window_capture"
    assert receipt["input_mode"] == "read_only_capture_gate"
    frame = session.observe()
    assert frame.sequence == 0
    assert frame.image.size == (320, 192)
    assert len(frame.rgb_sha256) == 64
    authorized = session.authorize_dispatch(frame)
    assert authorized["frame_rgb_sha256"] == frame.rgb_sha256
    session.record_key_down("J", frame)
    assert session.keys.held == ("J",)
    session.record_key_up("J")
    assert session.keys.held == ()
    assert released == []
    assert desktop.captures == 1


def test_focus_loss_releases_owned_keys_and_requires_explicit_resume() -> None:
    desktop = FakeDesktop()
    released: list[str] = []
    session = DesktopWindowSession(desktop, release_key=released.append)
    session.select(41)
    session.confirm()
    frame = session.observe()
    session.record_key_down("UP", frame)
    desktop.foreground = 99
    with pytest.raises(ValueError, match="lost focus"):
        session.observe()
    assert session.state == "paused"
    assert session.reason == "focus_lost"
    assert session.keys.held == ()
    assert released == ["UP"]
    with pytest.raises(ValueError, match="focus"):
        session.resume()
    desktop.foreground = 41
    resumed = session.resume()
    assert resumed["previous_reason"] == "focus_lost"
    assert session.state == "ready"
    assert session.observe().sequence == 1


def test_identity_change_needs_selection_and_confirmation_again() -> None:
    desktop = FakeDesktop()
    released: list[str] = []
    session = DesktopWindowSession(desktop, release_key=released.append)
    session.select(41)
    session.confirm()
    frame = session.observe()
    session.record_key_down("LEFT", frame)
    desktop.identity = replace(IDENTITY, title="Replacement Target")
    with pytest.raises(ValueError, match="identity changed"):
        session.authorize_dispatch(frame)
    assert session.state == "paused"
    assert released == ["LEFT"]
    with pytest.raises(ValueError, match="select and confirm"):
        session.resume()
    selected = session.select(41)
    assert selected.title == "Replacement Target"
    session.confirm()
    assert session.observe().identity.title == "Replacement Target"


def test_stale_or_nonlatest_frames_pause_before_dispatch() -> None:
    desktop = FakeDesktop()
    now = 100
    session = DesktopWindowSession(
        desktop,
        release_key=lambda _: None,
        clock=lambda: now,
        max_frame_age_ns=10,
    )
    session.select(41)
    session.confirm()
    first = session.observe()
    second = session.observe()
    with pytest.raises(ValueError, match="latest"):
        session.authorize_dispatch(first)
    assert session.reason == "frame_not_latest"
    session.resume()
    second = session.observe()
    now = 111
    with pytest.raises(ValueError, match="stale"):
        session.authorize_dispatch(second)
    assert session.reason == "frame_stale"


def test_capture_is_revalidated_after_backend_returns() -> None:
    class MutatingDesktop(FakeDesktop):
        def capture(self, identity: WindowIdentity) -> Image.Image:
            image = super().capture(identity)
            self.foreground = 99
            return image

    desktop = MutatingDesktop()
    session = DesktopWindowSession(desktop, release_key=lambda _: None)
    session.select(41)
    session.confirm()
    with pytest.raises(ValueError, match="lost focus"):
        session.observe()
    assert session.latest_frame is None
    assert session.state == "paused"


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows rejection only")
def test_native_backend_is_explicitly_windows_only() -> None:
    with pytest.raises(OSError, match="only on Windows"):
        NativeWindowsBackend()
