"""Read-only Windows capture session with explicit focus and takeover gates."""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from PIL import Image, ImageGrab


@dataclass(frozen=True)
class WindowIdentity:
    """Fields that must stay stable while a desktop session is armed."""

    handle: int
    process_id: int
    thread_id: int
    process_started_100ns: int
    class_name: str
    title: str
    client_bounds: tuple[int, int, int, int]
    dpi: int

    @property
    def stable_token(self) -> tuple[object, ...]:
        return (
            self.handle,
            self.process_id,
            self.thread_id,
            self.process_started_100ns,
            self.class_name,
            self.title,
            self.client_bounds,
            self.dpi,
        )

    def manifest(self) -> dict[str, object]:
        return asdict(self)


class DesktopWindowBackend(Protocol):
    """Minimal read-only surface required by the guarded session."""

    def list_windows(self) -> tuple[WindowIdentity, ...]: ...

    def describe(self, handle: int) -> WindowIdentity | None: ...

    def foreground_handle(self) -> int | None: ...

    def capture(self, identity: WindowIdentity) -> Image.Image: ...


@dataclass(frozen=True)
class DesktopFrame:
    sequence: int
    captured_monotonic_ns: int
    rgb_sha256: str
    identity: WindowIdentity
    image: Image.Image

    def manifest(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "captured_monotonic_ns": self.captured_monotonic_ns,
            "rgb_sha256": self.rgb_sha256,
            "width": self.image.width,
            "height": self.image.height,
            "identity": self.identity.manifest(),
        }


class HeldKeyLedger:
    """Remember only keys owned by this session and release every one on pause."""

    def __init__(self, release: Callable[[str], None]) -> None:
        self._release = release
        self._held: set[str] = set()

    @property
    def held(self) -> tuple[str, ...]:
        return tuple(sorted(self._held))

    def pressed(self, key: str) -> None:
        if not key or key != key.upper():
            raise ValueError("tracked desktop keys must use a non-empty canonical name")
        self._held.add(key)

    def released(self, key: str) -> None:
        self._held.discard(key)

    def release_all(self) -> tuple[str, ...]:
        released = self.held
        for key in released:
            self._release(key)
        self._held.clear()
        return released


class DesktopWindowSession:
    """Bind one confirmed window and stop authorization immediately on focus loss."""

    def __init__(
        self,
        backend: DesktopWindowBackend,
        *,
        release_key: Callable[[str], None],
        clock: Callable[[], int] = time.monotonic_ns,
        max_frame_age_ns: int = 500_000_000,
    ) -> None:
        if max_frame_age_ns <= 0:
            raise ValueError("desktop frame age must be positive")
        self.backend = backend
        self.keys = HeldKeyLedger(release_key)
        self.clock = clock
        self.max_frame_age_ns = max_frame_age_ns
        self.state = "idle"
        self.reason: str | None = None
        self.identity: WindowIdentity | None = None
        self.latest_frame: DesktopFrame | None = None
        self.sequence = 0
        self.events: list[dict[str, object]] = []

    def select(self, handle: int) -> WindowIdentity:
        if self.state == "closed":
            raise ValueError("desktop window session is closed")
        identity = self.backend.describe(handle)
        if identity is None:
            raise ValueError("selected desktop window is unavailable")
        released = self.keys.release_all()
        self.identity = identity
        self.latest_frame = None
        self.state = "awaiting_confirmation"
        self.reason = None
        self.events.append(
            {
                "kind": "selected",
                "identity": identity.manifest(),
                "released_keys": list(released),
            }
        )
        return identity

    def confirm(self) -> dict[str, object]:
        if self.state != "awaiting_confirmation" or self.identity is None:
            raise ValueError("select a desktop window before confirming it")
        current = self.backend.describe(self.identity.handle)
        if current is None or current.stable_token != self.identity.stable_token:
            self._pause("window_identity_changed")
            raise ValueError("selected desktop window identity changed before confirmation")
        if self.backend.foreground_handle() != self.identity.handle:
            self._pause("focus_lost")
            raise ValueError("selected desktop window must be focused for confirmation")
        self.state = "ready"
        receipt: dict[str, object] = {
            "scope": "project_owned_external_window_capture",
            "confirmed_monotonic_ns": self.clock(),
            "identity": self.identity.manifest(),
            "input_mode": "read_only_capture_gate",
        }
        self.events.append({"kind": "confirmed", **receipt})
        return receipt

    def observe(self) -> DesktopFrame:
        identity = self._require_ready_identity()
        self._validate_live_window(identity)
        image = self.backend.capture(identity).convert("RGB")
        self._validate_live_window(identity)
        captured = self.clock()
        frame = DesktopFrame(
            sequence=self.sequence,
            captured_monotonic_ns=captured,
            rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            identity=identity,
            image=image,
        )
        self.sequence += 1
        self.latest_frame = frame
        self.events.append({"kind": "captured", **frame.manifest()})
        return frame

    def authorize_dispatch(self, frame: DesktopFrame) -> dict[str, object]:
        identity = self._require_ready_identity()
        self._validate_live_window(identity)
        now = self.clock()
        if self.latest_frame is not frame:
            self._pause("frame_not_latest")
            raise ValueError("dispatch requires the latest captured frame")
        age = now - frame.captured_monotonic_ns
        if age < 0 or age > self.max_frame_age_ns:
            self._pause("frame_stale")
            raise ValueError("dispatch frame is stale")
        receipt = {
            "authorized": True,
            "frame_sequence": frame.sequence,
            "frame_rgb_sha256": frame.rgb_sha256,
            "checked_monotonic_ns": now,
            "identity": identity.manifest(),
        }
        self.events.append({"kind": "dispatch_authorized", **receipt})
        return receipt

    def record_key_down(self, key: str, frame: DesktopFrame) -> None:
        self.authorize_dispatch(frame)
        self.keys.pressed(key)
        self.events.append({"kind": "key_down_recorded", "key": key})

    def record_key_up(self, key: str) -> None:
        self.keys.released(key)
        self.events.append({"kind": "key_up_recorded", "key": key})

    def takeover(self, reason: str = "manual_takeover") -> tuple[str, ...]:
        return self._pause(reason)

    def resume(self) -> dict[str, object]:
        if self.state != "paused" or self.identity is None:
            raise ValueError("only a paused desktop session can resume")
        current = self.backend.describe(self.identity.handle)
        if current is None or current.stable_token != self.identity.stable_token:
            raise ValueError("window identity changed; select and confirm it again")
        if self.backend.foreground_handle() != self.identity.handle:
            raise ValueError("focus the selected desktop window before resuming")
        previous = self.reason
        self.state = "ready"
        self.reason = None
        receipt: dict[str, object] = {
            "resumed": True,
            "previous_reason": previous,
            "confirmed_monotonic_ns": self.clock(),
            "identity": self.identity.manifest(),
        }
        self.events.append({"kind": "resumed", **receipt})
        return receipt

    def close(self) -> None:
        if self.state != "closed":
            released = self.keys.release_all()
            self.events.append({"kind": "closed", "released_keys": list(released)})
            self.state = "closed"
            self.reason = "closed"

    def _require_ready_identity(self) -> WindowIdentity:
        if self.state != "ready" or self.identity is None:
            raise ValueError(f"desktop window session is not ready: {self.state}")
        return self.identity

    def _validate_live_window(self, identity: WindowIdentity) -> None:
        current = self.backend.describe(identity.handle)
        if current is None or current.stable_token != identity.stable_token:
            self._pause("window_identity_changed")
            raise ValueError("desktop window identity changed")
        if self.backend.foreground_handle() != identity.handle:
            self._pause("focus_lost")
            raise ValueError("desktop window lost focus")

    def _pause(self, reason: str) -> tuple[str, ...]:
        released = self.keys.release_all()
        if self.state != "closed":
            self.state = "paused"
            self.reason = reason
            self.latest_frame = None
            self.events.append(
                {"kind": "paused", "reason": reason, "released_keys": list(released)}
            )
        return released


class NativeWindowsBackend:
    """Enumerate and capture visible Win32 client areas without injecting input."""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("native desktop capture is available only on Windows")
        import ctypes
        from ctypes import wintypes

        self.ctypes: Any = ctypes
        self.wintypes: Any = wintypes
        self.user32: Any = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsWindowVisible.restype = wintypes.BOOL
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.ClientToScreen.restype = wintypes.BOOL
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetClassNameW.restype = ctypes.c_int
        get_dpi = getattr(self.user32, "GetDpiForWindow", None)
        if get_dpi is not None:
            get_dpi.argtypes = [wintypes.HWND]
            get_dpi.restype = wintypes.UINT
        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        self.kernel32.GetProcessTimes.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL

    def list_windows(self) -> tuple[WindowIdentity, ...]:
        windows: list[WindowIdentity] = []
        callback_type = self.ctypes.WINFUNCTYPE(
            self.wintypes.BOOL, self.wintypes.HWND, self.wintypes.LPARAM
        )

        def collect(handle: int, _: int) -> bool:
            identity = self.describe(int(handle))
            if identity is not None and identity.title:
                windows.append(identity)
            return True

        callback = callback_type(collect)
        if not self.user32.EnumWindows(callback, 0):
            raise OSError("failed to enumerate Windows desktop windows")
        return tuple(sorted(windows, key=lambda item: (item.title.casefold(), item.handle)))

    def describe(self, handle: int) -> WindowIdentity | None:
        if not self.user32.IsWindow(handle) or not self.user32.IsWindowVisible(handle):
            return None
        rect = self.wintypes.RECT()
        if not self.user32.GetClientRect(handle, self.ctypes.byref(rect)):
            return None
        origin = self.wintypes.POINT(0, 0)
        if not self.user32.ClientToScreen(handle, self.ctypes.byref(origin)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        process_id = self.wintypes.DWORD()
        thread_id = int(self.user32.GetWindowThreadProcessId(handle, self.ctypes.byref(process_id)))
        if thread_id <= 0 or process_id.value <= 0:
            return None
        title = self._window_text(handle)
        class_name = self._class_name(handle)
        get_dpi = getattr(self.user32, "GetDpiForWindow", None)
        dpi = int(get_dpi(handle)) if get_dpi is not None else 96
        return WindowIdentity(
            handle=handle,
            process_id=int(process_id.value),
            thread_id=thread_id,
            process_started_100ns=self._process_started(int(process_id.value)),
            class_name=class_name,
            title=title,
            client_bounds=(origin.x, origin.y, origin.x + width, origin.y + height),
            dpi=dpi or 96,
        )

    def foreground_handle(self) -> int | None:
        handle = int(self.user32.GetForegroundWindow())
        return handle or None

    def capture(self, identity: WindowIdentity) -> Image.Image:
        current = self.describe(identity.handle)
        if current is None or current.stable_token != identity.stable_token:
            raise ValueError("desktop window changed before capture")
        return ImageGrab.grab(
            bbox=identity.client_bounds,
            include_layered_windows=True,
            all_screens=True,
        ).convert("RGB")

    def _window_text(self, handle: int) -> str:
        length = int(self.user32.GetWindowTextLengthW(handle))
        buffer = self.ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(handle, buffer, len(buffer))
        return str(buffer.value)

    def _class_name(self, handle: int) -> str:
        buffer = self.ctypes.create_unicode_buffer(256)
        if not self.user32.GetClassNameW(handle, buffer, len(buffer)):
            return ""
        return str(buffer.value)

    def _process_started(self, process_id: int) -> int:
        handle = self.kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
        )
        if not handle:
            return 0
        created = self.wintypes.FILETIME()
        exited = self.wintypes.FILETIME()
        kernel = self.wintypes.FILETIME()
        user = self.wintypes.FILETIME()
        try:
            if not self.kernel32.GetProcessTimes(
                handle,
                self.ctypes.byref(created),
                self.ctypes.byref(exited),
                self.ctypes.byref(kernel),
                self.ctypes.byref(user),
            ):
                return 0
            return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        finally:
            self.kernel32.CloseHandle(handle)
