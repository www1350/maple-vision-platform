"""Deterministic opaque test frames, not a video decoder or a game simulator."""

from .contracts import Frame, nonnegative_int


class DemoSource:
    def __init__(self, frames: int = 3, interval_ms: int = 50) -> None:
        nonnegative_int(frames, "frames")
        nonnegative_int(interval_ms, "interval_ms")
        if interval_ms == 0:
            raise ValueError("interval_ms must be positive")
        self.frames = frames
        self.interval_ms = interval_ms
        self._sequence = 0

    def read(self) -> Frame | None:
        if self._sequence >= self.frames:
            return None
        frame = Frame(self._sequence, self._sequence * self.interval_ms, b"\x00\x00\x00")
        self._sequence += 1
        return frame
