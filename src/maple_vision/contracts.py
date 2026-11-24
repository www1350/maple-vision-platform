from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Status(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONFLICTED = "conflicted"


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp_ms: int
    pixels: bytes


@dataclass(frozen=True)
class Observation:
    sequence: int
    timestamp_ms: int
    status: Status


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...


class Perception(Protocol):
    def observe(self, frame: Frame) -> Observation: ...


class Simulator(Protocol):
    def reset(self, seed: int) -> None: ...
    def step(self) -> Frame: ...
