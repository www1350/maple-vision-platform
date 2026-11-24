"""P0 TOML configuration: reject unknown keys, unknown modes and invalid limits."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .contracts import SCHEMA_VERSION


@dataclass(frozen=True)
class RuntimeConfig:
    frames: int = 3
    frame_interval_ms: int = 50
    max_observation_age_ms: int = 250
    mode: str = "demo"

    def __post_init__(self) -> None:
        if self.mode != "demo":
            raise ValueError("P0 only implements mode='demo'")
        for name, value, lower, upper in (
            ("frames", self.frames, 1, 10_000),
            ("frame_interval_ms", self.frame_interval_ms, 1, 60_000),
            ("max_observation_age_ms", self.max_observation_age_ms, 0, 60_000),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def load_config(path: Path) -> RuntimeConfig:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) - {"schema_version", "runtime"}:
        raise ValueError("unknown root configuration keys")
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("unsupported configuration schema_version")
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a TOML table")
    if set(runtime) - {"frames", "frame_interval_ms", "max_observation_age_ms", "mode"}:
        raise ValueError("unknown runtime configuration keys")
    return RuntimeConfig(**runtime)
