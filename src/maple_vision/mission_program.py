"""Validated finite sequences of visual objectives on a declared map route."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import canonical_json, strict_json
from .local_files import read_regular

PHASES = ("fight", "collect", "talk")


@dataclass(frozen=True)
class MissionStage:
    map_id: str
    phases: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.map_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", self.map_id) is None
        ):
            raise ValueError("invalid mission map ID")
        if (
            not isinstance(self.phases, tuple)
            or not 1 <= len(self.phases) <= 3
            or any(p not in PHASES for p in self.phases)
            or len(set(self.phases)) != len(self.phases)
        ):
            raise ValueError("each map needs 1..3 distinct fight, collect or talk objectives")
        if "collect" in self.phases and (
            "fight" not in self.phases or self.phases.index("fight") > self.phases.index("collect")
        ):
            raise ValueError("collect requires a preceding fight objective")

    def to_dict(self) -> dict[str, Any]:
        return {"map_id": self.map_id, "phases": list(self.phases)}


@dataclass(frozen=True)
class MissionProgram:
    stages: tuple[MissionStage, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stages, tuple)
            or not 1 <= len(self.stages) <= 8
            or any(type(stage) is not MissionStage for stage in self.stages)
            or len({stage.map_id for stage in self.stages}) != len(self.stages)
        ):
            raise ValueError("mission needs 1..8 distinct, typed map stages")

    @property
    def maps(self) -> tuple[str, ...]:
        return tuple(stage.map_id for stage in self.stages)

    def bind_route(self, route: tuple[str, ...]) -> None:
        if self.maps != route[: len(self.stages)]:
            raise ValueError("mission must follow a prefix of the available map route")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "stages": [stage.to_dict() for stage in self.stages]}

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "MissionProgram":
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "stages"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not isinstance(value["stages"], list)
        ):
            raise ValueError("invalid mission program schema")
        stages = []
        for row in value["stages"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"map_id", "phases"}
                or not isinstance(row["phases"], list)
            ):
                raise ValueError("invalid mission stage")
            stages.append(MissionStage(row["map_id"], tuple(row["phases"])))
        return cls(tuple(stages))


def load_mission_program(path: Path) -> MissionProgram:
    return MissionProgram.from_dict(
        strict_json(read_regular(path, 16 * 1024, "mission program").decode())
    )
