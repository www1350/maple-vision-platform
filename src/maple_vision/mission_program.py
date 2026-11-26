"""Validated finite sequences of visual objectives on a declared map route."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import canonical_json, strict_json
from .local_files import read_regular
from .map_pack import MapPack

PHASES = ("navigate", "fight", "collect", "talk")
BRANCH_CONDITIONS = ("always", "loot_visible", "no_loot_visible")


@dataclass(frozen=True)
class NavigationObjective:
    target_platform: str
    required_edge_kind: str = "ladder"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_platform, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.target_platform) is None
        ):
            raise ValueError("invalid navigation target platform")
        if self.required_edge_kind not in {"ladder"}:
            raise ValueError("unsupported required navigation edge")

    def to_dict(self) -> dict[str, str]:
        return {
            "target_platform": self.target_platform,
            "required_edge_kind": self.required_edge_kind,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NavigationObjective":
        if not isinstance(value, dict) or set(value) != {
            "target_platform",
            "required_edge_kind",
        }:
            raise ValueError("invalid navigation objective")
        return cls(value["target_platform"], value["required_edge_kind"])


@dataclass(frozen=True)
class MissionTransition:
    condition: str
    target_map: str

    def __post_init__(self) -> None:
        if self.condition not in BRANCH_CONDITIONS:
            raise ValueError("unsupported mission transition condition")
        if (
            not isinstance(self.target_map, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", self.target_map) is None
        ):
            raise ValueError("invalid mission transition target")

    def to_dict(self) -> dict[str, str]:
        return {"condition": self.condition, "target_map": self.target_map}

    @classmethod
    def from_dict(cls, value: object) -> "MissionTransition":
        if not isinstance(value, dict) or set(value) != {"condition", "target_map"}:
            raise ValueError("invalid mission transition")
        return cls(value["condition"], value["target_map"])


@dataclass(frozen=True)
class MissionStage:
    map_id: str
    phases: tuple[str, ...]
    navigation: NavigationObjective | None = None
    transitions: tuple[MissionTransition, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.map_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", self.map_id) is None
        ):
            raise ValueError("invalid mission map ID")
        if (
            not isinstance(self.phases, tuple)
            or not 1 <= len(self.phases) <= 4
            or any(p not in PHASES for p in self.phases)
            or len(set(self.phases)) != len(self.phases)
        ):
            raise ValueError(
                "each map needs 1..4 distinct navigate, fight, collect or talk objectives"
            )
        if ("navigate" in self.phases) != (self.navigation is not None):
            raise ValueError("navigate phase requires one navigation objective")
        if "navigate" in self.phases and self.phases[0] != "navigate":
            raise ValueError("navigate must be the first map objective")
        if self.navigation is not None and type(self.navigation) is not NavigationObjective:
            raise ValueError("invalid navigation objective")
        if (
            not isinstance(self.transitions, tuple)
            or len(self.transitions) > 2
            or any(type(transition) is not MissionTransition for transition in self.transitions)
        ):
            raise ValueError("mission stage needs zero to two typed transitions")
        if "collect" in self.phases and (
            "fight" not in self.phases or self.phases.index("fight") > self.phases.index("collect")
        ):
            raise ValueError("collect requires a preceding fight objective")

    def to_dict(self, *, schema_version: int | None = None) -> dict[str, Any]:
        if schema_version is None:
            schema_version = 3 if self.transitions else (2 if self.navigation is not None else 1)
        result: dict[str, Any] = {"map_id": self.map_id, "phases": list(self.phases)}
        if schema_version in {2, 3}:
            result["navigation"] = None if self.navigation is None else self.navigation.to_dict()
        if schema_version == 3:
            result["transitions"] = [transition.to_dict() for transition in self.transitions]
        return result


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
        if self.conditional:
            positions = {stage.map_id: index for index, stage in enumerate(self.stages)}
            for index, stage in enumerate(self.stages):
                transitions = stage.transitions
                conditions = {transition.condition for transition in transitions}
                if len(transitions) == 1 and conditions != {"always"}:
                    raise ValueError("a single mission transition must be unconditional")
                if len(transitions) == 2 and conditions != {
                    "loot_visible",
                    "no_loot_visible",
                }:
                    raise ValueError("a mission branch must cover visible loot and no visible loot")
                for transition in transitions:
                    target_index = positions.get(transition.target_map)
                    if target_index is None or target_index <= index:
                        raise ValueError("mission transitions must target a later declared stage")
            reachable = {self.stages[0].map_id}
            for stage in self.stages:
                if stage.map_id in reachable:
                    reachable.update(transition.target_map for transition in stage.transitions)
            if reachable != set(positions):
                raise ValueError("every conditional mission stage must be reachable from the entry")

    @property
    def conditional(self) -> bool:
        return any(stage.transitions for stage in self.stages)

    @property
    def maps(self) -> tuple[str, ...]:
        return tuple(stage.map_id for stage in self.stages)

    def bind_map_pack(self, pack: MapPack | None) -> None:
        if self.conditional:
            if pack is None or self.maps[0] != pack.entry_map:
                raise ValueError("conditional mission must start at its map pack entry")
            for stage in self.stages:
                destinations = {portal.destination for portal in pack.get(stage.map_id).portals}
                if any(
                    transition.target_map not in destinations for transition in stage.transitions
                ):
                    raise ValueError("mission transition needs a matching directed portal")
            return
        if pack is None or all(len(m.portals) == 1 for m in pack.maps):
            self.bind_route(("meadow", "cavern") if pack is None else pack.map_ids)
            return
        if self.maps[0] != pack.entry_map:
            raise ValueError("mission must start at the map pack entry")
        for source, target in zip(self.maps, self.maps[1:]):
            if not any(p.destination == target for p in pack.get(source).portals):
                raise ValueError("mission maps must be connected by a directed portal")

    def bind_route(self, route: tuple[str, ...]) -> None:
        if self.conditional:
            if self.maps != route:
                raise ValueError("conditional mission runtime needs every declared stage")
            return
        if self.maps != route[: len(self.stages)]:
            raise ValueError("mission must follow a prefix of the available map route")

    def transition(self, map_id: str, *, loot_visible: bool) -> MissionTransition | None:
        if type(loot_visible) is not bool:
            raise ValueError("loot visibility must be boolean")
        try:
            stage = next(stage for stage in self.stages if stage.map_id == map_id)
        except StopIteration as exc:
            raise ValueError("mission transition source is not declared") from exc
        for transition in stage.transitions:
            if transition.condition == "always" or transition.condition == (
                "loot_visible" if loot_visible else "no_loot_visible"
            ):
                return transition
        return None

    def to_dict(self) -> dict[str, Any]:
        schema_version = (
            3
            if self.conditional
            else (2 if any(stage.navigation is not None for stage in self.stages) else 1)
        )
        return {
            "schema_version": schema_version,
            "stages": [stage.to_dict(schema_version=schema_version) for stage in self.stages],
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "MissionProgram":
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "stages"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] not in {1, 2, 3}
            or not isinstance(value["stages"], list)
        ):
            raise ValueError("invalid mission program schema")
        stages = []
        schema_version = value["schema_version"]
        for row in value["stages"]:
            expected = {"map_id", "phases"}
            if schema_version in {2, 3}:
                expected.add("navigation")
            if schema_version == 3:
                expected.add("transitions")
            if (
                not isinstance(row, dict)
                or set(row) != expected
                or not isinstance(row["phases"], list)
            ):
                raise ValueError("invalid mission stage")
            navigation = None
            if schema_version == 2 and row["navigation"] is not None:
                navigation = NavigationObjective.from_dict(row["navigation"])
            if schema_version == 3 and row["navigation"] is not None:
                navigation = NavigationObjective.from_dict(row["navigation"])
            transitions: tuple[MissionTransition, ...] = ()
            if schema_version == 3:
                if not isinstance(row["transitions"], list):
                    raise ValueError("invalid mission transitions")
                transitions = tuple(
                    MissionTransition.from_dict(transition) for transition in row["transitions"]
                )
            stages.append(
                MissionStage(row["map_id"], tuple(row["phases"]), navigation, transitions)
            )
        result = cls(tuple(stages))
        if result.to_dict()["schema_version"] != schema_version:
            raise ValueError("mission program schema does not match its navigation objectives")
        return result


def load_mission_program(path: Path) -> MissionProgram:
    return MissionProgram.from_dict(
        strict_json(read_regular(path, 16 * 1024, "mission program").decode())
    )
