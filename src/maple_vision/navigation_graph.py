"""Bounded static world routes and executable fixture-motion platform graphs.

These planners consume authored geometry, explicit capabilities and a caller's
motion estimate. They never read a Simulator, live snapshots or dynamic truth.
The local graph is implicit: vertices are complete motion states, not merely
platform IDs. Every edge is a fixed-step movement witness with exact endpoints.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .catalog import canonical_json, strict_json, validate_provenance
from .map_pack import MapPack
from .sim_types import ClassSpec, SimAction

MAX_GRAPH_BYTES = 128 * 1024
MAX_WORLD_NODES = 128
MAX_WORLD_PORTALS = 512
MAX_PLATFORMS = 32
MAX_LADDERS = 16
PHYSICS_MODEL = "synthetic-fixture-motion-v1"
PRIMITIVE_SET = "settled-constant-control-v1"
WIDTH, HEIGHT, GROUND = 320, 192, 168
PLAYER_WIDTH, PLAYER_HEIGHT, GRAVITY, CLIMB_SPEED = 14, 22, 650, 60
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NAV_ACTIONS = frozenset({"idle", "move", "jump", "climb"})


def _integer(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _number(value: object, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{label} is outside its finite range")
    return float(value) if value else 0.0


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a bounded lowercase identifier")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be SHA256 hex")
    return value


def _object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has invalid fields")
    return value


def _array(value: object, label: str, low: int, high: int) -> list[Any]:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise ValueError(f"{label} requires {low}..{high} entries")
    return value


def _entries(value: object, cls: type, label: str, low: int, high: int) -> None:
    if (
        not isinstance(value, tuple)
        or not low <= len(value) <= high
        or any(type(item) is not cls for item in value)
    ):
        raise ValueError(f"{label} requires {low}..{high} immutable {cls.__name__} entries")


def _schema(raw: dict[str, Any], kind: str) -> None:
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported navigation schema")
    if raw["kind"] != kind:
        raise ValueError("wrong navigation document kind")


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _manifest(content: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "content": content, "content_sha256": _hash(content)}


def _envelope(value: object) -> dict[str, Any]:
    raw = _object(value, {"schema_version", "content", "content_sha256"}, "graph manifest")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported graph manifest schema")
    _digest(raw["content_sha256"], "content hash")
    return raw


def _check_size(value: object) -> None:
    try:
        size = len(canonical_json(value).encode("utf-8"))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid bounded graph text") from exc
    if size > MAX_GRAPH_BYTES:
        raise ValueError("navigation document exceeds byte budget")


def _load(path: Path) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("navigation source must be a regular file, not a symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_GRAPH_BYTES + 1)
    if len(raw) > MAX_GRAPH_BYTES:
        raise ValueError("navigation source exceeds byte budget")
    try:
        return strict_json(raw.decode("utf-8"))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("navigation source is not bounded UTF-8 JSON") from exc


@dataclass(frozen=True)
class GraphProvenance:
    source_uri: str
    permission: str
    license: str
    game_version: str
    scope: str = "synthetic_fixture"
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_provenance(self.source_uri, self.permission, self.license)
        if (
            not isinstance(self.game_version, str)
            or not self.game_version.strip()
            or len(self.game_version) > 64
            or any(ord(char) < 32 for char in self.game_version)
        ):
            raise ValueError("game_version must be bounded nonempty text")
        if not isinstance(self.scope, str) or self.scope not in {
            "synthetic_fixture",
            "game_reference",
        }:
            raise ValueError("unsupported graph provenance scope")
        if self.source_sha256 is not None:
            _digest(self.source_sha256, "source SHA256")
        _check_size(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> GraphProvenance:
        return cls(**_object(value, set(cls.__dataclass_fields__), "graph provenance"))


@dataclass(frozen=True)
class WorldPortal:
    source: str
    portal_id: str
    target: str
    cost: int = 1

    def __post_init__(self) -> None:
        for field in ("source", "portal_id", "target"):
            _identifier(getattr(self, field), field)
        _integer(self.cost, "portal cost", 1, 1_000_000)

    @property
    def identity(self) -> tuple[str, str]:
        """Portal identity is local to a map; two maps may both name an exit 'east'."""
        return self.source, self.portal_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> WorldPortal:
        return cls(**_object(value, set(cls.__dataclass_fields__), "world portal"))


@dataclass(frozen=True)
class WorldRoute:
    graph_sha256: str
    source: str
    target: str
    portals: tuple[WorldPortal, ...]

    def __post_init__(self) -> None:
        _digest(self.graph_sha256, "world graph hash")
        _identifier(self.source, "route source")
        _identifier(self.target, "route target")
        _entries(self.portals, WorldPortal, "route portals", 0, MAX_WORLD_NODES - 1)
        current = self.source
        for portal in self.portals:
            if portal.source != current:
                raise ValueError("world route is not composable")
            current = portal.target
        if current != self.target:
            raise ValueError("world route does not reach its target")

    @property
    def total_cost(self) -> int:
        return sum(item.cost for item in self.portals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_sha256": self.graph_sha256,
            "source": self.source,
            "target": self.target,
            "total_cost": self.total_cost,
            "portals": [item.to_dict() for item in self.portals],
        }


@dataclass(frozen=True)
class WorldDiagnostics:
    entry: str
    reachable: tuple[str, ...]
    unreachable: tuple[str, ...]
    isolated: tuple[str, ...]
    dead_ends: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "reachable": list(self.reachable),
            "unreachable": list(self.unreachable),
            "isolated": list(self.isolated),
            "dead_ends": list(self.dead_ends),
        }


@dataclass(frozen=True)
class WorldMapGraph:
    id: str
    nodes: tuple[str, ...]
    portals: tuple[WorldPortal, ...]
    provenance: GraphProvenance

    def __post_init__(self) -> None:
        _identifier(self.id, "world graph id")
        _entries(self.nodes, str, "world nodes", 1, MAX_WORLD_NODES)
        for node in self.nodes:
            _identifier(node, "map id")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("world graph contains duplicate map nodes")
        _entries(self.portals, WorldPortal, "world portals", 0, MAX_WORLD_PORTALS)
        if len({portal.identity for portal in self.portals}) != len(self.portals):
            raise ValueError("world graph contains duplicate map-local portal identities")
        if any(
            item.source not in self.nodes or item.target not in self.nodes for item in self.portals
        ):
            raise ValueError("world portal source or target does not name a node")
        if type(self.provenance) is not GraphProvenance:
            raise ValueError("world graph needs explicit static provenance")
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes)))
        object.__setattr__(self, "portals", tuple(sorted(self.portals, key=lambda p: p.identity)))
        _check_size(self.to_dict())

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "world-map-graph",
            "id": self.id,
            "nodes": list(self.nodes),
            "portals": [item.to_dict() for item in self.portals],
            "provenance": self.provenance.to_dict(),
        }

    def manifest(self) -> dict[str, Any]:
        return _manifest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> WorldMapGraph:
        raw = _object(
            value, {"schema_version", "kind", "id", "nodes", "portals", "provenance"}, "world graph"
        )
        _schema(raw, "world-map-graph")
        nodes = _array(raw["nodes"], "world nodes", 1, MAX_WORLD_NODES)
        portals = _array(raw["portals"], "world portals", 0, MAX_WORLD_PORTALS)
        return cls(
            raw["id"],
            tuple(nodes),
            tuple(WorldPortal.from_dict(p) for p in portals),
            GraphProvenance.from_dict(raw["provenance"]),
        )

    @classmethod
    def from_manifest(cls, value: object) -> WorldMapGraph:
        raw = _envelope(value)
        graph = cls.from_dict(raw["content"])
        if graph.content_sha256 != raw["content_sha256"]:
            raise ValueError("world graph content SHA256 mismatch")
        return graph

    @classmethod
    def from_map_pack(cls, pack: MapPack) -> WorldMapGraph:
        if type(pack) is not MapPack:
            raise ValueError("world adaptation requires a verified MapPack")
        return cls(
            pack.id,
            tuple(item.id for item in pack.maps),
            tuple(
                WorldPortal(item.id, portal.id, portal.destination)
                for item in pack.maps
                for portal in item.portals
            ),
            GraphProvenance(
                pack.source_uri,
                pack.permission,
                pack.license,
                pack.version,
                source_sha256=pack.content_sha256,
            ),
        )

    def shortest_path(self, source: str, target: str) -> WorldRoute | None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("route endpoint is not a world node")
        # Positive weights mean an optimal route is simple (at most N-1 doors).
        # Ties use cost, hop count, then the complete map-local portal sequence.
        adjacency = {
            node: tuple(p for p in self.portals if p.source == node) for node in self.nodes
        }
        best: dict[str, tuple[int, int, tuple[tuple[str, str], ...]]] = {source: (0, 0, ())}
        queue: list[tuple[int, int, tuple[tuple[str, str], ...], str, tuple[WorldPortal, ...]]] = [
            (0, 0, (), source, ())
        ]
        while queue:
            cost, hops, key, node, path = heapq.heappop(queue)
            if best[node] != (cost, hops, key):
                continue
            if node == target:
                return WorldRoute(self.content_sha256, source, target, path)
            for portal in adjacency[node]:
                candidate = cost + portal.cost, hops + 1, key + (portal.identity,)
                previous = best.get(portal.target)
                if previous is None or candidate < previous:
                    best[portal.target] = candidate
                    heapq.heappush(queue, (*candidate, portal.target, path + (portal,)))
        return None

    def diagnostics(self, entry: str) -> WorldDiagnostics:
        if entry not in self.nodes:
            raise ValueError("diagnostic entry is not a world node")
        seen = {entry}
        pending = [entry]
        while pending:
            node = pending.pop()
            for portal in self.portals:
                if portal.source == node and portal.target not in seen:
                    seen.add(portal.target)
                    pending.append(portal.target)
        connected = {
            node for p in self.portals if p.source != p.target for node in (p.source, p.target)
        }
        outgoing = {p.source for p in self.portals if p.source != p.target}
        return WorldDiagnostics(
            entry,
            tuple(sorted(seen)),
            tuple(n for n in self.nodes if n not in seen),
            tuple(n for n in self.nodes if n not in connected),
            tuple(n for n in self.nodes if n not in outgoing),
        )

    def verify_route(self, route: WorldRoute) -> dict[str, Any]:
        if type(route) is not WorldRoute or route.graph_sha256 != self.content_sha256:
            raise ValueError("world route is not bound to this graph")
        if route.source not in self.nodes or route.target not in self.nodes:
            raise ValueError("world route endpoint does not name a node")
        if any(portal not in self.portals for portal in route.portals):
            raise ValueError("world route contains an undeclared portal or cost")
        return {
            "verified": True,
            "graph_sha256": self.content_sha256,
            "total_cost": route.total_cost,
            "hops": len(route.portals),
            "physical_portal_execution_evaluated": False,
        }


def load_world_graph(path: Path) -> WorldMapGraph:
    raw = _load(path)
    if isinstance(raw, dict) and "content" in raw:
        return WorldMapGraph.from_manifest(raw)
    return WorldMapGraph.from_dict(raw)


@dataclass(frozen=True)
class PlatformSurface:
    id: str
    x1: int
    x2: int
    y: int

    def __post_init__(self) -> None:
        _identifier(self.id, "platform id")
        _integer(self.x1, "platform x1", 0, WIDTH - 1)
        _integer(self.x2, "platform x2", self.x1 + 1, WIDTH)
        _integer(self.y, "platform y", PLAYER_HEIGHT, GROUND)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> PlatformSurface:
        return cls(**_object(value, set(cls.__dataclass_fields__), "platform"))


@dataclass(frozen=True)
class NavigationLadder:
    id: str
    x: int
    y: int
    h: int
    w: int = 10

    def __post_init__(self) -> None:
        _identifier(self.id, "ladder id")
        _integer(self.x, "ladder x", 2, WIDTH - 12)
        _integer(self.y, "ladder y", PLAYER_HEIGHT, GROUND - 1)
        _integer(self.h, "ladder height", 1, GROUND - PLAYER_HEIGHT)
        _integer(self.w, "ladder width", 10, 10)
        if self.y + self.h != GROUND:
            raise ValueError("fixture ladders must terminate at ground")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> NavigationLadder:
        return cls(**_object(value, set(cls.__dataclass_fields__), "ladder"))


@dataclass(frozen=True)
class PlatformGeometry:
    map_id: str
    platforms: tuple[PlatformSurface, ...]
    ladders: tuple[NavigationLadder, ...]
    provenance: GraphProvenance

    def __post_init__(self) -> None:
        _identifier(self.map_id, "platform map id")
        _entries(self.platforms, PlatformSurface, "platforms", 1, MAX_PLATFORMS)
        _entries(self.ladders, NavigationLadder, "ladders", 0, MAX_LADDERS)
        if type(self.provenance) is not GraphProvenance:
            raise ValueError("platform geometry needs explicit static provenance")
        if len({p.id for p in self.platforms}) != len(self.platforms):
            raise ValueError("duplicate platform id")
        if len({p.id for p in self.ladders}) != len(self.ladders):
            raise ValueError("duplicate ladder id")
        if {p.id for p in self.platforms} & {p.id for p in self.ladders}:
            raise ValueError("platform and ladder ids must be distinct")
        floors = [p for p in self.platforms if p.y == GROUND]
        if len(floors) != 1 or (floors[0].x1, floors[0].x2) != (0, WIDTH):
            raise ValueError("fixture geometry requires exactly one full-width ground")
        for index, platform in enumerate(self.platforms):
            if any(
                platform.y == other.y and max(platform.x1, other.x1) <= min(platform.x2, other.x2)
                for other in self.platforms[index + 1 :]
            ):
                raise ValueError("same-height platforms cannot overlap or touch")
        for ladder in self.ladders:
            if not any(
                p.y == ladder.y and p.x1 <= ladder.x and ladder.x + ladder.w <= p.x2
                for p in self.platforms
            ):
                raise ValueError("ladder top must attach to a supporting platform")
        object.__setattr__(self, "platforms", tuple(sorted(self.platforms, key=lambda p: p.id)))
        # Ladder order is semantic: the fixture acquires the first ladder in range.
        _check_size(self.to_dict())

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict())

    @property
    def platform_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.platforms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "platform-geometry",
            "map_id": self.map_id,
            "physics_model": PHYSICS_MODEL,
            "width": WIDTH,
            "height": HEIGHT,
            "ground": GROUND,
            "platforms": [p.to_dict() for p in self.platforms],
            "ladders": [p.to_dict() for p in self.ladders],
            "provenance": self.provenance.to_dict(),
        }

    def manifest(self) -> dict[str, Any]:
        return _manifest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> PlatformGeometry:
        raw = _object(
            value,
            {
                "schema_version",
                "kind",
                "map_id",
                "physics_model",
                "width",
                "height",
                "ground",
                "platforms",
                "ladders",
                "provenance",
            },
            "platform geometry",
        )
        _schema(raw, "platform-geometry")
        if raw["physics_model"] != PHYSICS_MODEL:
            raise ValueError("unsupported platform physics model")
        for key, expected in (("width", WIDTH), ("height", HEIGHT), ("ground", GROUND)):
            _integer(raw[key], key, expected, expected)
        platforms = _array(raw["platforms"], "platforms", 1, MAX_PLATFORMS)
        ladders = _array(raw["ladders"], "ladders", 0, MAX_LADDERS)
        return cls(
            raw["map_id"],
            tuple(PlatformSurface.from_dict(p) for p in platforms),
            tuple(NavigationLadder.from_dict(p) for p in ladders),
            GraphProvenance.from_dict(raw["provenance"]),
        )

    @classmethod
    def from_manifest(cls, value: object) -> PlatformGeometry:
        raw = _envelope(value)
        result = cls.from_dict(raw["content"])
        if result.content_sha256 != raw["content_sha256"]:
            raise ValueError("platform geometry content SHA256 mismatch")
        return result

    @classmethod
    def from_map_pack(cls, pack: MapPack, map_id: str) -> PlatformGeometry:
        if type(pack) is not MapPack:
            raise ValueError("platform adaptation requires a verified MapPack")
        item = pack.get(map_id)
        return cls(
            item.id,
            tuple(
                PlatformSurface(
                    f"platform_{i}",
                    _integer(p.x1, "adapted platform x1", 0, WIDTH - 1),
                    _integer(p.x2, "adapted platform x2", 1, WIDTH),
                    _integer(p.y, "adapted platform y", PLAYER_HEIGHT, GROUND),
                )
                for i, p in enumerate(item.platforms)
            ),
            tuple(
                NavigationLadder(
                    f"ladder_{i}",
                    _integer(p.x, "adapted ladder x", 2, WIDTH - 12),
                    _integer(p.y, "adapted ladder y", PLAYER_HEIGHT, GROUND - 1),
                    _integer(p.h, "adapted ladder height", 1, GROUND - PLAYER_HEIGHT),
                    _integer(p.w, "adapted ladder width", 10, 10),
                )
                for i, p in enumerate(item.ladders)
            ),
            GraphProvenance(
                pack.source_uri,
                pack.permission,
                pack.license,
                pack.version,
                source_sha256=pack.content_sha256,
            ),
        )


def load_platform_geometry(path: Path) -> PlatformGeometry:
    raw = _load(path)
    if isinstance(raw, dict) and "content" in raw:
        return PlatformGeometry.from_manifest(raw)
    return PlatformGeometry.from_dict(raw)


@dataclass(frozen=True)
class MovementCapabilities:
    walk_speed: int = 80
    jump_speed: int = 280
    step_ms: int = 50
    can_walk: bool = True
    can_jump: bool = True
    can_drop: bool = True
    can_climb: bool = True

    def __post_init__(self) -> None:
        _integer(self.walk_speed, "walk speed", 1, 200)
        _integer(self.jump_speed, "jump speed", 1, 400)
        _integer(self.step_ms, "step milliseconds", 10, 100)
        for key in ("can_walk", "can_jump", "can_drop", "can_climb"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "movement-capabilities",
            "physics_model": PHYSICS_MODEL,
            "body_width": PLAYER_WIDTH,
            "body_height": PLAYER_HEIGHT,
            "gravity": GRAVITY,
            "climb_speed": CLIMB_SPEED,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, value: object) -> MovementCapabilities:
        fields = set(cls.__dataclass_fields__)
        raw = _object(
            value,
            fields
            | {
                "schema_version",
                "kind",
                "physics_model",
                "body_width",
                "body_height",
                "gravity",
                "climb_speed",
            },
            "movement capabilities",
        )
        _schema(raw, "movement-capabilities")
        if raw["physics_model"] != PHYSICS_MODEL:
            raise ValueError("unsupported capability physics model")
        for key, expected in (
            ("body_width", PLAYER_WIDTH),
            ("body_height", PLAYER_HEIGHT),
            ("gravity", GRAVITY),
            ("climb_speed", CLIMB_SPEED),
        ):
            _integer(raw[key], key, expected, expected)
        return cls(**{key: raw[key] for key in fields})

    @classmethod
    def from_class_spec(cls, spec: ClassSpec, *, step_ms: int = 50) -> MovementCapabilities:
        if type(spec) is not ClassSpec:
            raise ValueError("movement adaptation requires a static ClassSpec")
        return cls(walk_speed=spec.walk_speed, jump_speed=spec.jump_speed, step_ms=step_ms)


def load_movement_capabilities(path: Path) -> MovementCapabilities:
    return MovementCapabilities.from_dict(_load(path))


@dataclass(frozen=True)
class SearchBudget:
    max_expanded_states: int = 4096
    max_generated_edges: int = 32768
    max_simulated_steps: int = 500_000
    max_primitive_steps: int = 512
    max_path_steps: int = 16384

    def __post_init__(self) -> None:
        for key, maximum in (
            ("max_expanded_states", 16384),
            ("max_generated_edges", 131072),
            ("max_simulated_steps", 2_000_000),
            ("max_primitive_steps", 2048),
            ("max_path_steps", 65536),
        ):
            _integer(getattr(self, key), key, 1, maximum)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SearchBudget:
        return cls(**_object(value, set(cls.__dataclass_fields__), "search budget"))


@dataclass(frozen=True)
class MotionState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = True
    climbing: bool = False

    def __post_init__(self) -> None:
        for key, low, high in (
            ("x", 0, WIDTH - PLAYER_WIDTH),
            ("y", 0, GROUND - PLAYER_HEIGHT),
            ("vx", -200, 200),
            ("vy", -400, 2000),
        ):
            object.__setattr__(self, key, _number(getattr(self, key), key, low, high))
        if type(self.grounded) is not bool or type(self.climbing) is not bool:
            raise ValueError("motion flags must be boolean")
        if self.grounded and (self.climbing or self.vy != 0):
            raise ValueError("grounded motion cannot climb or have vertical velocity")
        if self.climbing and (self.vx != 0 or self.vy != 0):
            raise ValueError("climbing motion must have zero free-flight velocity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> MotionState:
        return cls(**_object(value, set(cls.__dataclass_fields__), "motion state"))


def _support(geometry: PlatformGeometry, state: MotionState) -> tuple[str, ...]:
    if not state.grounded:
        return ()
    return tuple(
        p.id
        for p in geometry.platforms
        if abs(state.y + PLAYER_HEIGHT - p.y) <= 1e-6
        and state.x + PLAYER_WIDTH > p.x1
        and state.x < p.x2
    )


def _validate_state(geometry: PlatformGeometry, state: MotionState) -> None:
    if type(state) is not MotionState:
        raise ValueError("navigation requires an explicit MotionState estimate")
    if state.grounded and not _support(geometry, state):
        raise ValueError("grounded state has no supporting platform")
    if state.climbing and not any(
        abs(state.x + PLAYER_WIDTH / 2 - (p.x + p.w / 2)) <= 1e-6
        and p.y - PLAYER_HEIGHT <= state.y <= p.y + p.h - PLAYER_HEIGHT
        for p in geometry.ladders
    ):
        raise ValueError("climbing state must occupy an authored ladder")


def _navigation_action(action: object) -> SimAction:
    if type(action) is not SimAction or action.kind not in _NAV_ACTIONS:
        raise ValueError("navigation primitives may only idle, move, jump or climb")
    return action


def _available(capabilities: MovementCapabilities, action: SimAction) -> bool:
    return (
        action.kind == "idle"
        or {
            "move": capabilities.can_walk,
            "jump": capabilities.can_jump,
            "climb": capabilities.can_climb,
        }[action.kind]
    )


def _physical_step(
    geometry: PlatformGeometry,
    capabilities: MovementCapabilities,
    state: MotionState,
    action: SimAction,
) -> tuple[MotionState, bool]:
    """Independent implementation of the declared fixture's discrete motion law."""
    x, y, vx, vy = state.x, state.y, state.vx, state.vy
    grounded, climbing = state.grounded, state.climbing
    accepted = True
    if grounded:
        vx = 0.0
    if action.kind not in {"climb", "idle"}:
        climbing = False
    if action.kind == "move":
        vx = float(capabilities.walk_speed * action.direction)
    elif action.kind == "jump":
        if not grounded:
            accepted = False
        else:
            vy = float(-capabilities.jump_speed)
            vx = float(capabilities.walk_speed * action.direction)
            grounded = False
    elif action.kind == "climb":
        accepted = False
        for ladder in geometry.ladders:
            if (
                abs(ladder.x + ladder.w / 2 - (x + PLAYER_WIDTH / 2)) <= 12
                and y + PLAYER_HEIGHT >= ladder.y
                and y <= ladder.y + ladder.h
            ):
                climbing, grounded, vx, vy = True, False, 0.0, 0.0
                x = ladder.x + ladder.w / 2 - PLAYER_WIDTH / 2
                y = max(
                    ladder.y - PLAYER_HEIGHT,
                    min(
                        ladder.y + ladder.h - PLAYER_HEIGHT,
                        y + action.direction * CLIMB_SPEED * capabilities.step_ms / 1000,
                    ),
                )
                if y + PLAYER_HEIGHT <= ladder.y or y + PLAYER_HEIGHT >= GROUND:
                    climbing, grounded = False, True
                accepted = True
                break
    if not climbing:
        dt = capabilities.step_ms / 1000
        old_bottom = y + PLAYER_HEIGHT
        x = max(0.0, min(float(WIDTH - PLAYER_WIDTH), x + vx * dt))
        vy += GRAVITY * dt
        new_y = y + vy * dt
        landing = [
            p.y
            for p in geometry.platforms
            if vy >= 0
            and x + PLAYER_WIDTH > p.x1
            and x < p.x2
            and old_bottom <= p.y + 1e-7
            and new_y + PLAYER_HEIGHT >= p.y
        ]
        if landing:
            y, vy, grounded = float(min(landing) - PLAYER_HEIGHT), 0.0, True
        else:
            y, grounded = max(0.0, new_y), False
            if new_y < 0:
                vy = 0.0
    return MotionState(x, y, vx, vy, grounded, climbing), accepted


def _trace_hash(start: MotionState) -> Any:
    digest = hashlib.sha256()
    digest.update((canonical_json(start.to_dict()) + "\n").encode())
    return digest


def _trace_add(digest: Any, action: SimAction, state: MotionState, accepted: bool) -> None:
    digest.update(
        (
            canonical_json(
                {"action": asdict(action), "state": state.to_dict(), "accepted": accepted}
            )
            + "\n"
        ).encode()
    )


@dataclass(frozen=True)
class MotionTrace:
    states: tuple[MotionState, ...]
    accepted: tuple[bool, ...]
    trajectory_sha256: str


def simulate_actions(
    geometry: PlatformGeometry,
    capabilities: MovementCapabilities,
    start: MotionState,
    actions: tuple[SimAction, ...],
    *,
    max_steps: int = 16384,
) -> MotionTrace:
    if type(geometry) is not PlatformGeometry or type(capabilities) is not MovementCapabilities:
        raise ValueError("motion replay requires static geometry and capabilities")
    _integer(max_steps, "motion replay budget", 1, 65536)
    _entries(actions, SimAction, "motion actions", 0, max_steps)
    _validate_state(geometry, start)
    states, accepted_steps, state = [start], [], start
    digest = _trace_hash(start)
    for action in actions:
        _navigation_action(action)
        if not _available(capabilities, action):
            raise ValueError("motion action is disabled by capabilities")
        state, accepted = _physical_step(geometry, capabilities, state, action)
        _trace_add(digest, action, state, accepted)
        states.append(state)
        accepted_steps.append(accepted)
    return MotionTrace(tuple(states), tuple(accepted_steps), digest.hexdigest())


@dataclass(frozen=True)
class MovementPrimitive:
    action: SimAction
    ticks: int

    def __post_init__(self) -> None:
        _navigation_action(self.action)
        _integer(self.ticks, "primitive ticks", 1, 2048)

    def to_dict(self) -> dict[str, Any]:
        return {"action": asdict(self.action), "ticks": self.ticks}

    @classmethod
    def from_dict(cls, value: object) -> MovementPrimitive:
        raw = _object(value, {"action", "ticks"}, "movement primitive")
        action = _object(raw["action"], set(SimAction.__dataclass_fields__), "navigation action")
        return cls(SimAction(**action), raw["ticks"])


@dataclass(frozen=True)
class PlatformEdge:
    kind: str
    start: MotionState
    end: MotionState
    primitives: tuple[MovementPrimitive, ...]
    source_platforms: tuple[str, ...]
    target_platforms: tuple[str, ...]
    geometry_sha256: str
    capabilities_sha256: str
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {
            "walk",
            "jump",
            "drop",
            "ladder",
            "settle",
        }:
            raise ValueError("unsupported platform edge kind")
        if type(self.start) is not MotionState or type(self.end) is not MotionState:
            raise ValueError("platform edge requires exact motion endpoints")
        if not self.end.grounded or self.end.climbing or self.end.vx or self.end.vy:
            raise ValueError("platform edge must end in settled grounded motion")
        _entries(self.primitives, MovementPrimitive, "edge primitives", 1, 4)
        if self.steps > 2048:
            raise ValueError("platform edge exceeds its primitive step budget")
        for label in ("source_platforms", "target_platforms"):
            values = getattr(self, label)
            _entries(values, str, label, 0 if label == "source_platforms" else 1, MAX_PLATFORMS)
            for value in values:
                _identifier(value, label)
            if tuple(sorted(set(values))) != values:
                raise ValueError("platform support identities must be unique and sorted")
        for label in ("geometry_sha256", "capabilities_sha256", "trajectory_sha256"):
            _digest(getattr(self, label), label)

    @property
    def steps(self) -> int:
        return sum(item.ticks for item in self.primitives)

    @property
    def actions(self) -> tuple[SimAction, ...]:
        return tuple(item.action for item in self.primitives for _ in range(item.ticks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "primitives": [p.to_dict() for p in self.primitives],
            "source_platforms": list(self.source_platforms),
            "target_platforms": list(self.target_platforms),
            "geometry_sha256": self.geometry_sha256,
            "capabilities_sha256": self.capabilities_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlatformEdge:
        raw = _object(value, set(cls.__dataclass_fields__) | {"steps"}, "platform edge")
        primitives = _array(raw["primitives"], "edge primitives", 1, 4)
        source = _array(raw["source_platforms"], "source supports", 0, MAX_PLATFORMS)
        target = _array(raw["target_platforms"], "target supports", 1, MAX_PLATFORMS)
        edge = cls(
            raw["kind"],
            MotionState.from_dict(raw["start"]),
            MotionState.from_dict(raw["end"]),
            tuple(MovementPrimitive.from_dict(p) for p in primitives),
            tuple(source),
            tuple(target),
            raw["geometry_sha256"],
            raw["capabilities_sha256"],
            raw["trajectory_sha256"],
        )
        if type(raw["steps"]) is not int or raw["steps"] != edge.steps:
            raise ValueError("edge step count mismatch")
        return edge


@dataclass(frozen=True)
class NavigationPath:
    graph_sha256: str
    map_id: str
    initial_state: MotionState
    target_platform: str
    edges: tuple[PlatformEdge, ...]

    def __post_init__(self) -> None:
        _digest(self.graph_sha256, "platform graph hash")
        _identifier(self.map_id, "navigation map")
        _identifier(self.target_platform, "target platform")
        if type(self.initial_state) is not MotionState:
            raise ValueError("path requires an exact initial motion precondition")
        _entries(self.edges, PlatformEdge, "path edges", 0, 16384)
        if self.steps > 65536:
            raise ValueError("navigation path exceeds its step budget")
        previous = self.initial_state
        for edge in self.edges:
            if edge.start != previous:
                raise ValueError("platform path has incompatible motion endpoints")
            previous = edge.end

    @property
    def steps(self) -> int:
        return sum(edge.steps for edge in self.edges)

    @property
    def end(self) -> MotionState:
        return self.edges[-1].end if self.edges else self.initial_state

    @property
    def actions(self) -> tuple[SimAction, ...]:
        return tuple(action for edge in self.edges for action in edge.actions)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "navigation-path",
            "graph_sha256": self.graph_sha256,
            "map_id": self.map_id,
            "initial_state": self.initial_state.to_dict(),
            "target_platform": self.target_platform,
            "edges": [edge.to_dict() for edge in self.edges],
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, value: object) -> NavigationPath:
        raw = _object(
            value,
            set(cls.__dataclass_fields__) | {"schema_version", "kind", "steps"},
            "navigation path",
        )
        _schema(raw, "navigation-path")
        edges = _array(raw["edges"], "path edges", 0, 16384)
        result = cls(
            raw["graph_sha256"],
            raw["map_id"],
            MotionState.from_dict(raw["initial_state"]),
            raw["target_platform"],
            tuple(PlatformEdge.from_dict(edge) for edge in edges),
        )
        if type(raw["steps"]) is not int or raw["steps"] != result.steps:
            raise ValueError("path step count mismatch")
        return result


@dataclass(frozen=True)
class EdgeExpansion:
    edges: tuple[PlatformEdge, ...]
    simulated_steps: int
    incomplete: tuple[str, ...]


@dataclass(frozen=True)
class NavigationResult:
    status: str
    reason: str
    path: NavigationPath | None
    expanded_states: int
    generated_edges: int
    simulated_steps: int
    reachable_platforms: tuple[str, ...]
    incomplete_primitives: tuple[str, ...]
    optimal_within_primitive_graph: bool
    budget: SearchBudget

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "reason": self.reason,
            "path": None if self.path is None else self.path.to_dict(),
            "expanded_states": self.expanded_states,
            "generated_edges": self.generated_edges,
            "simulated_steps": self.simulated_steps,
            "reachable_platforms": list(self.reachable_platforms),
            "incomplete_primitives": list(self.incomplete_primitives),
            "optimal_within_primitive_graph": self.optimal_within_primitive_graph,
            "coverage": PRIMITIVE_SET,
            "budget": self.budget.to_dict(),
        }


@dataclass
class _Work:
    budget: SearchBudget
    expanded: int = 0
    edges: int = 0
    steps: int = 0


class _BudgetStop(Exception):
    pass


def _rollout(
    geometry: PlatformGeometry,
    capabilities: MovementCapabilities,
    start: MotionState,
    kind: str,
    first: SimAction,
    follow: SimAction,
    work: _Work,
) -> PlatformEdge | None:
    state = start
    segments: list[MovementPrimitive] = []
    digest = _trace_hash(start)
    ticks = 0

    def step(action: SimAction) -> bool:
        nonlocal state, ticks
        if ticks >= work.budget.max_primitive_steps:
            raise _BudgetStop("primitive_steps")
        if work.steps >= work.budget.max_simulated_steps:
            raise _BudgetStop("simulated_steps")
        state, accepted = _physical_step(geometry, capabilities, state, action)
        ticks += 1
        work.steps += 1
        _trace_add(digest, action, state, accepted)
        if segments and segments[-1].action == action:
            segments[-1] = MovementPrimitive(action, segments[-1].ticks + 1)
        else:
            segments.append(MovementPrimitive(action, 1))
        return accepted

    if not step(first):
        return None
    if kind == "walk" and not state.grounded:
        if not capabilities.can_drop:
            return None
        kind = "drop"
    while not state.grounded or state.climbing:
        if not step(follow):
            return None
    if state.vx != 0:
        if not step(SimAction()):
            return None
    if state == start:
        return None
    return PlatformEdge(
        kind,
        start,
        state,
        tuple(segments),
        _support(geometry, start),
        _support(geometry, state),
        geometry.content_sha256,
        capabilities.content_sha256,
        digest.hexdigest(),
    )


def _controls(
    capabilities: MovementCapabilities, start: MotionState
) -> tuple[tuple[str, SimAction, SimAction], ...]:
    candidates: list[tuple[str, SimAction, SimAction]] = []
    if start.grounded:
        if start.vx != 0:
            candidates.append(("settle", SimAction(), SimAction()))
        else:
            if capabilities.can_walk:
                candidates.extend(
                    ("walk", SimAction("move", d), SimAction("move", d)) for d in (-1, 1)
                )
            if capabilities.can_jump:
                candidates.extend(("jump", SimAction("jump", d), SimAction()) for d in (-1, 0, 1))
    elif not start.climbing:
        candidates.append(("settle", SimAction(), SimAction()))
        if capabilities.can_walk:
            candidates.extend(
                ("settle", SimAction("move", d), SimAction("move", d)) for d in (-1, 1)
            )
    if capabilities.can_climb:
        candidates.extend(("ladder", SimAction("climb", d), SimAction("climb", d)) for d in (-1, 1))
    return tuple(candidates)


def _expand(
    geometry: PlatformGeometry, capabilities: MovementCapabilities, start: MotionState, work: _Work
) -> EdgeExpansion:
    edges: list[PlatformEdge] = []
    incomplete: set[str] = set()
    before = work.steps
    for kind, first, follow in _controls(capabilities, start):
        try:
            edge = _rollout(geometry, capabilities, start, kind, first, follow, work)
        except _BudgetStop as exc:
            incomplete.add(str(exc))
            if str(exc) == "simulated_steps":
                break
            continue
        if edge is not None:
            if work.edges >= work.budget.max_generated_edges:
                incomplete.add("generated_edges")
                break
            work.edges += 1
            edges.append(edge)
    return EdgeExpansion(tuple(edges), work.steps - before, tuple(sorted(incomplete)))


@dataclass(frozen=True)
class PlatformGraph:
    geometry: PlatformGeometry
    capabilities: MovementCapabilities
    budget: SearchBudget = SearchBudget()

    def __post_init__(self) -> None:
        if (
            type(self.geometry) is not PlatformGeometry
            or type(self.capabilities) is not MovementCapabilities
            or type(self.budget) is not SearchBudget
        ):
            raise ValueError("platform graph requires static geometry, capabilities and a budget")

    @property
    def content_sha256(self) -> str:
        return _hash(
            {
                "schema_version": 1,
                "kind": "platform-graph",
                "primitive_set": PRIMITIVE_SET,
                "geometry_sha256": self.geometry.content_sha256,
                "capabilities_sha256": self.capabilities.content_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "platform-graph",
            "primitive_set": PRIMITIVE_SET,
            "geometry": self.geometry.manifest(),
            "capabilities": self.capabilities.to_dict(),
            "budget": self.budget.to_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlatformGraph:
        raw = _object(
            value,
            {
                "schema_version",
                "kind",
                "primitive_set",
                "geometry",
                "capabilities",
                "budget",
                "content_sha256",
            },
            "platform graph",
        )
        _schema(raw, "platform-graph")
        if raw["primitive_set"] != PRIMITIVE_SET:
            raise ValueError("unsupported navigation primitive set")
        graph = cls(
            PlatformGeometry.from_manifest(raw["geometry"]),
            MovementCapabilities.from_dict(raw["capabilities"]),
            SearchBudget.from_dict(raw["budget"]),
        )
        if graph.content_sha256 != raw["content_sha256"]:
            raise ValueError("platform graph content hash mismatch")
        return graph

    def edges_from(
        self, start: MotionState, *, budget: SearchBudget | None = None
    ) -> EdgeExpansion:
        _validate_state(self.geometry, start)
        selected = self.budget if budget is None else budget
        if type(selected) is not SearchBudget:
            raise ValueError("edge expansion requires a SearchBudget")
        return _expand(self.geometry, self.capabilities, start, _Work(selected))

    def find_path(
        self, start: MotionState, target_platform: str, *, budget: SearchBudget | None = None
    ) -> NavigationResult:
        _validate_state(self.geometry, start)
        if target_platform not in self.geometry.platform_ids:
            raise ValueError("navigation target is not an authored platform")
        selected = self.budget if budget is None else budget
        if type(selected) is not SearchBudget:
            raise ValueError("path search requires a SearchBudget")
        work = _Work(selected)
        pending: list[tuple[int, int, MotionState]] = [(0, 0, start)]
        best: dict[MotionState, int] = {start: 0}
        parent: dict[MotionState, PlatformEdge] = {}
        reached = set(_support(self.geometry, start))
        incomplete: set[str] = set()
        serial = 0

        def result(
            status: str, reason: str, path: NavigationPath | None = None
        ) -> NavigationResult:
            return NavigationResult(
                status,
                reason,
                path,
                work.expanded,
                work.edges,
                work.steps,
                tuple(sorted(reached)),
                tuple(sorted(incomplete)),
                status == "found" and not incomplete,
                selected,
            )

        while pending:
            cost, _, state = heapq.heappop(pending)
            if best[state] != cost:
                continue
            if target_platform in _support(self.geometry, state):
                edges = []
                current = state
                while current != start:
                    edge = parent[current]
                    edges.append(edge)
                    current = edge.start
                path = NavigationPath(
                    self.content_sha256,
                    self.geometry.map_id,
                    start,
                    target_platform,
                    tuple(reversed(edges)),
                )
                return result("found", "executable_fixed_step_witness", path)
            if work.expanded >= selected.max_expanded_states:
                incomplete.add("expanded_states")
                return result("unknown_budget", "expanded_states")
            work.expanded += 1
            expansion = _expand(self.geometry, self.capabilities, state, work)
            incomplete.update(expansion.incomplete)
            if "simulated_steps" in incomplete or "generated_edges" in incomplete:
                return result(
                    "unknown_budget",
                    "simulated_steps" if "simulated_steps" in incomplete else "generated_edges",
                )
            for edge in expansion.edges:
                reached.update(edge.target_platforms)
                distance = cost + edge.steps
                if distance > selected.max_path_steps:
                    incomplete.add("path_steps")
                    continue
                if edge.end not in best or distance < best[edge.end]:
                    best[edge.end] = distance
                    parent[edge.end] = edge
                    serial += 1
                    heapq.heappush(pending, (distance, serial, edge.end))
        if incomplete:
            return result("unknown_budget", "incomplete_primitive_or_path_budget")
        return result(
            "unreachable_in_primitive_graph", "exhausted_reachable_constant_control_states"
        )

    def verify_path(
        self, path: NavigationPath, *, initial_state: MotionState | None = None
    ) -> dict[str, Any]:
        if (
            type(path) is not NavigationPath
            or path.graph_sha256 != self.content_sha256
            or path.map_id != self.geometry.map_id
        ):
            raise ValueError("path is not bound to this geometry and capability graph")
        if initial_state is not None and initial_state != path.initial_state:
            raise ValueError("current motion state does not meet path precondition")
        _validate_state(self.geometry, path.initial_state)
        if path.steps > self.budget.max_path_steps:
            raise ValueError("path exceeds replay step budget")
        work = _Work(self.budget)
        for edge in path.edges:
            if (
                edge.geometry_sha256 != self.geometry.content_sha256
                or edge.capabilities_sha256 != self.capabilities.content_sha256
            ):
                raise ValueError("edge static evidence binding mismatch")
            if edge.steps > self.budget.max_primitive_steps:
                raise ValueError("edge exceeds primitive replay budget")
            first_action = edge.primitives[0].action
            control = next(
                (
                    (kind, first, follow)
                    for kind, first, follow in _controls(self.capabilities, edge.start)
                    if first == first_action
                    and kind == ("walk" if edge.kind == "drop" else edge.kind)
                ),
                None,
            )
            if control is None:
                raise ValueError("edge is not an available navigation primitive")
            try:
                reproduced = _rollout(self.geometry, self.capabilities, edge.start, *control, work)
            except _BudgetStop as exc:
                raise ValueError("path replay exceeds motion budget") from exc
            if reproduced != edge:
                raise ValueError("edge primitive, motion or support witness does not reproduce")
        if path.target_platform not in _support(self.geometry, path.end):
            raise ValueError("navigation path does not reach its declared platform")
        return {
            "verified": True,
            "map_id": self.geometry.map_id,
            "graph_sha256": self.content_sha256,
            "path_sha256": path.content_sha256,
            "steps": path.steps,
            "duration_ms": path.steps * self.capabilities.step_ms,
            "target_platform": path.target_platform,
            "verification": "all_primitive_motion_and_supports",
            "dynamic_obstacles_evaluated": False,
        }


def build_platform_graph(
    geometry: PlatformGeometry,
    capabilities: MovementCapabilities,
    budget: SearchBudget | None = None,
) -> PlatformGraph:
    return PlatformGraph(geometry, capabilities, SearchBudget() if budget is None else budget)
