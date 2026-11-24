"""Bounded, immutable authored map data for the independent fixture simulator.

The v1 pack format deliberately fixes the existing 320x192 world and ground
physics. It describes data, never executable scripts or external asset files.
Provenance fields are the author's assertions; the content hash establishes
integrity, not ownership, authenticity, or a guarantee of mission solvability.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .catalog import canonical_json, strict_json, validate_provenance
from .sim_types import NPC, Ladder, Platform, Portal

MAP_WIDTH = 320
MAP_HEIGHT = 192
MAP_GROUND = 168
MAX_MAP_PACK_BYTES = 128 * 1024
_ID = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Kept local to avoid a renderer -> map data -> renderer import cycle.
_RESERVED_COLORS = frozenset(
    {
        (59, 221, 255),
        (238, 82, 113),
        (248, 207, 65),
        (183, 108, 255),
        (71, 204, 126),
        (122, 90, 60),
        (213, 153, 82),
        (35, 163, 95),
        (108, 83, 169),
        (17, 23, 36),
        (35, 42, 60),
        (239, 68, 68),
        (80, 140, 255),
        (219, 229, 242),
    }
)


def _integer(value: object, field: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase identifier of at most 40 characters")
    return value


def _object(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} has invalid fields")
    return value


def _array(value: object, field: str, low: int, high: int) -> list[Any]:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise ValueError(f"{field} must contain {low}..{high} entries")
    return value


def _immutable_entries(value: object, cls: type, field: str, low: int, high: int) -> None:
    if (
        not isinstance(value, tuple)
        or not low <= len(value) <= high
        or any(type(item) is not cls for item in value)
    ):
        raise ValueError(f"{field} needs {low}..{high} immutable {cls.__name__} entries")


@dataclass(frozen=True)
class MapDefinition:
    id: str
    theme: str
    marker: tuple[int, int, int]
    spawn_x: int
    platforms: tuple[Platform, ...]
    portals: tuple[Portal, ...]
    npcs: tuple[NPC, ...]
    ladders: tuple[Ladder, ...]
    monster_x: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "map id")
        if self.id in {"meadow", "cavern"}:
            raise ValueError("external map ids cannot alias built-in maps")
        if not isinstance(self.theme, str) or self.theme not in {"meadow", "cavern"}:
            raise ValueError("map theme must be meadow or cavern")
        if not isinstance(self.marker, tuple) or len(self.marker) != 3:
            raise ValueError("map marker must be an immutable RGB tuple")
        for channel in self.marker:
            _integer(channel, "marker channel", 0, 255)
        if self.marker in _RESERVED_COLORS:
            raise ValueError("map marker conflicts with a reserved fixture color")
        _integer(self.spawn_x, "spawn_x", 0, MAP_WIDTH - 14)
        _immutable_entries(self.platforms, Platform, "platforms", 2, 8)
        for platform in self.platforms:
            _integer(platform.x1, "platform x1", 0, MAP_WIDTH - 1)
            _integer(platform.x2, "platform x2", 1, MAP_WIDTH)
            _integer(platform.y, "platform y", 72, MAP_GROUND)
            if platform.x2 - platform.x1 < 16:
                raise ValueError("platform width must be at least 16 pixels")
        if self.platforms[0] != Platform(0, MAP_WIDTH, MAP_GROUND):
            raise ValueError("first platform must be the full-width ground")
        if any(platform.y == MAP_GROUND for platform in self.platforms[1:]):
            raise ValueError("only the first platform may be ground")
        for index, platform in enumerate(self.platforms):
            for other in self.platforms[index + 1 :]:
                if platform.y == other.y and max(platform.x1, other.x1) <= min(
                    platform.x2, other.x2
                ):
                    raise ValueError("platforms on the same level must not overlap or touch")
        _immutable_entries(self.portals, Portal, "portals", 1, 1)
        portal = self.portals[0]
        _identifier(portal.id, "portal id")
        _identifier(portal.destination, "portal destination")
        _integer(portal.x, "portal x", 0, MAP_WIDTH - 20)
        for field, actual, required in (
            ("y", portal.y, 132),
            ("w", portal.w, 20),
            ("h", portal.h, 36),
        ):
            _integer(actual, f"portal {field}", required, required)
        _immutable_entries(self.npcs, NPC, "npcs", 1, 1)
        npc = self.npcs[0]
        _identifier(npc.id, "npc id")
        _integer(npc.x, "npc x", 0, MAP_WIDTH - 14)
        for field, actual, required in (("y", npc.y, 144), ("w", npc.w, 14), ("h", npc.h, 24)):
            _integer(actual, f"npc {field}", required, required)
        if portal.id == npc.id:
            raise ValueError("portal and NPC ids must be distinct within a map")
        _immutable_entries(self.ladders, Ladder, "ladders", 1, 4)
        for ladder in self.ladders:
            # Climbing centers a 14px player on a 10px ladder. The two-pixel
            # margin keeps that player inside the simulator's saved-state bounds.
            _integer(ladder.x, "ladder x", 2, MAP_WIDTH - 12)
            _integer(ladder.y, "ladder y", 72, MAP_GROUND - 1)
            _integer(ladder.w, "ladder w", 10, 10)
            _integer(ladder.h, "ladder h", 1, MAP_GROUND - 72)
            if ladder.y + ladder.h != MAP_GROUND or not any(
                platform.y == ladder.y
                and platform.x1 <= ladder.x
                and ladder.x + ladder.w <= platform.x2
                for platform in self.platforms[1:]
            ):
                raise ValueError("ladder must connect ground to a supporting platform")
        if len(set(self.ladders)) != len(self.ladders):
            raise ValueError("map contains duplicate ladders")
        _immutable_entries(self.monster_x, int, "monster_x", 1, 6)
        for x in self.monster_x:
            # Existing deterministic +/-3 spawn jitter must keep an 18px body in frame.
            _integer(x, "monster_x", 3, MAP_WIDTH - 18 - 3)
        if any(
            abs(left - right) < 24
            for index, left in enumerate(self.monster_x)
            for right in self.monster_x[index + 1 :]
        ):
            raise ValueError("monster spawns must be at least 24 pixels apart")

    @property
    def spawn_y(self) -> int:
        return MAP_GROUND - 22

    @property
    def width(self) -> int:
        return MAP_WIDTH

    @property
    def height(self) -> int:
        return MAP_HEIGHT

    @property
    def ground(self) -> int:
        return MAP_GROUND

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "theme": self.theme,
            "marker": list(self.marker),
            "spawn_x": self.spawn_x,
            "platforms": [asdict(item) for item in self.platforms],
            "portals": [asdict(item) for item in self.portals],
            "npcs": [asdict(item) for item in self.npcs],
            "ladders": [asdict(item) for item in self.ladders],
            "monster_x": list(self.monster_x),
        }

    @classmethod
    def from_dict(cls, value: object) -> "MapDefinition":
        raw = _object(
            value,
            {
                "id",
                "theme",
                "marker",
                "spawn_x",
                "platforms",
                "portals",
                "npcs",
                "ladders",
                "monster_x",
            },
            "map",
        )
        marker = _array(raw["marker"], "marker", 3, 3)
        return cls(
            id=raw["id"],
            theme=raw["theme"],
            marker=(marker[0], marker[1], marker[2]),
            spawn_x=raw["spawn_x"],
            platforms=tuple(
                Platform(**_object(item, {"x1", "x2", "y"}, "platform"))
                for item in _array(raw["platforms"], "platforms", 2, 8)
            ),
            portals=tuple(
                Portal(**_object(item, {"id", "x", "y", "w", "h", "destination"}, "portal"))
                for item in _array(raw["portals"], "portals", 1, 1)
            ),
            npcs=tuple(
                NPC(**_object(item, {"id", "x", "y", "w", "h"}, "npc"))
                for item in _array(raw["npcs"], "npcs", 1, 1)
            ),
            ladders=tuple(
                Ladder(**_object(item, {"x", "y", "w", "h"}, "ladder"))
                for item in _array(raw["ladders"], "ladders", 1, 4)
            ),
            monster_x=tuple(_array(raw["monster_x"], "monster_x", 1, 6)),
        )


@dataclass(frozen=True)
class MapPack:
    id: str
    version: str
    source_uri: str
    permission: str
    license: str
    maps: tuple[MapDefinition, ...]
    route: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "pack id")
        if (
            not isinstance(self.version, str)
            or len(self.version) > 40
            or not _VERSION.fullmatch(self.version)
        ):
            raise ValueError("map pack version must be a bounded numeric major.minor.patch")
        validate_provenance(self.source_uri, self.permission, self.license)
        _immutable_entries(self.maps, MapDefinition, "maps", 2, 8)
        _immutable_entries(self.route, str, "route", 2, 8)
        for map_id in self.route:
            _identifier(map_id, "route map id")
        if len(set(self.route)) != len(self.route):
            raise ValueError("map route contains duplicate map ids")
        by_id = {item.id: item for item in self.maps}
        if len(by_id) != len(self.maps) or set(by_id) != set(self.route):
            raise ValueError("route must contain every unique map exactly once")
        # Storage order has no gameplay meaning. Normalize once so immutable
        # equality, embedded restore, and content identity use the same ordering.
        object.__setattr__(self, "maps", tuple(by_id[map_id] for map_id in self.route))
        if len({item.marker for item in self.maps}) != len(self.maps):
            raise ValueError("map markers must be unique")
        for index, map_id in enumerate(self.route):
            expected = self.route[(index + 1) % len(self.route)]
            if by_id[map_id].portals[0].destination != expected:
                raise ValueError("portal destinations must follow the complete cyclic route")
        try:
            size = len(canonical_json(self.to_dict()).encode("utf-8"))
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("map pack contains invalid text") from exc
        if size > MAX_MAP_PACK_BYTES:
            raise ValueError("map pack exceeds 128 KiB")

    @property
    def map_ids(self) -> tuple[str, ...]:
        return self.route

    @property
    def entry_map(self) -> str:
        return self.route[0]

    @property
    def markers(self) -> Mapping[str, tuple[int, int, int]]:
        return MappingProxyType({item.id: item.marker for item in self.maps})

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def width(self) -> int:
        return MAP_WIDTH

    @property
    def height(self) -> int:
        return MAP_HEIGHT

    @property
    def ground(self) -> int:
        return MAP_GROUND

    def get(self, map_id: str) -> MapDefinition:
        for item in self.maps:
            if item.id == map_id:
                return item
        raise ValueError(f"unknown map in pack: {map_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "id": self.id,
            "version": self.version,
            "scope": "synthetic_fixture",
            "source_uri": self.source_uri,
            "permission": self.permission,
            "license": self.license,
            "width": MAP_WIDTH,
            "height": MAP_HEIGHT,
            "ground": MAP_GROUND,
            "route": list(self.route),
            "maps": [self.get(map_id).to_dict() for map_id in self.route],
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "content": self.to_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MapPack":
        raw = _object(
            value,
            {
                "schema_version",
                "id",
                "version",
                "scope",
                "source_uri",
                "permission",
                "license",
                "width",
                "height",
                "ground",
                "route",
                "maps",
            },
            "map pack",
        )
        _integer(raw["schema_version"], "map pack schema_version", 1, 1)
        for field, required in (
            ("width", MAP_WIDTH),
            ("height", MAP_HEIGHT),
            ("ground", MAP_GROUND),
        ):
            _integer(raw[field], field, required, required)
        if raw["scope"] != "synthetic_fixture":
            raise ValueError("map pack scope must be synthetic_fixture")
        return cls(
            id=raw["id"],
            version=raw["version"],
            source_uri=raw["source_uri"],
            permission=raw["permission"],
            license=raw["license"],
            maps=tuple(MapDefinition.from_dict(item) for item in _array(raw["maps"], "maps", 2, 8)),
            route=tuple(_array(raw["route"], "route", 2, 8)),
        )

    @classmethod
    def from_manifest(cls, value: object) -> "MapPack":
        raw = _object(value, {"schema_version", "content", "content_sha256"}, "map pack manifest")
        _integer(raw["schema_version"], "map pack manifest schema_version", 1, 1)
        digest = raw["content_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("invalid map pack content SHA256")
        result = cls.from_dict(raw["content"])
        if result.content_sha256 != digest:
            raise ValueError("map pack content SHA256 mismatch")
        return result


def load_map_pack(path: Path) -> MapPack:
    """Read one local bounded JSON file; no URI in its metadata is ever fetched."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("map pack source must be a regular file, not a symlink")
    with path.open("rb") as stream:
        data = stream.read(MAX_MAP_PACK_BYTES + 1)
    if len(data) > MAX_MAP_PACK_BYTES:
        raise ValueError("map pack exceeds 128 KiB")
    try:
        raw = strict_json(data.decode("utf-8"))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("map pack source is not bounded UTF-8 JSON") from exc
    return MapPack.from_dict(raw)
