"""Seeded, fixed-step synthetic fixture simulator; it cannot control another process.

Coordinates are sprite upper-left in pixels. Platforms are one-way floor segments.
Rules are deliberately independent fixtures and make no official-game accuracy claim.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from .map_pack import MapPack
from .profiles import class_spec_sha256, parse_class_spec
from .sim_types import (
    CLASS_SPECS,
    NPC,
    RULESET,
    SIM_VERSION,
    ClassSpec,
    DropState,
    Ladder,
    MonsterState,
    Platform,
    PlayerState,
    Portal,
    SimAction,
    SimSnapshot,
    StepOutcome,
)

WIDTH = 320
HEIGHT = 192
GROUND = 168
PLAYER_W = 14
PLAYER_H = 22
GRAVITY = 650
MAP_IDS = ("meadow", "cavern")
_MAX_SEQUENCE = 1_000_000_000
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class MapSpec:
    id: str
    platforms: tuple[Platform, ...]
    portals: tuple[Portal, ...]
    npcs: tuple[NPC, ...]
    ladders: tuple[Ladder, ...]


MAP_SPECS = {
    "meadow": MapSpec(
        "meadow",
        (Platform(0, WIDTH, GROUND), Platform(76, 136, 126), Platform(180, 238, 104)),
        (Portal("to_cavern", 292, 132, 20, 36, "cavern"),),
        (NPC("guide", 42, 144, 14, 24),),
        (Ladder(99, 126, 10, 42),),
    ),
    "cavern": MapSpec(
        "cavern",
        (Platform(0, WIDTH, GROUND), Platform(86, 152, 118), Platform(212, 270, 94)),
        (Portal("to_meadow", 8, 132, 20, 36, "meadow"),),
        (NPC("keeper", 274, 144, 14, 24),),
        (Ladder(115, 118, 10, 50),),
    ),
}


@dataclass
class _Player:
    x: float = 24.0
    y: float = float(GROUND - PLAYER_H)
    vx: float = 0.0
    vy: float = 0.0
    hp: int = 100
    mp: int = 60
    ammo: int = 200
    hp_potions: int = 8
    mp_potions: int = 8
    coins: int = 0
    grounded: bool = True
    facing: int = 1
    buff_until_ms: int = 0
    attack_ready_ms: int = 0
    potion_ready_ms: int = 0
    skill_ready_ms: int = 0
    invulnerable_until_ms: int = 0
    climbing: bool = False


@dataclass
class _Monster:
    id: str
    x: float
    hp: int = 36
    max_hp: int = 36


@dataclass
class _Drop:
    id: str
    x: float
    y: float
    item: str
    quantity: int


@dataclass
class _MapState:
    monsters: list[_Monster]
    drops: list[_Drop]


def _integer(value: object, name: str, low: int = 0, high: int = _MAX_SEQUENCE) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} is outside its finite range")
    return float(value)


def _mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has invalid fields")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 120:
        raise ValueError(f"{name} must be a nonempty short string")
    return value


def _overlap(
    ax: float, ay: float, aw: float, ah: float, bx: float, by: float, bw: float, bh: float
) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class Simulator:
    """A small local fixture environment. Snapshot access is for rendering/validation.

    A policy must receive only rendered observations, never this object or snapshots.
    `step` always advances one timestep, including rejected actions and death.
    """

    def __init__(
        self,
        seed: int = 0,
        job: str = "assassin",
        map_id: str = "meadow",
        step_ms: int = 50,
        *,
        class_spec: ClassSpec | None = None,
        map_pack: MapPack | None = None,
    ) -> None:
        _integer(seed, "seed", high=(1 << 63) - 1)
        _integer(step_ms, "step_ms", 10, 100)
        if not isinstance(job, str) or job not in CLASS_SPECS:
            raise ValueError("unknown fixture class")
        if map_pack is not None and not isinstance(map_pack, MapPack):
            raise ValueError("map_pack must be a verified MapPack")
        self._map_pack = map_pack
        self._map_ids = MAP_IDS if map_pack is None else map_pack.map_ids
        self._map_specs = (
            MAP_SPECS.copy()
            if map_pack is None
            else {
                definition.id: MapSpec(
                    definition.id,
                    definition.platforms,
                    definition.portals,
                    definition.npcs,
                    definition.ladders,
                )
                for definition in map_pack.maps
            }
        )
        if not isinstance(map_id, str) or map_id not in self._map_specs:
            raise ValueError("unknown fixture map")
        if class_spec is not None and (
            not isinstance(class_spec, ClassSpec) or class_spec.id != job
        ):
            raise ValueError("injected ClassSpec must match the simulator job")
        self.seed = seed
        self.job = job
        self.step_ms = step_ms
        self._rng_state = (seed + 0x9E3779B97F4A7C15) & _MASK64 or 1
        self._sequence = 0
        self._map_id = map_id
        self._initial_map_id = map_id
        self._profile_injected = class_spec is not None
        self._class_spec = CLASS_SPECS[job] if class_spec is None else class_spec
        spec = self.class_spec
        self._player = _Player(hp=spec.max_hp, mp=spec.max_mp, ammo=200 if spec.ammo_cost else 0)
        if map_pack is not None:
            self._player.x = float(map_pack.get(map_id).spawn_x)
        self._maps: dict[str, _MapState] = {}
        self._next_drop_id = 0
        self._events: tuple[str, ...] = ()
        self._dialogue: str | None = None
        for key in self._map_ids:
            monster_x = (126, 210, 258) if map_pack is None else map_pack.get(key).monster_x
            self._maps[key] = _MapState(
                [
                    _Monster(f"{key}-monster-{i}", float(x + self._roll(7) - 3))
                    for i, x in enumerate(monster_x)
                ],
                [],
            )

    @property
    def timestamp_ms(self) -> int:
        return self._sequence * self.step_ms

    @property
    def class_spec(self) -> ClassSpec:
        """Immutable static rules; contains no dynamic environment state."""
        return self._class_spec

    @property
    def map_pack(self) -> MapPack | None:
        """Immutable authored map data, used by renderers and replay validators."""
        return self._map_pack

    @property
    def map_ids(self) -> tuple[str, ...]:
        """The static mission route; contains no dynamic environment state."""
        return self._map_ids

    def reset(self, seed: int | None = None) -> None:
        Simulator.__init__(
            self,
            self.seed if seed is None else seed,
            self.job,
            self._initial_map_id,
            self.step_ms,
            class_spec=self.class_spec if self._profile_injected else None,
            map_pack=self._map_pack,
        )

    def _roll(self, bound: int) -> int:
        """Explicit versioned xorshift64* state is portable in the save format."""
        x = self._rng_state
        x ^= x >> 12
        x ^= (x << 25) & _MASK64
        x ^= x >> 27
        self._rng_state = x & _MASK64
        return ((x * 0x2545F4914F6CDD1D) & _MASK64) % bound

    def snapshot(self) -> SimSnapshot:
        p = self._player
        cls = self.class_spec
        terrain = self._map_specs[self._map_id]
        world = self._maps[self._map_id]
        player = PlayerState(
            p.x,
            p.y,
            PLAYER_W,
            PLAYER_H,
            p.vx,
            p.vy,
            p.hp,
            cls.max_hp,
            p.mp,
            cls.max_mp,
            p.ammo,
            p.hp_potions,
            p.mp_potions,
            p.coins,
            ("power",) if self.timestamp_ms < p.buff_until_ms else (),
            p.grounded,
            p.facing,
            self.job,
            p.hp > 0,
        )
        return SimSnapshot(
            self._sequence,
            self.timestamp_ms,
            self._map_id,
            WIDTH,
            HEIGHT,
            player,
            tuple(
                MonsterState(m.id, m.x, 152, 18, 16, m.hp, m.max_hp)
                for m in world.monsters
                if m.hp > 0
            ),
            tuple(DropState(d.id, d.x, d.y, 8, 8, d.item, d.quantity) for d in world.drops),
            terrain.platforms,
            terrain.portals,
            terrain.npcs,
            terrain.ladders,
            self._events,
            self._dialogue,
        )

    def step(self, action: SimAction | None = None) -> StepOutcome:
        if action is None:
            action = SimAction()
        if not isinstance(action, SimAction):
            raise ValueError("action must be SimAction")
        if self._sequence >= _MAX_SEQUENCE:
            raise ValueError("simulation sequence limit reached")
        events: list[str] = []
        p = self._player
        if p.hp <= 0:
            accepted, reason = False, "player_dead"
        else:
            accepted, reason = self._apply_action(action, events)
            self._physics()
            self._contact_damage(events)
        self._sequence += 1
        if p.buff_until_ms and self.timestamp_ms >= p.buff_until_ms:
            p.buff_until_ms = 0
            events.append("buff_expired")
        self._events = tuple(events)
        return StepOutcome(self.snapshot(), accepted, reason, self._events)

    def _apply_action(self, action: SimAction, events: list[str]) -> tuple[bool, str]:
        p = self._player
        cls = self.class_spec
        kind = action.kind
        if kind not in {"interact", "idle"}:
            self._dialogue = None
        if action.direction and kind != "climb":
            p.facing = action.direction
        if p.grounded:
            p.vx = 0
        if kind not in {"climb", "idle"}:
            p.climbing = False
        if kind == "idle":
            return True, "idle"
        if kind == "move":
            p.vx = float(cls.walk_speed * action.direction)
            return True, "moving"
        if kind in {"jump", "jump_attack"}:
            if not p.grounded:
                return False, "airborne"
            if kind == "jump_attack":
                error = self._attack_error(action.skill)
                if error:
                    return False, error
                if not cls.can_jump_attack:
                    return False, "jump_attack_unavailable"
            p.vy = float(-cls.jump_speed)
            p.vx = float(cls.walk_speed * action.direction)
            p.grounded = False
            events.append("jump")
            if kind == "jump_attack":
                self._attack(action.skill, events)
            return True, kind
        if kind == "attack":
            error = self._attack_error(action.skill)
            if error:
                return False, error
            self._attack(action.skill, events)
            return True, "attack"
        if kind == "loot":
            found = []
            for drop in self._maps[self._map_id].drops:
                if abs(drop.x + 4 - (p.x + PLAYER_W / 2)) <= 25 and abs(drop.y - p.y) <= 30:
                    found.append(drop)
            if not found:
                return False, "no_drop_in_range"
            for drop in found:
                if drop.item == "coin":
                    p.coins += drop.quantity
                elif drop.item == "hp_potion":
                    p.hp_potions += drop.quantity
                elif drop.item == "mp_potion":
                    p.mp_potions += drop.quantity
                elif drop.item == "ammo":
                    p.ammo += drop.quantity
                self._maps[self._map_id].drops.remove(drop)
            events.append("loot")
            return True, "loot"
        if kind in {"potion_hp", "potion_mp"}:
            return self._potion(kind, events)
        if kind in {"buff", "heal"}:
            if self.timestamp_ms < p.skill_ready_ms:
                return False, "skill_cooldown"
            if kind == "heal" and not cls.heal_amount:
                return False, "heal_unavailable"
            if kind == "heal" and p.hp >= cls.max_hp:
                return False, "hp_full"
            cost = 8 if kind == "buff" else 12
            if p.mp < cost:
                return False, "insufficient_mp"
            if kind == "buff" and self.timestamp_ms < p.buff_until_ms:
                return False, "buff_already_active"
            p.mp -= cost
            p.skill_ready_ms = self.timestamp_ms + 500
            if kind == "buff":
                p.buff_until_ms = self.timestamp_ms + 5000
            else:
                p.hp = min(cls.max_hp, p.hp + cls.heal_amount)
            events.append(kind)
            return True, kind
        if kind == "interact":
            for npc in self._map_specs[self._map_id].npcs:
                if action.target and action.target != npc.id:
                    continue
                vertical_overlap = self._map_pack is None or (
                    p.y < npc.y + npc.h and p.y + PLAYER_H > npc.y
                )
                if abs(npc.x - p.x) <= 32 and abs(npc.y - p.y) <= 28 and vertical_overlap:
                    self._dialogue = (
                        f"{npc.id}: Independent fixture. Explore, practice, and return."
                    )
                    events.append("dialogue")
                    return True, "dialogue"
            return False, "no_npc_in_range"
        if kind == "portal":
            for portal in self._map_specs[self._map_id].portals:
                if action.target and action.target != portal.id:
                    continue
                vertical_overlap = self._map_pack is None or (
                    p.y < portal.y + portal.h and p.y + PLAYER_H > portal.y
                )
                if (
                    abs(portal.x + portal.w / 2 - (p.x + PLAYER_W / 2)) <= 27
                    and p.grounded
                    and vertical_overlap
                ):
                    self._map_id = portal.destination
                    p.x = (
                        (35.0 if portal.destination == "cavern" else 270.0)
                        if self._map_pack is None
                        else float(self._map_pack.get(portal.destination).spawn_x)
                    )
                    p.y = float(GROUND - PLAYER_H)
                    p.vx = p.vy = 0.0
                    p.grounded = True
                    events.append("map_changed")
                    return True, "map_changed"
            return False, "no_portal_in_range"
        if kind == "climb":
            for ladder in self._map_specs[self._map_id].ladders:
                if abs(ladder.x + ladder.w / 2 - (p.x + PLAYER_W / 2)) <= 12 and (
                    p.y + PLAYER_H >= ladder.y and p.y <= ladder.y + ladder.h
                ):
                    p.climbing = True
                    p.grounded = False
                    p.vx = p.vy = 0
                    p.x = ladder.x + ladder.w / 2 - PLAYER_W / 2
                    p.y = max(
                        ladder.y - PLAYER_H,
                        min(
                            ladder.y + ladder.h - PLAYER_H,
                            p.y + action.direction * 60 * self.step_ms / 1000,
                        ),
                    )
                    if p.y + PLAYER_H <= ladder.y:
                        p.climbing = False
                        p.grounded = True
                    elif p.y + PLAYER_H >= GROUND:
                        p.climbing = False
                        p.grounded = True
                    events.append("climb")
                    return True, "climb"
            return False, "no_ladder_in_range"
        raise AssertionError("SimAction validation failed")

    def _attack_error(self, skill: str) -> str | None:
        p = self._player
        cls = self.class_spec
        if skill == "area" and self.job != "dragon_knight":
            return "skill_unavailable"
        if self.timestamp_ms < p.attack_ready_ms:
            return "attack_cooldown"
        if p.ammo < cls.ammo_cost:
            return "insufficient_ammo"
        cost = 8 if skill == "area" else cls.attack_mp_cost
        if p.mp < cost:
            return "insufficient_mp"
        return None

    def _attack(self, skill: str, events: list[str]) -> None:
        p = self._player
        cls = self.class_spec
        p.mp -= 8 if skill == "area" else cls.attack_mp_cost
        p.ammo -= cls.ammo_cost
        p.attack_ready_ms = self.timestamp_ms + cls.attack_cooldown_ms
        damage = cls.attack_damage + (6 if self.timestamp_ms < p.buff_until_ms else 0)
        reach = 72 if skill == "area" else cls.attack_range
        events.append("attack")
        targets = []
        for monster in self._maps[self._map_id].monsters:
            delta = monster.x + 9 - (p.x + PLAYER_W / 2)
            if monster.hp > 0 and abs(delta) <= reach and abs(160 - (p.y + PLAYER_H / 2)) <= 38:
                if skill == "area" or delta * p.facing >= 0:
                    targets.append(monster)
        targets.sort(key=lambda m: abs(m.x - p.x))
        for monster in targets[: 3 if skill == "area" else 1]:
            monster.hp = max(0, monster.hp - damage)
            events.append("attack_hit")
            if monster.hp == 0:
                self._next_drop_id += 1
                self._maps[self._map_id].drops.append(
                    _Drop(
                        f"drop-{self._next_drop_id}", monster.x + 5, 160, "coin", 1 + self._roll(5)
                    )
                )
                events.append("monster_defeated")

    def _potion(self, kind: str, events: list[str]) -> tuple[bool, str]:
        p = self._player
        cls = self.class_spec
        if self.timestamp_ms < p.potion_ready_ms:
            return False, "potion_cooldown"
        if kind == "potion_hp":
            if p.hp_potions <= 0:
                return False, "no_hp_potions"
            if p.hp >= cls.max_hp:
                return False, "hp_full"
            p.hp_potions -= 1
            p.hp = min(cls.max_hp, p.hp + 45)
        else:
            if p.mp_potions <= 0:
                return False, "no_mp_potions"
            if p.mp >= cls.max_mp:
                return False, "mp_full"
            p.mp_potions -= 1
            p.mp = min(cls.max_mp, p.mp + 40)
        p.potion_ready_ms = self.timestamp_ms + 300
        events.append(kind)
        return True, kind

    def _physics(self) -> None:
        p = self._player
        if p.climbing:
            return
        dt = self.step_ms / 1000
        old_bottom = p.y + PLAYER_H
        p.x = max(0.0, min(float(WIDTH - PLAYER_W), p.x + p.vx * dt))
        p.vy += GRAVITY * dt
        new_y = p.y + p.vy * dt
        landing = []
        if p.vy >= 0:
            for platform in self._map_specs[self._map_id].platforms:
                if (
                    p.x + PLAYER_W > platform.x1
                    and p.x < platform.x2
                    and old_bottom <= platform.y + 1e-7
                    and new_y + PLAYER_H >= platform.y
                ):
                    landing.append(platform.y)
        if landing:
            p.y = min(landing) - PLAYER_H
            p.vy = 0.0
            p.grounded = True
        else:
            p.y = max(0.0, new_y)
            p.grounded = False
            if new_y < 0:
                p.vy = 0.0

    def _contact_damage(self, events: list[str]) -> None:
        p = self._player
        if self.timestamp_ms < p.invulnerable_until_ms:
            return
        for monster in self._maps[self._map_id].monsters:
            if monster.hp and _overlap(p.x, p.y, PLAYER_W, PLAYER_H, monster.x, 152, 18, 16):
                p.hp = max(0, p.hp - 9)
                p.invulnerable_until_ms = self.timestamp_ms + 500
                events.append("damage")
                if p.hp == 0:
                    p.vx = p.vy = 0.0
                    p.climbing = False
                    events.append("death")
                break

    def save(self) -> dict[str, Any]:
        """Independent JSON state copy, including RNG and embedded external maps."""
        result: dict[str, Any] = {
            "schema_version": (
                3 if self._map_pack is not None else (2 if self._profile_injected else SIM_VERSION)
            ),
            "ruleset": RULESET,
            "seed": self.seed,
            "job": self.job,
            "step_ms": self.step_ms,
            "sequence": self._sequence,
            "map_id": self._map_id,
            "initial_map_id": self._initial_map_id,
            "rng_state": self._rng_state,
            "next_drop_id": self._next_drop_id,
            "player": asdict(self._player),
            "maps": {key: asdict(value) for key, value in self._maps.items()},
            "events": list(self._events),
            "dialogue": self._dialogue,
        }
        if self._profile_injected:
            result["class_profile"] = {
                "spec": asdict(self.class_spec),
                "sha256": class_spec_sha256(self.class_spec),
            }
        if self._map_pack is not None:
            result["map_pack"] = self._map_pack.manifest()
        return result

    @classmethod
    def restore(cls, state: object) -> "Simulator":
        """Validate an untrusted JSON save before constructing a new environment."""
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] not in (SIM_VERSION, 2, 3)
        ):
            raise ValueError("unsupported simulator schema")
        expected_fields = {
            "schema_version",
            "ruleset",
            "seed",
            "job",
            "step_ms",
            "sequence",
            "map_id",
            "initial_map_id",
            "rng_state",
            "next_drop_id",
            "player",
            "maps",
            "events",
            "dialogue",
        }
        if state["schema_version"] == 2 or (
            state["schema_version"] == 3 and "class_profile" in state
        ):
            expected_fields.add("class_profile")
        if state["schema_version"] == 3:
            expected_fields.add("map_pack")
        root = _mapping(state, expected_fields, "save")
        if root["ruleset"] != RULESET:
            raise ValueError("unsupported fixture ruleset")
        injected_spec = None
        if "class_profile" in root:
            profile = _mapping(root["class_profile"], {"spec", "sha256"}, "class profile")
            injected_spec = parse_class_spec(profile["spec"])
            if class_spec_sha256(injected_spec) != profile["sha256"]:
                raise ValueError("embedded class profile SHA256 mismatch")
        map_pack = MapPack.from_manifest(root["map_pack"]) if root["schema_version"] == 3 else None
        sim = cls(
            root["seed"],
            root["job"],
            root["map_id"],
            root["step_ms"],
            class_spec=injected_spec,
            map_pack=map_pack,
        )
        initial_map = root["initial_map_id"]
        if not isinstance(initial_map, str) or initial_map not in sim._map_specs:
            raise ValueError("unknown initial map")
        sim._initial_map_id = initial_map
        sim._sequence = _integer(root["sequence"], "sequence")
        sim._rng_state = _integer(root["rng_state"], "rng_state", 1, _MASK64)
        sim._next_drop_id = _integer(root["next_drop_id"], "next_drop_id")
        p = _mapping(root["player"], set(_Player.__dataclass_fields__), "player")
        class_spec = sim.class_spec
        values: dict[str, Any] = {}
        for key, low, high in (
            ("x", 0, WIDTH - PLAYER_W),
            ("y", 0, GROUND - PLAYER_H),
            ("vx", -200, 200),
            ("vy", -400, 1000),
        ):
            values[key] = _number(p[key], key, low, high)
        for key, high in (
            ("hp", class_spec.max_hp),
            ("mp", class_spec.max_mp),
            ("ammo", 1_000_000),
            ("hp_potions", 1_000_000),
            ("mp_potions", 1_000_000),
            ("coins", 1_000_000_000),
        ):
            values[key] = _integer(p[key], key, 0, high)
        for key in (
            "buff_until_ms",
            "attack_ready_ms",
            "potion_ready_ms",
            "skill_ready_ms",
            "invulnerable_until_ms",
        ):
            values[key] = _integer(
                p[key], key, 0, _MAX_SEQUENCE * 100 + max(5000, class_spec.attack_cooldown_ms)
            )
        values["facing"] = _integer(p["facing"], "facing", -1, 1)
        if values["facing"] == 0:
            raise ValueError("facing must be -1 or 1")
        for key in ("grounded", "climbing"):
            if type(p[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
            values[key] = p[key]
        if values["grounded"] and values["climbing"]:
            raise ValueError("player cannot be grounded and climbing")
        sim._player = _Player(**values)
        maps = _mapping(root["maps"], set(sim._map_ids), "maps")
        all_ids: set[str] = set()
        for map_id, payload in maps.items():
            world = _mapping(payload, {"monsters", "drops"}, "map state")
            if not isinstance(world["monsters"], list) or len(world["monsters"]) > 100:
                raise ValueError("invalid monsters list")
            if not isinstance(world["drops"], list) or len(world["drops"]) > 1000:
                raise ValueError("invalid drops list")
            monsters: list[_Monster] = []
            drops: list[_Drop] = []
            for raw in world["monsters"]:
                row = _mapping(raw, {"id", "x", "hp", "max_hp"}, "monster")
                entity_id = _text(row["id"], "monster.id")
                if entity_id in all_ids:
                    raise ValueError("duplicate entity ID")
                all_ids.add(entity_id)
                maximum = _integer(row["max_hp"], "monster.max_hp", 1, 1_000_000)
                monsters.append(
                    _Monster(
                        entity_id,
                        _number(row["x"], "monster.x", 0, WIDTH - 18),
                        _integer(row["hp"], "monster.hp", 0, maximum),
                        maximum,
                    )
                )
            for raw in world["drops"]:
                row = _mapping(raw, {"id", "x", "y", "item", "quantity"}, "drop")
                entity_id = _text(row["id"], "drop.id")
                if entity_id in all_ids:
                    raise ValueError("duplicate entity ID")
                all_ids.add(entity_id)
                if not isinstance(row["item"], str) or row["item"] not in {
                    "coin",
                    "hp_potion",
                    "mp_potion",
                    "ammo",
                }:
                    raise ValueError("unknown drop item")
                drops.append(
                    _Drop(
                        entity_id,
                        _number(row["x"], "drop.x", 0, WIDTH - 8),
                        _number(row["y"], "drop.y", 0, HEIGHT - 8),
                        row["item"],
                        _integer(row["quantity"], "drop.quantity", 1, 1_000_000),
                    )
                )
            sim._maps[map_id] = _MapState(monsters, drops)
        raw_events = root["events"]
        if not isinstance(raw_events, list) or len(raw_events) > 100:
            raise ValueError("events must be a bounded list")
        sim._events = tuple(_text(event, "event") for event in raw_events)
        dialogue = root["dialogue"]
        if dialogue is not None and (not isinstance(dialogue, str) or len(dialogue) > 500):
            raise ValueError("invalid dialogue")
        sim._dialogue = dialogue
        sim._validate_spatial_state()
        return sim

    def _validate_spatial_state(self) -> None:
        p = self._player
        terrain = self._map_specs[self._map_id]
        if p.grounded:
            if p.vy != 0 or not any(
                abs(p.y + PLAYER_H - platform.y) <= 1e-6
                and p.x + PLAYER_W > platform.x1
                and p.x < platform.x2
                for platform in terrain.platforms
            ):
                raise ValueError("grounded player must stand on a platform")
        if p.climbing and not any(
            abs(p.x + PLAYER_W / 2 - (ladder.x + ladder.w / 2)) <= 1e-6
            and ladder.y - PLAYER_H <= p.y <= ladder.y + ladder.h - PLAYER_H
            for ladder in terrain.ladders
        ):
            raise ValueError("climbing player must occupy a ladder")
        existing_drop_ids = {d.id for world in self._maps.values() for d in world.drops}
        for entity_id in existing_drop_ids:
            if entity_id.startswith("drop-") and entity_id[5:].isdigit():
                if int(entity_id[5:]) > self._next_drop_id:
                    raise ValueError("drop counter would reuse an existing ID")
