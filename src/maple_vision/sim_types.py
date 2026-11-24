"""Public, immutable independent-simulator fixtures (not official game rules)."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

SIM_VERSION: Final = 1
RULESET: Final = "synthetic-fixture-v1"
SUPPORTED_JOBS: Final = frozenset({"assassin", "dragon_knight", "cleric"})


@dataclass(frozen=True)
class ClassSpec:
    id: str
    max_hp: int
    max_mp: int
    attack_damage: int
    attack_range: int
    attack_mp_cost: int
    attack_cooldown_ms: int
    ammo_cost: int
    can_jump_attack: bool
    heal_amount: int = 0
    walk_speed: int = 80
    jump_speed: int = 280

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or self.id not in SUPPORTED_JOBS:
            raise ValueError("class profile must identify a supported fixture job")
        for name, value, minimum, maximum in (
            ("max_hp", self.max_hp, 1, 100_000),
            ("max_mp", self.max_mp, 0, 100_000),
            ("attack_damage", self.attack_damage, 1, 100_000),
            ("attack_range", self.attack_range, 1, 320),
            ("attack_mp_cost", self.attack_mp_cost, 0, self.max_mp),
            ("attack_cooldown_ms", self.attack_cooldown_ms, 10, 60_000),
            ("ammo_cost", self.ammo_cost, 0, 200),
            ("heal_amount", self.heal_amount, 0, self.max_hp),
            ("walk_speed", self.walk_speed, 1, 200),
            ("jump_speed", self.jump_speed, 1, 400),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"class profile {name} must be an integer in [{minimum}, {maximum}]"
                )
        if type(self.can_jump_attack) is not bool:
            raise ValueError("class profile can_jump_attack must be boolean")


CLASS_SPECS = MappingProxyType(
    {
        "assassin": ClassSpec("assassin", 100, 60, 18, 92, 2, 200, 1, True),
        "dragon_knight": ClassSpec("dragon_knight", 150, 50, 24, 42, 0, 300, 0, True),
        "cleric": ClassSpec("cleric", 95, 120, 16, 76, 4, 250, 0, True, 35),
    }
)

ACTION_KINDS: Final = frozenset(
    {
        "idle",
        "move",
        "jump",
        "attack",
        "jump_attack",
        "loot",
        "potion_hp",
        "potion_mp",
        "buff",
        "heal",
        "interact",
        "portal",
        "climb",
    }
)


@dataclass(frozen=True)
class SimAction:
    kind: str = "idle"
    direction: int = 0
    skill: str = "basic"
    target: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in ACTION_KINDS:
            raise ValueError("unknown simulator action")
        if type(self.direction) is not int or self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or 1")
        if self.kind in {"move", "climb"} and self.direction == 0:
            raise ValueError("move/climb requires a direction")
        if not isinstance(self.skill, str) or self.skill not in {"basic", "area"}:
            raise ValueError("unknown skill")
        if self.skill != "basic" and self.kind not in {"attack", "jump_attack"}:
            raise ValueError("a skill is only valid on an attack")
        if not isinstance(self.target, str) or len(self.target) > 80:
            raise ValueError("target must be a short string")
        if self.target and self.kind not in {"interact", "portal"}:
            raise ValueError("target is only valid on portal/interact")
        if self.direction and self.kind not in {"move", "climb", "jump", "jump_attack", "attack"}:
            raise ValueError("direction is not valid on this action")


@dataclass(frozen=True)
class PlayerState:
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    hp: int
    max_hp: int
    mp: int
    max_mp: int
    ammo: int
    hp_potions: int
    mp_potions: int
    coins: int
    buffs: tuple[str, ...]
    grounded: bool
    facing: int
    job: str
    alive: bool


@dataclass(frozen=True)
class MonsterState:
    id: str
    x: float
    y: float
    w: float
    h: float
    hp: int
    max_hp: int


@dataclass(frozen=True)
class DropState:
    id: str
    x: float
    y: float
    w: float
    h: float
    item: str
    quantity: int


@dataclass(frozen=True)
class Platform:
    x1: float
    x2: float
    y: float


@dataclass(frozen=True)
class Portal:
    id: str
    x: float
    y: float
    w: float
    h: float
    destination: str


@dataclass(frozen=True)
class NPC:
    id: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Ladder:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class SimSnapshot:
    sequence: int
    timestamp_ms: int
    map_id: str
    width: int
    height: int
    player: PlayerState
    monsters: tuple[MonsterState, ...]
    drops: tuple[DropState, ...]
    platforms: tuple[Platform, ...]
    portals: tuple[Portal, ...]
    npcs: tuple[NPC, ...]
    ladders: tuple[Ladder, ...]
    events: tuple[str, ...]
    dialogue: str | None


@dataclass(frozen=True)
class StepOutcome:
    snapshot: SimSnapshot
    accepted: bool
    reason: str
    events: tuple[str, ...]
