"""Bounded authored capability data and a compiler for the offline simulator.

Packages contain no executable code, paths to fetch, loops or predicates. The
compiler resolves selected skill levels and finite linear action programs into
static capabilities. It never imports the simulator, policy or observations.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .catalog import canonical_json, strict_json, validate_provenance
from .sim_types import SimAction

MAX_CLASS_PACKAGE_BYTES = 64 * 1024
CAPABILITY_COMPILER_VERSION = 3
_ID = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = ("combat", "area", "airborne", "heal", "buff")


def _integer(value: object, field: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid bounded {field}")
    return value


def _choice(value: object, values: set[str], field: str) -> str:
    if not isinstance(value, str) or value not in values:
        raise ValueError(f"invalid {field}")
    return value


def _object(value: object, names: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"invalid {field} fields")
    return value


def _array(value: object, field: str, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{field} requires {minimum}..{maximum} entries")
    return value


def _entries(value: object, kind: type, field: str, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, tuple)
        or not minimum <= len(value) <= maximum
        or any(type(item) is not kind for item in value)
    ):
        raise ValueError(f"invalid immutable {field}")


@dataclass(frozen=True)
class ClassResources:
    max_hp: int
    max_mp: int
    initial_ammo: int
    walk_speed: int
    jump_speed: int

    def __post_init__(self) -> None:
        for field, low, high in (
            ("max_hp", 1, 100_000),
            ("max_mp", 0, 100_000),
            ("initial_ammo", 0, 1_000_000),
            ("walk_speed", 1, 200),
            ("jump_speed", 1, 400),
        ):
            _integer(getattr(self, field), field, low, high)

    @classmethod
    def from_dict(cls, value: object) -> ClassResources:
        return cls(**_object(value, {field.name for field in fields(cls)}, "class resources"))


@dataclass(frozen=True)
class SkillLevel:
    level: int
    power: int
    range_px: int
    vertical_range_px: int
    max_targets: int
    hp_cost: int
    mp_cost: int
    ammo_cost: int
    cooldown_ms: int
    duration_ms: int
    hit_count: int = 1

    def __post_init__(self) -> None:
        for field, low, high in (
            ("level", 1, 100),
            ("power", 1, 100_000),
            ("range_px", 0, 320),
            ("vertical_range_px", 0, 192),
            ("max_targets", 0, 8),
            ("hp_cost", 0, 99_999),
            ("mp_cost", 0, 100_000),
            ("ammo_cost", 0, 200),
            ("cooldown_ms", 10, 60_000),
            ("duration_ms", 0, 60_000),
            ("hit_count", 1, 16),
        ):
            _integer(getattr(self, field), field, low, high)

    @classmethod
    def from_dict(cls, value: object, *, include_hit_count: bool = False) -> SkillLevel:
        names = {field.name for field in fields(cls)}
        if not include_hit_count:
            names.remove("hit_count")
        raw = dict(_object(value, names, "skill level"))
        if not include_hit_count:
            raw["hit_count"] = 1
        return cls(**raw)

    def to_dict(self, *, include_hit_count: bool = False) -> dict[str, int]:
        result = asdict(self)
        if not include_hit_count:
            result.pop("hit_count")
        return result


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    kind: str
    shape: str
    posture: str
    cooldown_group: str
    selected_level: int
    levels: tuple[SkillLevel, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "skill ID")
        _identifier(self.cooldown_group, "cooldown group")
        _choice(self.kind, {"damage", "heal", "buff"}, "skill kind")
        _choice(self.shape, {"directional", "radial", "self"}, "skill shape")
        _choice(self.posture, {"ground", "air", "any"}, "skill posture")
        _integer(self.selected_level, "selected level", 1, 100)
        _entries(self.levels, SkillLevel, "skill levels", 1, 16)
        levels = {level.level: level for level in self.levels}
        if len(levels) != len(self.levels) or self.selected_level not in levels:
            raise ValueError("skill levels must be unique and include the selected level")
        object.__setattr__(self, "levels", tuple(levels[key] for key in sorted(levels)))
        for level in self.levels:
            if self.kind == "damage":
                if self.shape == "self" or not all(
                    (level.range_px, level.vertical_range_px, level.max_targets)
                ):
                    raise ValueError("damage requires a spatial shape, range and target budget")
            elif self.shape != "self" or any(
                (level.range_px, level.vertical_range_px, level.max_targets, level.ammo_cost)
            ):
                raise ValueError("heal/buff must be self-only without spatial or ammo parameters")
            if (self.kind == "buff") != (level.duration_ms > 0):
                raise ValueError("only buff skills have a positive duration")

    @property
    def selected(self) -> SkillLevel:
        return next(level for level in self.levels if level.level == self.selected_level)

    def to_dict(self, *, include_hit_count: bool = False) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "shape": self.shape,
            "posture": self.posture,
            "cooldown_group": self.cooldown_group,
            "selected_level": self.selected_level,
            "levels": [level.to_dict(include_hit_count=include_hit_count) for level in self.levels],
        }

    @classmethod
    def from_dict(cls, value: object, *, include_hit_count: bool = False) -> SkillDefinition:
        raw = dict(_object(value, {field.name for field in fields(cls)}, "skill definition"))
        raw["levels"] = tuple(
            SkillLevel.from_dict(level, include_hit_count=include_hit_count)
            for level in _array(raw["levels"], "levels", 1, 16)
        )
        return cls(**raw)


@dataclass(frozen=True)
class ActionInstruction:
    op: str
    skill: str = ""
    direction: int | str = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        _choice(self.op, {"cast", "move", "jump", "wait"}, "DSL operation")
        if self.op == "wait":
            _integer(self.duration_ms, "wait duration", 1, 5000)
            if self.skill != "" or type(self.direction) is not int or self.direction != 0:
                raise ValueError("wait cannot contain a skill or direction")
        else:
            _integer(self.duration_ms, "action duration", 0, 0)
            if not (
                (type(self.direction) is int and self.direction in {-1, 0, 1})
                or (type(self.direction) is str and self.direction == "target")
            ):
                raise ValueError("DSL direction must be -1/0/1 or target")
            if self.op == "move" and self.direction == 0:
                raise ValueError("move requires a direction")
            if self.op == "cast":
                _identifier(self.skill, "cast skill")
            elif self.skill != "":
                raise ValueError("only cast instructions reference skills")

    def to_dict(self) -> dict[str, object]:
        if self.op == "wait":
            return {"op": self.op, "duration_ms": self.duration_ms}
        result: dict[str, object] = {"op": self.op, "direction": self.direction}
        if self.op == "cast":
            result["skill"] = self.skill
        return result

    @classmethod
    def from_dict(cls, value: object) -> ActionInstruction:
        if not isinstance(value, dict) or not isinstance(value.get("op"), str):
            raise ValueError("DSL instruction requires an operation")
        names = {"op", "duration_ms"} if value["op"] == "wait" else {"op", "direction"}
        if value["op"] == "cast":
            names.add("skill")
        return cls(**_object(value, names, "DSL instruction"))


@dataclass(frozen=True)
class ActionProgram:
    id: str
    steps: tuple[ActionInstruction, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "program ID")
        _entries(self.steps, ActionInstruction, "program instructions", 1, 16)
        if self.steps[-1].op == "wait":
            raise ValueError("a program cannot end in an unobservable trailing wait")
        if not 1 <= sum(step.op != "wait" for step in self.steps) <= 8:
            raise ValueError("program requires 1..8 executable actions")
        if sum(step.duration_ms for step in self.steps) > 5000:
            raise ValueError("program wait budget exceeds 5000ms")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, value: object) -> ActionProgram:
        raw = _object(value, {"id", "steps"}, "action program")
        return cls(
            raw["id"],
            tuple(
                ActionInstruction.from_dict(step)
                for step in _array(raw["steps"], "program instructions", 1, 16)
            ),
        )


@dataclass(frozen=True)
class ProgramBindings:
    combat: str
    area: str | None
    airborne: str | None
    heal: str | None
    buff: str | None

    def __post_init__(self) -> None:
        _identifier(self.combat, "combat program")
        for role in _ROLES[1:]:
            value = getattr(self, role)
            if value is not None:
                _identifier(value, f"{role} program")

    @classmethod
    def from_dict(cls, value: object) -> ProgramBindings:
        return cls(**_object(value, set(_ROLES), "program bindings"))


@dataclass(frozen=True)
class TimedEffect:
    skill: str
    kind: str
    duration_ms: int
    interval_ms: int
    tick_damage: int

    def __post_init__(self) -> None:
        _identifier(self.skill, "effect skill")
        _choice(self.kind, {"periodic_damage", "disable_contact"}, "timed effect")
        _integer(self.duration_ms, "effect duration", 100, 60000)
        if self.kind == "periodic_damage":
            _integer(self.interval_ms, "tick interval", 100, self.duration_ms)
            _integer(self.tick_damage, "tick damage", 1, 100000)
        else:
            _integer(self.interval_ms, "control tick interval", 0, 0)
            _integer(self.tick_damage, "control tick damage", 0, 0)

    @classmethod
    def from_dict(cls, value: object) -> TimedEffect:
        return cls(**_object(value, {field.name for field in fields(cls)}, "timed effect"))


@dataclass(frozen=True)
class ClassPackage:
    id: str
    version: str
    class_id: str
    source_uri: str
    permission: str
    license: str
    resources: ClassResources
    skills: tuple[SkillDefinition, ...]
    programs: tuple[ActionProgram, ...]
    bindings: ProgramBindings
    timed_effects: tuple[TimedEffect, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.id, "class package ID")
        _identifier(self.class_id, "class ID")
        if (
            not isinstance(self.version, str)
            or len(self.version) > 40
            or not _VERSION.fullmatch(self.version)
        ):
            raise ValueError("class package version must be bounded major.minor.patch")
        validate_provenance(self.source_uri, self.permission, self.license)
        if type(self.resources) is not ClassResources or type(self.bindings) is not ProgramBindings:
            raise ValueError("class package requires immutable resources and bindings")
        _entries(self.skills, SkillDefinition, "skill registry", 1, 16)
        _entries(self.programs, ActionProgram, "program registry", 1, 16)
        if len({skill.id for skill in self.skills}) != len(self.skills):
            raise ValueError("duplicate skill ID")
        if len({program.id for program in self.programs}) != len(self.programs):
            raise ValueError("duplicate program ID")
        object.__setattr__(self, "skills", tuple(sorted(self.skills, key=lambda skill: skill.id)))
        object.__setattr__(
            self, "programs", tuple(sorted(self.programs, key=lambda program: program.id))
        )
        for skill in self.skills:
            for level in skill.levels:
                if (
                    level.hp_cost >= self.resources.max_hp
                    or level.mp_cost > self.resources.max_mp
                    or level.ammo_cost > self.resources.initial_ammo
                    or (skill.kind == "heal" and level.power > self.resources.max_hp)
                ):
                    raise ValueError("skill level exceeds class resources or cannot preserve 1HP")
        _entries(self.timed_effects, TimedEffect, "timed effects", 0, 16)
        damage_skills = {skill.id for skill in self.skills if skill.kind == "damage"}
        attached = {effect.skill for effect in self.timed_effects}
        if len(attached) != len(self.timed_effects) or not attached <= damage_skills:
            raise ValueError("each timed effect needs a unique registered damage skill")
        object.__setattr__(
            self, "timed_effects", tuple(sorted(self.timed_effects, key=lambda s: s.skill))
        )
        try:
            size = len(canonical_json(self.to_dict()).encode("utf-8"))
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("class package contains invalid text") from exc
        if size > MAX_CLASS_PACKAGE_BYTES:
            raise ValueError("class package exceeds 64KiB")
        # Semantic validation is also required for direct immutable construction.
        compile_class_package(self)

    def to_dict(self) -> dict[str, object]:
        version = self.compiler_version
        return {
            "schema_version": version,
            **(
                {"timed_effects": [asdict(effect) for effect in self.timed_effects]}
                if version >= 2
                else {}
            ),
            "id": self.id,
            "version": self.version,
            "scope": "synthetic_fixture",
            "class_id": self.class_id,
            "source_uri": self.source_uri,
            "permission": self.permission,
            "license": self.license,
            "resources": asdict(self.resources),
            "skills": [skill.to_dict(include_hit_count=version >= 3) for skill in self.skills],
            "programs": [program.to_dict() for program in self.programs],
            "bindings": asdict(self.bindings),
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def compiler_version(self) -> int:
        if any(level.hit_count != 1 for skill in self.skills for level in skill.levels):
            return 3
        return 2 if self.timed_effects else 1

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "compiler_version": self.compiler_version,
            "content": self.to_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClassPackage:
        names = {field.name for field in fields(cls)} | {"schema_version", "scope"}
        if not isinstance(value, dict):
            raise ValueError("invalid class package")
        version = _integer(
            value.get("schema_version"), "class package schema", 1, CAPABILITY_COMPILER_VERSION
        )
        if version == 1:
            names.remove("timed_effects")
        raw = dict(_object(value, names, "class package"))
        raw.pop("schema_version")
        if version >= 2:
            raw["timed_effects"] = tuple(
                TimedEffect.from_dict(row)
                for row in _array(
                    raw["timed_effects"], "timed effects", 0 if version >= 3 else 1, 16
                )
            )
        if raw.pop("scope") != "synthetic_fixture":
            raise ValueError("class package scope must be synthetic_fixture")
        raw["resources"] = ClassResources.from_dict(raw["resources"])
        raw["bindings"] = ProgramBindings.from_dict(raw["bindings"])
        raw["skills"] = tuple(
            SkillDefinition.from_dict(item, include_hit_count=version >= 3)
            for item in _array(raw["skills"], "skills", 1, 16)
        )
        raw["programs"] = tuple(
            ActionProgram.from_dict(item) for item in _array(raw["programs"], "programs", 1, 16)
        )
        package = cls(**raw)
        if package.compiler_version != version:
            raise ValueError("class package schema disagrees with declared capabilities")
        return package

    @classmethod
    def from_manifest(cls, value: object) -> ClassPackage:
        raw = _object(
            value,
            {"schema_version", "compiler_version", "content", "content_sha256"},
            "class manifest",
        )
        _integer(raw["schema_version"], "class manifest schema", 1, 1)
        _integer(
            raw["compiler_version"], "capability compiler version", 1, CAPABILITY_COMPILER_VERSION
        )
        digest = raw["content_sha256"]
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("invalid class package SHA256")
        package = cls.from_dict(raw["content"])
        if raw["compiler_version"] != package.compiler_version:
            raise ValueError("compiler version disagrees with content schema")
        if package.content_sha256 != digest:
            raise ValueError("class package SHA256 mismatch")
        return package


@dataclass(frozen=True)
class CompiledSkill:
    id: str
    kind: str
    shape: str
    posture: str
    cooldown_group: str
    level: int
    power: int
    range_px: int
    vertical_range_px: int
    max_targets: int
    hp_cost: int
    mp_cost: int
    ammo_cost: int
    cooldown_ms: int
    duration_ms: int
    hit_count: int


@dataclass(frozen=True)
class CompiledStep:
    action: SimAction
    not_before_ms: int


@dataclass(frozen=True)
class CompiledInstruction:
    instruction: ActionInstruction
    not_before_ms: int


@dataclass(frozen=True)
class CompiledProgram:
    id: str
    instructions: tuple[CompiledInstruction, ...]
    range_px: int
    posture: str
    hp_cost: int
    mp_cost: int
    ammo_cost: int
    duration_ms: int
    reuse_after_ms: int
    cooldown_ready_offsets: tuple[tuple[str, int], ...]
    skill_ids: tuple[str, ...]
    max_targets: int

    def instantiate(self, direction: int) -> tuple[CompiledStep, ...]:
        _integer(direction, "target direction", -1, 1)
        if direction == 0:
            raise ValueError("target direction must be -1 or 1")
        result = []
        for step in self.instructions:
            raw = step.instruction
            resolved = direction if raw.direction == "target" else raw.direction
            assert type(resolved) is int
            action = SimAction(raw.op, resolved, raw.skill if raw.op == "cast" else "basic")
            result.append(CompiledStep(action, step.not_before_ms))
        return tuple(result)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["instructions"] = [
            {"instruction": step.instruction.to_dict(), "not_before_ms": step.not_before_ms}
            for step in self.instructions
        ]
        result["cooldown_ready_offsets"] = [list(item) for item in self.cooldown_ready_offsets]
        result["skill_ids"] = list(self.skill_ids)
        return result


@dataclass(frozen=True)
class CapabilityClassSpec:
    """Static numeric facade; does not relax the historical three-job ClassSpec."""

    id: str
    max_hp: int
    max_mp: int
    attack_damage: int
    attack_range: int
    attack_mp_cost: int
    attack_cooldown_ms: int
    ammo_cost: int
    can_jump_attack: bool
    heal_amount: int
    walk_speed: int
    jump_speed: int


@dataclass(frozen=True)
class CompiledCapabilities:
    package: ClassPackage
    class_spec: CapabilityClassSpec
    skill_entries: tuple[CompiledSkill, ...]
    program_entries: tuple[CompiledProgram, ...]

    @property
    def class_id(self) -> str:
        return self.package.class_id

    @property
    def package_sha256(self) -> str:
        return self.package.content_sha256

    @property
    def skills(self) -> Mapping[str, CompiledSkill]:
        return MappingProxyType({skill.id: skill for skill in self.skill_entries})

    @property
    def programs(self) -> Mapping[str, CompiledProgram]:
        return MappingProxyType({program.id: program for program in self.program_entries})

    @property
    def cooldown_groups(self) -> tuple[str, ...]:
        return tuple(sorted({skill.cooldown_group for skill in self.skill_entries}))

    def program_for(self, role: str) -> CompiledProgram | None:
        _choice(role, set(_ROLES), "capability role")
        key = getattr(self.package.bindings, role)
        return None if key is None else self.programs[key]

    def to_dict(self) -> dict[str, object]:
        skills = [asdict(skill) for skill in self.skill_entries]
        if self.package.compiler_version < 3:
            for skill in skills:
                skill.pop("hit_count")
        return {
            "schema_version": 1,
            "compiler_version": self.package.compiler_version,
            "class_id": self.class_id,
            "package_sha256": self.package_sha256,
            "class_spec": asdict(self.class_spec),
            "skills": skills,
            "programs": [program.to_dict() for program in self.program_entries],
            "bindings": asdict(self.package.bindings),
        }


def _compile_program(
    program: ActionProgram, skills: Mapping[str, CompiledSkill]
) -> CompiledProgram:
    offset = 0
    instructions: list[CompiledInstruction] = []
    used: list[CompiledSkill] = []
    ready: dict[str, int] = {}
    for step in program.steps:
        if step.op == "wait":
            offset += step.duration_ms
            continue
        if step.op == "cast":
            if step.skill not in skills:
                raise ValueError(f"program references unknown skill: {step.skill}")
            skill = skills[step.skill]
            if skill.cooldown_group in ready and offset < ready[skill.cooldown_group]:
                raise ValueError("program violates a declared cooldown group")
            ready[skill.cooldown_group] = offset + skill.cooldown_ms
            used.append(skill)
        instructions.append(CompiledInstruction(step, offset))
    if not used:
        raise ValueError("a capability program must invoke at least one registered skill")
    damage = [skill for skill in used if skill.kind == "damage"]
    first = instructions[0].instruction
    posture = (
        "ground"
        if first.op == "jump"
        else (skills[first.skill].posture if first.op == "cast" else "any")
    )
    return CompiledProgram(
        program.id,
        tuple(instructions),
        min((skill.range_px for skill in damage), default=0),
        posture,
        sum(skill.hp_cost for skill in used),
        sum(skill.mp_cost for skill in used),
        sum(skill.ammo_cost for skill in used),
        offset,
        max(ready.values()),
        tuple(sorted(ready.items())),
        tuple(skill.id for skill in used),
        max((skill.max_targets for skill in damage), default=0),
    )


def compile_class_package(package: ClassPackage) -> CompiledCapabilities:
    if type(package) is not ClassPackage:
        raise ValueError("compiler requires an immutable ClassPackage")
    skills = {
        definition.id: CompiledSkill(
            definition.id,
            definition.kind,
            definition.shape,
            definition.posture,
            definition.cooldown_group,
            **asdict(definition.selected),
        )
        for definition in package.skills
    }
    programs = {program.id: _compile_program(program, skills) for program in package.programs}
    resources = package.resources
    for program in programs.values():
        if (
            program.hp_cost >= resources.max_hp
            or program.mp_cost > resources.max_mp
            or program.ammo_cost > resources.initial_ammo
        ):
            raise ValueError(
                "program resource budget exceeds class capacity or cannot preserve 1HP"
            )
    for role in _ROLES:
        key = getattr(package.bindings, role)
        if key is None:
            continue
        if key not in programs:
            raise ValueError(f"binding references unknown program: {key}")
        kinds = {skills[skill_id].kind for skill_id in programs[key].skill_ids}
        expected = role if role in {"heal", "buff"} else "damage"
        if kinds != {expected}:
            raise ValueError("program role disagrees with skill effects")
        if role == "area" and programs[key].max_targets < 2:
            raise ValueError("area program must declare multiple targets")
    combat = programs[package.bindings.combat]
    if combat.posture == "air":
        raise ValueError("default combat program cannot require starting airborne")
    heal_key = package.bindings.heal
    heal_amount = (
        0 if heal_key is None else max(skills[key].power for key in programs[heal_key].skill_ids)
    )
    spec = CapabilityClassSpec(
        package.class_id,
        resources.max_hp,
        resources.max_mp,
        sum(skills[key].power * skills[key].hit_count for key in combat.skill_ids),
        combat.range_px,
        combat.mp_cost,
        combat.reuse_after_ms,
        combat.ammo_cost,
        package.bindings.airborne is not None,
        heal_amount,
        resources.walk_speed,
        resources.jump_speed,
    )
    return CompiledCapabilities(package, spec, tuple(skills.values()), tuple(programs.values()))


def load_class_package(path: str | Path) -> ClassPackage:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("class package must be a regular local file, not a symlink")
    with path.open("rb") as stream:
        data = stream.read(MAX_CLASS_PACKAGE_BYTES + 1)
    if len(data) > MAX_CLASS_PACKAGE_BYTES:
        raise ValueError("class package exceeds 64KiB")
    try:
        raw = strict_json(data.decode("utf-8"))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("class package must be bounded UTF-8 JSON") from exc
    return ClassPackage.from_dict(raw)
