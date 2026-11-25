"""Versioned keyboard bindings for offline simulated skill dispatch.

Bindings are evidence about a user-confirmed key layout.  They never send input
to an official client.  Only the skill component of a supported simulated
action is gated here; movement and other simulator actions remain explicit.
"""

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import strict_json
from .class_package import ClassPackage
from .sim_types import SimAction

MAX_INPUT_BINDINGS_BYTES = 64 * 1024


def load_input_profile(path: Path) -> "InputBindingProfile":
    """Load one bounded local JSON profile without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("input binding profile must be a regular nonsymlink file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BINDINGS_BYTES + 1)
    if len(raw) > MAX_INPUT_BINDINGS_BYTES:
        raise ValueError("input binding profile exceeds byte budget")
    try:
        return InputBindingProfile.from_manifest(strict_json(raw.decode("utf-8")))
    except UnicodeError as exc:
        raise ValueError("input binding profile must be UTF-8 JSON") from exc


def normalize_key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("key name must be text")
    name = "".join(unicodedata.normalize("NFKC", value).casefold().split())
    return {"control": "ctrl", "esc": "escape", " ": "space"}.get(name, name)


def _normalize_skill(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("skill name must be text")
    return unicodedata.normalize("NFKC", value).strip().casefold()


def action_for_skill(job: str, class_package: ClassPackage | None, skill: str) -> SimAction | None:
    """Resolve a configured label to a supported offline simulator action."""
    normalized = _normalize_skill(skill)
    if class_package is not None:
        for item in class_package.skills:
            if normalized == _normalize_skill(item.id):
                return SimAction("cast", skill=item.id)
        return None
    if normalized in {_normalize_skill("basic"), _normalize_skill("普通攻击")}:
        return SimAction("attack")
    if job == "dragon_knight" and normalized in {
        _normalize_skill("area"),
        _normalize_skill("模拟范围攻击"),
    }:
        return SimAction("attack", skill="area")
    return None


def _required_skill(
    action: SimAction, job: str, class_package: ClassPackage | None
) -> tuple[str, frozenset[str]] | None:
    if class_package is not None and action.kind == "cast":
        if action.skill not in {item.id for item in class_package.skills}:
            raise ValueError("simulated cast is absent from the class package")
        return action.skill, frozenset({_normalize_skill(action.skill)})
    if class_package is None and action.kind == "attack":
        if action.skill == "area":
            if job != "dragon_knight":
                raise ValueError("area attack is not a supported built-in skill")
            return "模拟范围攻击", frozenset(
                {_normalize_skill("area"), _normalize_skill("模拟范围攻击")}
            )
        if action.skill == "basic":
            return "普通攻击", frozenset({_normalize_skill("basic"), _normalize_skill("普通攻击")})
    return None


@dataclass(frozen=True)
class KeyBinding:
    key: str
    skill: str

    def __post_init__(self) -> None:
        if not 1 <= len(self.key) <= 80 or not normalize_key(self.key):
            raise ValueError("key name must contain 1..80 visible characters")
        if not 1 <= len(self.skill) <= 80 or not _normalize_skill(self.skill):
            raise ValueError("skill name must contain 1..80 visible characters")

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "skill": self.skill}


@dataclass(frozen=True)
class InputBindingProfile:
    profile_id: str
    bindings: tuple[KeyBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not 1 <= len(self.profile_id.strip()) <= 100:
            raise ValueError("input profile ID must contain 1..100 characters")
        if not isinstance(self.bindings, tuple) or not 1 <= len(self.bindings) <= 128:
            raise ValueError("input profile must contain 1..128 bindings")
        keys = [normalize_key(item.key) for item in self.bindings]
        skills = [_normalize_skill(item.skill) for item in self.bindings]
        if len(set(keys)) != len(keys):
            raise ValueError("input profile contains duplicate normalized keys")
        if len(set(skills)) != len(skills):
            raise ValueError("input profile contains duplicate normalized skills")

    @classmethod
    def from_manifest(cls, value: object) -> "InputBindingProfile":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "profile_id",
            "bindings",
        }:
            raise ValueError("invalid input binding profile fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported input binding profile schema")
        raw = value["bindings"]
        if not isinstance(raw, list):
            raise ValueError("input profile bindings must be a list")
        bindings: list[KeyBinding] = []
        for item in raw:
            if not isinstance(item, dict) or set(item) != {"key", "skill"}:
                raise ValueError("invalid input binding entry")
            bindings.append(KeyBinding(item["key"], item["skill"]))
        return cls(value["profile_id"], tuple(bindings))

    @classmethod
    def from_key_map(cls, profile_id: str, value: dict[str, str]) -> "InputBindingProfile":
        if not isinstance(value, dict):
            raise ValueError("input bindings must be an object")
        return cls(profile_id, tuple(KeyBinding(key, skill) for key, skill in value.items()))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "bindings": [item.to_dict() for item in self.bindings],
        }

    def action_for_key(
        self, key: str, job: str, class_package: ClassPackage | None
    ) -> tuple[str | None, SimAction | None]:
        normalized = normalize_key(key)
        if not normalized:
            raise ValueError("key name must not be blank")
        matching = [item for item in self.bindings if normalize_key(item.key) == normalized]
        if not matching:
            return None, None
        item = matching[0]
        return item.skill, action_for_skill(job, class_package, item.skill)

    def dispatch(
        self, action: SimAction, job: str, class_package: ClassPackage | None
    ) -> tuple[SimAction, dict[str, Any] | None]:
        required = _required_skill(action, job, class_package)
        if required is None:
            return action, None
        action_skill, aliases = required
        matches = [item for item in self.bindings if _normalize_skill(item.skill) in aliases]
        if len(matches) > 1:
            raise ValueError("multiple bindings resolve to the same simulated skill")
        item = None if not matches else matches[0]
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "scope": "offline_simulated_skill_dispatch",
            "action_skill": action_skill,
            "key": None if item is None else item.key,
            "normalized_key": None if item is None else normalize_key(item.key),
            "configured_skill": None if item is None else item.skill,
            "accepted": item is not None,
            "reason": "bound_skill" if item is not None else "unbound_skill",
        }
        return (action if item is not None else SimAction()), evidence
