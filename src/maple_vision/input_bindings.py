"""Versioned keyboard bindings for offline simulated skill dispatch.

Bindings are evidence about a user-confirmed key layout.  They never send input
to an official client.  Only the skill component of a supported simulated
action is gated here; movement and other simulator actions remain explicit.
"""

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import canonical_json, strict_json
from .class_package import ClassPackage
from .sim_types import SimAction

MAX_INPUT_BINDINGS_BYTES = 64 * 1024
MAX_INPUT_CONFIRMATION_BYTES = 128 * 1024
MAX_INPUT_SOURCE_IMAGE_BYTES = 16 * 1024 * 1024
_SOURCE_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_KEY_GRID_ROW = "ASDFGHJKL"


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


def load_input_profile_confirmation(path: Path) -> tuple["InputBindingProfile", dict[str, Any]]:
    """Load and verify one bounded local manual-confirmation receipt."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("input profile confirmation must be a regular nonsymlink file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_CONFIRMATION_BYTES + 1)
    if len(raw) > MAX_INPUT_CONFIRMATION_BYTES:
        raise ValueError("input profile confirmation exceeds byte budget")
    try:
        value = strict_json(raw.decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("input profile confirmation must be UTF-8 JSON") from exc
    return verify_input_profile_confirmation(value)


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

    def binding_for_key(self, key: str) -> KeyBinding | None:
        """Return the configured entry for one normalized key without dispatching it."""
        normalized = normalize_key(key)
        if not normalized:
            raise ValueError("key name must not be blank")
        return next((item for item in self.bindings if normalize_key(item.key) == normalized), None)

    def action_for_key(
        self, key: str, job: str, class_package: ClassPackage | None
    ) -> tuple[str | None, SimAction | None]:
        item = self.binding_for_key(key)
        if item is None:
            return None, None
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


def confirm_input_profile(
    profile: InputBindingProfile,
    comparison_source: str,
    source_image: object = None,
    key_grid_review: object = None,
) -> dict[str, Any]:
    """Create a deterministic receipt for a user's visible key-layout comparison.

    A browser-selected image contributes metadata and a locally computed digest only.
    The image bytes remain in the browser and this receipt never upgrades manual
    comparison into automatic skill recognition.
    """
    sources = {"browser_selected_image_digest", "live_window_manual_comparison"}
    if comparison_source not in sources:
        raise ValueError("unsupported input binding comparison source")
    normalized_image = None
    if comparison_source == "live_window_manual_comparison":
        if source_image is not None:
            raise ValueError("live-window comparison cannot carry selected-image metadata")
    else:
        if not isinstance(source_image, dict) or set(source_image) != {
            "sha256",
            "bytes",
            "media_type",
            "width",
            "height",
        }:
            raise ValueError("selected-image comparison requires exact image metadata")
        digest = source_image["sha256"]
        size = source_image["bytes"]
        media_type = source_image["media_type"]
        width = source_image["width"]
        height = source_image["height"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("source image SHA-256 must be lowercase hexadecimal")
        if type(size) is not int or not 1 <= size <= MAX_INPUT_SOURCE_IMAGE_BYTES:
            raise ValueError("source image byte count exceeds the 16 MiB limit")
        if media_type not in _SOURCE_IMAGE_MEDIA_TYPES:
            raise ValueError("unsupported source image media type")
        if (
            type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 8192
            or not 1 <= height <= 8192
            or width * height > 32 * 1024 * 1024
        ):
            raise ValueError("source image dimensions exceed the 32-megapixel limit")
        normalized_image = {
            "sha256": digest,
            "bytes": size,
            "media_type": media_type,
            "width": width,
            "height": height,
            "digest_origin": "client_reported_browser_web_crypto",
            "image_bytes_uploaded": False,
        }
    normalized_review = None
    if key_grid_review is not None:
        if normalized_image is None:
            raise ValueError("key-grid review requires a selected source image")
        if not isinstance(key_grid_review, dict) or set(key_grid_review) != {
            "schema_version",
            "row",
            "anchors",
            "algorithm_candidate_keys",
            "confirmed_content_keys",
            "user_reviewed",
        }:
            raise ValueError("invalid key-grid review fields")
        if (
            key_grid_review["schema_version"] != 1
            or type(key_grid_review["schema_version"]) is not int
            or key_grid_review["row"] != _KEY_GRID_ROW
            or key_grid_review["user_reviewed"] is not True
        ):
            raise ValueError("unsupported key-grid review")
        anchors = key_grid_review["anchors"]
        if (
            not isinstance(anchors, list)
            or len(anchors) != 2
            or any(not isinstance(item, dict) or set(item) != {"key", "x", "y"} for item in anchors)
            or [item["key"] for item in anchors] != ["A", "L"]
        ):
            raise ValueError("key-grid review requires ordered A and L anchors")
        coordinates: list[tuple[float, float]] = []
        for item in anchors:
            x, y = item["x"], item["y"]
            if (
                type(x) not in {int, float}
                or type(y) not in {int, float}
                or not math.isfinite(x)
                or not math.isfinite(y)
                or not 0 <= x < normalized_image["width"]
                or not 0 <= y < normalized_image["height"]
            ):
                raise ValueError("key-grid anchor is outside the source image")
            coordinates.append((float(x), float(y)))
        pitch = (coordinates[1][0] - coordinates[0][0]) / 8
        tile_size = math.floor(pitch * 0.9 + 0.5)
        if (
            not 8 <= pitch <= 160
            or abs(coordinates[1][1] - coordinates[0][1]) > pitch / 3
            or coordinates[0][0] - tile_size / 2 < 0
            or coordinates[1][0] + tile_size / 2 > normalized_image["width"]
            or min(coordinates[0][1], coordinates[1][1]) - tile_size / 2 < 0
            or max(coordinates[0][1], coordinates[1][1]) + tile_size / 2
            > normalized_image["height"]
        ):
            raise ValueError("key-grid anchors do not describe one bounded A-L row")

        def normalized_keys(name: str) -> list[str]:
            keys = key_grid_review[name]
            if (
                not isinstance(keys, list)
                or any(not isinstance(key, str) or key not in _KEY_GRID_ROW for key in keys)
                or keys != [key for key in _KEY_GRID_ROW if key in keys]
            ):
                raise ValueError(f"{name} must be unique keys in A-L row order")
            return keys

        algorithm_keys = normalized_keys("algorithm_candidate_keys")
        confirmed_keys = normalized_keys("confirmed_content_keys")
        normalized_review = {
            "schema_version": 1,
            "kind": "user_reviewed_key_content_candidates",
            "source_image_sha256": normalized_image["sha256"],
            "row": _KEY_GRID_ROW,
            "anchors": [
                {"key": key, "x": round(x, 3), "y": round(y, 3)}
                for key, (x, y) in zip(("A", "L"), coordinates, strict=True)
            ],
            "tile_size": tile_size,
            "heuristic": {
                "sample_region": [0.19, 0.52, 0.62, 0.26],
                "dark_max_channel_lt": 195,
                "dark_ratio_gte": 0.35,
                "candidate_keys": algorithm_keys,
            },
            "player_review": {
                "user_reviewed": True,
                "content_keys": confirmed_keys,
                "added_keys": [key for key in confirmed_keys if key not in algorithm_keys],
                "removed_keys": [key for key in algorithm_keys if key not in confirmed_keys],
            },
        }
    manifest = profile.manifest()
    receipt: dict[str, Any] = {
        "schema_version": 2 if normalized_review is not None else 1,
        "kind": "user_confirmed_input_binding_profile",
        "scope": "manual_visible_comparison_only",
        "profile": manifest,
        "profile_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        "comparison_source": comparison_source,
        "source_image": normalized_image,
        "recognition": {
            "method": "manual_visible_comparison",
            "automatic_skill_recognition": False,
            "candidates_generated": False,
        },
        "confirmation": {"user_confirmed": True},
        "limitations": [
            "the receipt records a user-confirmed binding profile, not automatic skill recognition",
            "a selected-image digest is client-reported; image bytes are not uploaded or "
            "server-rehashed",
            "the receipt does not prove that a key was pressed or that an official "
            "client accepted input",
        ],
    }
    if normalized_review is not None:
        receipt["key_grid_review"] = normalized_review
        receipt["recognition"]["key_content_candidates_generated"] = True
        receipt["recognition"]["skill_candidates_generated"] = False
        receipt["limitations"].insert(
            1, "key-grid content candidates do not identify a skill or prove a binding"
        )
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
    return receipt


def verify_input_profile_confirmation(
    value: object,
) -> tuple[InputBindingProfile, dict[str, Any]]:
    """Verify and normalize a deterministic manual key-layout confirmation receipt."""
    if not isinstance(value, dict):
        raise ValueError("input profile confirmation must be an object")
    try:
        profile = InputBindingProfile.from_manifest(value["profile"])
        comparison_source = value["comparison_source"]
        recorded_image = value["source_image"]
    except KeyError as exc:
        raise ValueError("input profile confirmation is incomplete") from exc
    if not isinstance(comparison_source, str):
        raise ValueError("input profile confirmation source is invalid")
    image = None
    if recorded_image is not None:
        if not isinstance(recorded_image, dict):
            raise ValueError("input profile confirmation source image is invalid")
        try:
            image = {
                key: recorded_image[key]
                for key in ("sha256", "bytes", "media_type", "width", "height")
            }
        except KeyError as exc:
            raise ValueError("input profile confirmation source image is incomplete") from exc
    review = None
    recorded_review = value.get("key_grid_review")
    if recorded_review is not None:
        if not isinstance(recorded_review, dict):
            raise ValueError("input profile confirmation key-grid review is invalid")
        try:
            heuristic = recorded_review["heuristic"]
            player_review = recorded_review["player_review"]
            if not isinstance(heuristic, dict) or not isinstance(player_review, dict):
                raise ValueError("input profile confirmation key-grid review is invalid")
            review = {
                "schema_version": recorded_review["schema_version"],
                "row": recorded_review["row"],
                "anchors": recorded_review["anchors"],
                "algorithm_candidate_keys": heuristic["candidate_keys"],
                "confirmed_content_keys": player_review["content_keys"],
                "user_reviewed": player_review["user_reviewed"],
            }
        except KeyError as exc:
            raise ValueError("input profile confirmation key-grid review is incomplete") from exc
    expected = confirm_input_profile(profile, comparison_source, image, review)
    if canonical_json(value) != canonical_json(expected):
        raise ValueError("input profile confirmation receipt is not canonical or was modified")
    return profile, expected
