"""Versioned configuration for the opt-in structured offline runtime.

The original image-only policy remains the legacy default. This config is
embedded in schema-2 run evidence so replay never guesses which runtime ran.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .mission_program import MissionProgram
from .state_estimation import StateConfig
from .task_runtime import TaskRuntimeConfig

RUNTIME_ENGINE = "structured-v1"
MAX_RUNTIME_CONFIG_BYTES = 64 * 1024


@dataclass(frozen=True)
class RuntimeConfig:
    state: StateConfig = field(default_factory=StateConfig)
    tasks: TaskRuntimeConfig = field(default_factory=TaskRuntimeConfig)
    mission: MissionProgram | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not StateConfig or type(self.tasks) is not TaskRuntimeConfig:
            raise ValueError("runtime configuration requires strict state and task configs")
        if self.mission is not None and type(self.mission) is not MissionProgram:
            raise ValueError("invalid mission program")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": 1 if self.mission is None else 2,
            "engine": RUNTIME_ENGINE,
            "state": self.state.to_dict(),
            "tasks": self.tasks.to_dict(),
        }
        if self.mission is not None:
            result["mission"] = self.mission.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: object) -> "RuntimeConfig":
        if not isinstance(value, dict):
            raise ValueError("invalid structured runtime configuration")
        fields = {"schema_version", "engine", "state", "tasks"}
        if value.get("schema_version") == 2:
            fields.add("mission")
        if (
            set(value) != fields
            or type(value["schema_version"]) is not int
            or value["schema_version"] not in (1, 2)
            or value["engine"] != RUNTIME_ENGINE
        ):
            raise ValueError("invalid structured runtime configuration")
        return cls(
            StateConfig.from_dict(value["state"]),
            TaskRuntimeConfig.from_dict(value["tasks"]),
            MissionProgram.from_dict(value["mission"]) if value["schema_version"] == 2 else None,
        )

    @property
    def content_sha256(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "settings": self.to_dict(),
            "settings_sha256": self.content_sha256,
        }

    @classmethod
    def from_manifest(cls, value: object) -> "RuntimeConfig":
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "settings", "settings_sha256"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not isinstance(value["settings_sha256"], str)
        ):
            raise ValueError("invalid structured runtime manifest")
        config = cls.from_dict(value["settings"])
        if value["settings_sha256"] != config.content_sha256:
            raise ValueError("structured runtime config checksum mismatch")
        return config
