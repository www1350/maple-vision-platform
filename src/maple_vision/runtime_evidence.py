"""Strict structured-runtime evidence checks independent of action simulation.

These checks bind declared intent, execution and configuration. Reproducing
beliefs, scheduling choices and visual effects still requires actual reinference.
"""

import json
from pathlib import Path
from typing import Any

from .catalog import strict_json
from .class_package import ClassPackage
from .runtime_pipeline import MAX_RUNTIME_CONFIG_BYTES, RuntimeConfig
from .sim_types import SimAction
from .task_runtime import ActionIntent


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid structured runtime {label} fields")
    return value


def _version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("invalid structured runtime evidence schema")


def read_runtime_config(path: Path) -> RuntimeConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid runtime configuration sidecar")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RUNTIME_CONFIG_BYTES + 1)
    if len(raw) > MAX_RUNTIME_CONFIG_BYTES:
        raise ValueError("runtime configuration sidecar exceeds byte budget")
    try:
        return RuntimeConfig.from_manifest(strict_json(raw.decode("utf-8")))
    except UnicodeError as exc:
        raise ValueError("runtime configuration sidecar must be UTF-8") from exc


def load_run_runtime(
    output: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> RuntimeConfig | None:
    schema = manifest.get("schema_version")
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("unsupported runtime bundle schema")
    if type(summary.get("schema_version")) is not int or summary["schema_version"] != schema:
        raise ValueError("runtime summary schema disagrees with manifest")
    path = output / "runtime-config.json"
    files = manifest["files"]
    if schema == 1:
        if (
            "runtime" in manifest
            or "runtime_evidence" in summary
            or "runtime-config.json" in files
            or path.exists()
            or path.is_symlink()
        ):
            raise ValueError("legacy bundle contains structured runtime evidence")
        return None
    if "runtime-config.json" not in files:
        raise ValueError("structured run omits runtime configuration sidecar")
    runtime = RuntimeConfig.from_manifest(manifest.get("runtime"))
    if read_runtime_config(path) != runtime:
        raise ValueError("runtime sidecar disagrees with manifest")
    evidence = _object(
        summary.get("runtime_evidence"), {"schema_version", "settings", "terminal"}, "summary"
    )
    _version(evidence["schema_version"])
    if RuntimeConfig.from_manifest(evidence["settings"]) != runtime:
        raise ValueError("runtime summary configuration disagrees with manifest")
    terminal = _object(
        evidence["terminal"],
        {"schema_version", "belief", "decision", "policy_evaluated", "interruption", "state"},
        "terminal",
    )
    _version(terminal["schema_version"])
    if (
        type(terminal["policy_evaluated"]) is not bool
        or not isinstance(terminal["belief"], dict)
        or not isinstance(terminal["state"], dict)
    ):
        raise ValueError("invalid runtime terminal state")
    perception = summary.get("perception_evidence")
    if not isinstance(perception, dict) or not isinstance(perception.get("terminal"), dict):
        raise ValueError("missing runtime terminal perception evidence")
    cause = perception["terminal"].get("cause")
    if cause == "policy":
        if not terminal["policy_evaluated"] or not isinstance(terminal["decision"], dict):
            raise ValueError("runtime terminal policy evidence is missing")
        if terminal["interruption"] is not None:
            raise ValueError("runtime policy termination has unexpected interruption")
    elif cause in {"supervisor", "step_budget_exhausted"}:
        if (
            terminal["policy_evaluated"]
            or terminal["decision"] is not None
            or not isinstance(terminal["interruption"], dict)
        ):
            raise ValueError("runtime terminal falsely claims an evaluated policy")
    else:
        raise ValueError("unsupported structured runtime terminal cause")
    return runtime


def validate_runtime_step(
    row: dict[str, Any],
    runtime: RuntimeConfig | None,
    *,
    now_ms: int,
    seen_intents: set[str],
) -> SimAction:
    requested = SimAction(**row["decision"]["action"])
    if runtime is None:
        if "runtime" in row:
            raise ValueError("legacy trace contains structured runtime evidence")
        return requested
    data = _object(
        row.get("runtime"),
        {
            "schema_version",
            "belief",
            "decision",
            "policy_evaluated",
            "dispatch",
            "receipt",
            "state",
        },
        "step",
    )
    _version(data["schema_version"])
    execution = row.get("execution")
    if (
        not isinstance(execution, dict)
        or type(execution.get("accepted")) is not bool
        or not isinstance(execution.get("reason"), str)
        or not isinstance(execution.get("applied_action"), dict)
    ):
        raise ValueError("invalid structured runtime execution receipt")
    if (
        type(data["policy_evaluated"]) is not bool
        or not isinstance(data["belief"], dict)
        or not isinstance(data["state"], dict)
    ):
        raise ValueError("invalid runtime belief or task state")
    belief = data["belief"]
    _version(belief.get("schema_version"))
    if (
        type(belief.get("actionable")) is not bool
        or type(belief.get("now_ms")) is not int
        or belief["now_ms"] != now_ms
        or not isinstance(belief.get("observation"), dict)
        or not isinstance(row.get("guard"), dict)
        or row["guard"].get("state") not in {"running", "paused"}
        or data["policy_evaluated"] != (row["guard"]["state"] == "running")
    ):
        raise ValueError("runtime belief clock or policy evaluation disagrees with guard")
    decision = _object(
        data["decision"],
        {"schema_version", "action", "intent", "task", "state", "reason", "evidence"},
        "decision",
    )
    _version(decision["schema_version"])
    for key in ("action", "task", "state", "reason"):
        if _canonical(decision[key]) != _canonical(row["decision"][key]):
            raise ValueError("runtime decision disagrees with requested action/task")
    if decision["intent"] is None:
        if requested != SimAction() or data["dispatch"] is not None or data["receipt"] is not None:
            raise ValueError("runtime action without an authorized intent")
        return SimAction()
    intent = ActionIntent.from_dict(decision["intent"])
    observation = row["observation"]
    if (
        intent.action != requested
        or type(observation.get("sequence")) is not int
        or type(observation.get("timestamp_ms")) is not int
        or intent.source_sequence != observation["sequence"]
        or intent.source_timestamp_ms != observation["timestamp_ms"]
        or intent.source_map != observation.get("map_id")
        or not data["policy_evaluated"]
        or not belief["actionable"]
        or not 0 <= now_ms - intent.source_timestamp_ms < runtime.tasks.max_observation_age_ms
        or intent.deadline_ms - intent.created_at_ms > runtime.tasks.intent_ttl_ms
        or intent.deadline_ms > intent.source_timestamp_ms + runtime.tasks.max_observation_age_ms
    ):
        raise ValueError("runtime intent action or observation binding mismatch")
    dispatch = _object(
        data["dispatch"], {"intent_id", "accepted", "reason", "action", "timestamp_ms"}, "dispatch"
    )
    if (
        dispatch["intent_id"] != intent.intent_id
        or type(dispatch["accepted"]) is not bool
        or not isinstance(dispatch["reason"], str)
        or type(dispatch["timestamp_ms"]) is not int
        or dispatch["timestamp_ms"] != now_ms
    ):
        raise ValueError("runtime dispatch identity or time mismatch")
    authorized = SimAction(**dispatch["action"])
    if not dispatch["accepted"]:
        if authorized != SimAction() or data["receipt"] is not None:
            raise ValueError("denied runtime intent has execution receipt")
        return SimAction()
    if (
        intent.intent_id in seen_intents
        or not intent.created_at_ms <= now_ms < intent.deadline_ms
        or authorized != intent.action
    ):
        raise ValueError("replayed, expired or mismatched runtime dispatch")
    seen_intents.add(intent.intent_id)
    receipt = _object(
        data["receipt"],
        {"intent_id", "timestamp_ms", "accepted", "reason", "applied_action"},
        "receipt",
    )
    if (
        receipt["intent_id"] != intent.intent_id
        or type(receipt["timestamp_ms"]) is not int
        or receipt["timestamp_ms"] != now_ms
        or type(receipt["accepted"]) is not bool
        or receipt["accepted"] != execution["accepted"]
        or receipt["reason"] != execution["reason"]
        or _canonical(receipt["applied_action"]) != _canonical(execution["applied_action"])
    ):
        raise ValueError("runtime receipt disagrees with actual execution")
    return authorized


def runtime_summary_fields(runtime: RuntimeConfig | None) -> dict[str, Any]:
    return {} if runtime is None else {"runtime": runtime.manifest()}


def validate_runtime_capabilities(data: dict[str, Any], package: ClassPackage | None) -> None:
    """Bind static class identity without claiming derived beliefs were recomputed."""

    def identity(container: object) -> None:
        if not isinstance(container, dict):
            raise ValueError("invalid runtime capability identity container")
        if package is None:
            if "capabilities" in container:
                raise ValueError("unexpected runtime capability identity without external package")
            return
        value = container.get("capabilities")
        if (
            not isinstance(value, dict)
            or value.get("class_id") != package.class_id
            or value.get("package_sha256") != package.content_sha256
        ):
            raise ValueError("runtime capability identity disagrees with embedded class package")

    def decision(value: object) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError("invalid runtime capability identity decision")
        identity(value.get("evidence"))

    state = data.get("state")
    identity(state)
    assert isinstance(state, dict)
    decision(state.get("last_decision"))
    decision(data.get("decision"))
    decision(data.get("interruption"))
