"""Live observe-decide-dispatch-verify route in the project-owned simulator window."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .agent import AgentDecision, Supervisor, verify_visual_effect
from .class_package import ClassPackage
from .input_bindings import InputBindingProfile, verify_input_profile_confirmation
from .lab import CapabilityRunConfig, _execute_episode, _mission_platform_graphs, canonical, digest
from .map_pack import MapPack
from .mission_program import load_mission_program
from .mission_verification import MissionOracle
from .runtime_pipeline import RuntimeConfig
from .scene import SceneRenderer
from .sim_types import SimAction
from .simulation import Simulator
from .state_estimation import TemporalStateEstimator
from .task_runtime import TaskRuntime
from .vision import PixelPerception


@dataclass(frozen=True)
class LiveRouteStep:
    source_sequence: int
    labels: tuple[str, ...]
    image: Image.Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_row(path: Path, value: object) -> None:
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class OwnedLiveRouteExecution:
    """Produce at most one fresh visual decision and simulator action per call."""

    def __init__(
        self,
        output: Path,
        *,
        package: ClassPackage,
        map_pack: MapPack,
        input_profile: InputBindingProfile,
        input_confirmation: dict[str, Any],
        confirmed_key: str,
    ) -> None:
        confirmed_profile, receipt = verify_input_profile_confirmation(input_confirmation)
        if confirmed_profile != input_profile:
            raise ValueError("live route confirmation disagrees with its input profile")
        if output.exists():
            raise FileExistsError(f"route output already exists: {output}")
        output.mkdir(parents=True)
        (output / "frames").mkdir()

        mission = load_mission_program(
            Path(__file__).with_name("resources")
            / "practice/missions/dead-forest-layered-hunt.json"
        )
        self.runtime = RuntimeConfig(mission=mission)
        mission.bind_map_pack(map_pack)
        self.config = CapabilityRunConfig(seed=17, job=package.class_id, record_stride=1)
        self.output = output
        self.package = package
        self.map_pack = map_pack
        self.input_profile = input_profile
        self.input_confirmation = receipt
        self.confirmed_key = confirmed_key
        self.simulator = Simulator(
            seed=17,
            job=package.class_id,
            map_id=map_pack.entry_map,
            map_pack=map_pack,
            class_package=package,
        )
        self.renderer = SceneRenderer(map_pack=map_pack)
        self.perception = PixelPerception(map_markers=map_pack.markers)
        graphs = _mission_platform_graphs(
            self.runtime, map_pack, self.simulator.class_spec, self.simulator.step_ms
        )
        self.policy = TaskRuntime(
            package.class_id,
            mission.maps,
            config=self.runtime.tasks,
            mission=mission,
            portal_centers={
                (item.id, portal.destination): portal.x + portal.w / 2
                for item in map_pack.maps
                if len(item.portals) > 1
                for portal in item.portals
            },
            platform_graphs=graphs,
            capabilities=self.simulator.capabilities,
        )
        self.supervisor = Supervisor()
        self.estimator = TemporalStateEstimator(self.runtime.state)
        self.oracle = MissionOracle(mission, map_pack)
        self.frame = self.renderer.render(self.simulator.snapshot())
        self.observation = self.perception.observe(self.frame)
        self.belief = self.estimator.update(self.observation, self.simulator.timestamp_ms)
        self.active = True
        self.status = "running"
        self.reason: str | None = None
        self.index = 0
        self.decision_cycles = 0
        self.summary: dict[str, Any] | None = None
        self.review: dict[str, Any] | None = None
        self.casts: list[dict[str, Any]] = []
        self.navigation: list[dict[str, Any]] = []
        self._pending_labels: dict[int, list[str]] = {}
        self._previews: list[Image.Image] = []
        self._input_reasons: Counter[str] = Counter()
        self._accepted_dispatches = 0
        self._initial = self.simulator.save()

        _write_json(output / "initial-state.json", self._initial)
        _write_json(output / "runtime-config.json", self.runtime.manifest())
        _write_json(output / "map-pack.json", map_pack.manifest())
        _write_json(output / "class-package.json", package.manifest())
        _write_json(output / "input-bindings.json", input_profile.manifest())
        _write_json(output / "input-binding-confirmation.json", receipt)
        (output / "trace.jsonl").touch(exist_ok=False)

    @property
    def last_source_sequence(self) -> int | None:
        return None if self.index == 0 else self.simulator.snapshot().sequence - 1

    def step(self) -> LiveRouteStep | None:
        """Run one current-frame decision; terminal decisions do not apply an action."""
        if not self.active:
            raise ValueError("live route execution is not active")
        before = self.simulator.snapshot()
        guard = self.supervisor.inspect(self.observation, before.timestamp_ms)
        if guard.state == "stopped":
            self.policy.interrupt(guard.reason, before.timestamp_ms, terminal=True)
            self.decision_cycles += 1
            self._finalize("failed", guard.reason)
            return None
        if guard.state == "paused":
            runtime_decision = self.policy.interrupt(guard.reason, before.timestamp_ms)
            policy_evaluated = False
        else:
            runtime_decision = self.policy.decide(self.belief, before.timestamp_ms)
            policy_evaluated = True
        self.decision_cycles += 1
        decision = AgentDecision(
            runtime_decision.action,
            runtime_decision.task,
            runtime_decision.state,
            runtime_decision.reason,
        )
        if decision.state in {"failed", "completed", "stopped"}:
            self._finalize(decision.state, decision.reason)
            return None

        dispatch = None
        authorized = SimAction()
        if runtime_decision.intent is not None:
            dispatch = self.policy.scheduler.dispatch(
                runtime_decision.intent, before.timestamp_ms
            )
            if dispatch.accepted:
                authorized = dispatch.action
            else:
                self.policy.interrupt(
                    f"dispatch_denied:{dispatch.reason}", before.timestamp_ms
                )
        applied, input_dispatch = self.input_profile.dispatch(
            authorized, self.package.class_id, self.package
        )
        if input_dispatch is not None:
            self._input_reasons[input_dispatch["reason"]] += 1
            self._accepted_dispatches += int(input_dispatch["accepted"])

        frame_path = self.output / "frames" / f"{before.sequence:06d}.png"
        frame_path.write_bytes(self.frame.png_bytes())
        if len(self._previews) < 600:
            self._previews.append(self.frame.to_image())
        outcome = self.simulator.step(applied)
        receipt = None
        if dispatch is not None and dispatch.accepted:
            self.policy.record_execution(
                dispatch.intent_id, outcome.accepted, outcome.reason, before.timestamp_ms
            )
            receipt = {
                "intent_id": dispatch.intent_id,
                "timestamp_ms": before.timestamp_ms,
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "applied_action": asdict(applied),
            }

        old_witnesses = len(self.oracle.witnesses)
        self.oracle.observe(before, outcome.snapshot, outcome.events)
        for witness in self.oracle.witnesses[old_witnesses:]:
            if witness["phase"] == "navigate":
                item = dict(witness)
                self.navigation.append(item)
                self._pending_labels.setdefault(item["sequence"], []).append("layer_reached")

        following_frame = self.renderer.render(outcome.snapshot)
        following = self.perception.observe(following_frame)
        effect = verify_visual_effect(self.observation, following, decision.action)
        execution = {
            "applied_action": asdict(applied),
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "events": list(outcome.events),
            "input_dispatch": input_dispatch,
        }
        row = {
            "schema_version": 2,
            "sequence": before.sequence,
            "frame_sha256": hashlib.sha256(self.frame.pixels).hexdigest(),
            "observation": asdict(self.observation),
            "guard": asdict(guard),
            "decision": asdict(decision),
            "execution": execution,
            "visual_effect": effect,
            "after_state_sha256": digest(self.simulator.save()),
            "runtime": {
                "schema_version": 1,
                "belief": self.belief.to_dict(),
                "decision": runtime_decision.to_dict(),
                "policy_evaluated": policy_evaluated,
                "dispatch": None if dispatch is None else dispatch.to_dict(),
                "receipt": receipt,
                "state": self.policy.snapshot(),
            },
        }
        _append_row(self.output / "trace.jsonl", row)

        labels = list(self._pending_labels.pop(before.sequence, []))
        if applied.kind == "cast" and input_dispatch is not None and input_dispatch["accepted"]:
            cast = {
                "source_sequence": before.sequence,
                "map_id": self.observation.map_id,
                "key": input_dispatch["key"],
                "skill": input_dispatch["action_skill"],
            }
            self.casts.append(cast)
            labels.append(f"skill_dispatched:{cast['key']}")
        self.index += 1
        image = self.frame.to_image()
        self.frame = following_frame
        self.observation = following
        self.belief = self.estimator.update(following, self.simulator.timestamp_ms)
        return LiveRouteStep(before.sequence, tuple(labels), image)

    def cancel(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason:
            raise ValueError("live route cancellation requires a reason")
        if self.active:
            self.policy.interrupt(reason, self.simulator.timestamp_ms, terminal=True)
            self._finalize("taken_over", reason)
        assert self.summary is not None
        return self.summary

    def _finalize(self, status: str, reason: str | None) -> None:
        self.active = False
        self.status = status
        self.reason = reason
        final = self.simulator.snapshot()
        mission = self.oracle.result(final)
        (self.output / "final.png").write_bytes(self.frame.png_bytes())
        _write_json(self.output / "final-state.json", self.simulator.save())
        if not self._previews:
            self._previews.append(self.frame.to_image())
        first, *rest = self._previews
        first.save(
            self.output / "preview.gif",
            save_all=True,
            append_images=rest,
            duration=self.simulator.step_ms,
            loop=0,
            optimize=False,
        )
        attempts = sum(self._input_reasons.values())
        self.summary = {
            "schema_version": 1,
            "scope": "project_owned_live_closed_loop_execution",
            "status": status,
            "reason": reason,
            "execution_mode": "observe_decide_dispatch_verify_per_step",
            "executed_steps": self.index,
            "decision_cycles": self.decision_cycles,
            "last_source_sequence": self.last_source_sequence,
            "remaining_steps": "not_precomputed",
            "trace_sha256": _sha256(self.output / "trace.jsonl"),
            "final_state_sha256": digest(self.simulator.save()),
            "mission_evidence": mission,
            "casts": len(self.casts),
            "hit_segments": sum(
                row["execution"]["events"].count("attack_hit")
                for row in (
                    json.loads(line)
                    for line in (self.output / "trace.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line
                )
            ),
            "ammo_before": self._initial["player"]["ammo"],
            "ammo_after": final.player.ammo,
            "input_dispatch_summary": {
                "attempts": attempts,
                "accepted": self._accepted_dispatches,
                "denied": attempts - self._accepted_dispatches,
                "reasons": dict(sorted(self._input_reasons.items())),
            },
        }
        _write_json(self.output / "summary.json", self.summary)
        self.review = {
            "schema_version": 2,
            "scope": "project_owned_live_synthetic_route",
            "status": status,
            "input_confirmation_receipt_sha256": self.input_confirmation["receipt_sha256"],
            "profile_id": self.input_profile.profile_id,
            "confirmed_key": self.confirmed_key,
            "navigation": list(self.navigation),
            "casts": list(self.casts),
            "preview": "preview.gif",
            "execution": self.summary,
            "source_files": {
                name: _sha256(self.output / name)
                for name in ("summary.json", "trace.jsonl", "preview.gif")
            },
            "limitations": [
                "each action was decided from the current project-owned simulator frame",
                "no complete action list existed before execution; takeover computes no suffix",
                "the result is not an official-client input or real-game effect",
            ],
        }
        _write_json(self.output / "player-review.json", self.review)


class _PrefixVerified(Exception):
    pass


def verify_live_route_execution(
    output: Path, review: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute every recorded visual decision, dispatch and result from its pixels."""
    if review is None:
        review = json.loads((output / "player-review.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        review.get("scope") != "project_owned_live_synthetic_route"
        or review.get("execution") != summary
        or summary.get("scope") != "project_owned_live_closed_loop_execution"
        or summary.get("status") not in {"completed", "taken_over"}
        or summary.get("remaining_steps") != "not_precomputed"
        or summary.get("trace_sha256") != _sha256(output / "trace.jsonl")
        or summary.get("executed_steps") != len(rows)
    ):
        raise ValueError("invalid live closed-loop route evidence")
    if not rows and not (
        summary["status"] == "taken_over"
        and summary["decision_cycles"] == 0
        and summary["last_source_sequence"] is None
    ):
        raise ValueError("invalid empty live closed-loop route evidence")
    sources = review.get("source_files")
    if not isinstance(sources, dict) or set(sources) != {
        "summary.json",
        "trace.jsonl",
        "preview.gif",
    }:
        raise ValueError("live route source bindings are incomplete")
    for name, expected in sources.items():
        if name not in {"summary.json", "trace.jsonl", "preview.gif"} or _sha256(
            output / name
        ) != expected:
            raise ValueError("live route source binding changed")
    if rows:
        if (
            summary["last_source_sequence"] != rows[-1]["sequence"]
            or summary["final_state_sha256"] != rows[-1]["after_state_sha256"]
            or summary["decision_cycles"]
            != len(rows) + int(summary["status"] == "completed")
        ):
            raise ValueError("live route terminal denominator changed")
    elif summary["final_state_sha256"] != digest(
        json.loads((output / "initial-state.json").read_text(encoding="utf-8"))
    ):
        raise ValueError("empty live route changed simulator state")

    casts = []
    for row in rows:
        dispatch = row["execution"].get("input_dispatch")
        if row["execution"]["applied_action"]["kind"] == "cast" and dispatch is not None:
            casts.append(
                {
                    "source_sequence": row["sequence"],
                    "map_id": row["observation"]["map_id"],
                    "key": dispatch["key"],
                    "skill": dispatch["action_skill"],
                }
            )
    records = [] if not rows else rows[-1]["runtime"]["state"]["navigation"]["records"]
    navigation = [
        {
            "map_id": item["map_id"],
            "phase": "navigate",
            "sequence": item["completed_sequence"],
            "target_platform": item["target_platform"],
        }
        for item in records
        if item["status"] == "completed"
    ]
    if review.get("casts") != casts or review.get("navigation") != navigation:
        raise ValueError("live route player index disagrees with executed evidence")

    package = ClassPackage.from_manifest(
        json.loads((output / "class-package.json").read_text(encoding="utf-8"))
    )
    map_pack = MapPack.from_manifest(
        json.loads((output / "map-pack.json").read_text(encoding="utf-8"))
    )
    profile = InputBindingProfile.from_manifest(
        json.loads((output / "input-bindings.json").read_text(encoding="utf-8"))
    )
    confirmation = json.loads(
        (output / "input-binding-confirmation.json").read_text(encoding="utf-8")
    )
    runtime = RuntimeConfig.from_manifest(
        json.loads((output / "runtime-config.json").read_text(encoding="utf-8"))
    )
    expected_initial = Simulator(
        seed=17,
        job=package.class_id,
        map_id=map_pack.entry_map,
        map_pack=map_pack,
        class_package=package,
    ).save()
    if json.loads((output / "initial-state.json").read_text(encoding="utf-8")) != expected_initial:
        raise ValueError("live route initial state changed")
    checked = 0

    def compare(row: dict[str, Any]) -> None:
        nonlocal checked
        if checked >= len(rows) or canonical(row) != canonical(rows[checked]):
            raise ValueError("live route decision or execution does not reproduce")
        checked += 1
        if summary["status"] == "taken_over" and checked == len(rows):
            raise _PrefixVerified

    result = None
    if not rows:
        return {
            "verified": True,
            "status": summary["status"],
            "executed_steps": 0,
            "decision_cycles": 0,
            "last_source_sequence": None,
            "remaining_steps": "not_precomputed",
            "trace_sha256": summary["trace_sha256"],
        }
    try:
        result = _execute_episode(
            None,
            CapabilityRunConfig(seed=17, job=package.class_id, record_stride=1),
            map_pack=map_pack,
            runtime=runtime,
            class_package=package,
            input_profile=profile,
            input_confirmation=confirmation,
            trace_consumer=compare,
        )
    except _PrefixVerified:
        if summary["status"] != "taken_over":
            raise
    if checked != len(rows):
        raise ValueError("live route verification denominator changed")
    if summary["status"] == "completed" and (
        result is None
        or result["status"] != "completed"
        or result["mission_evidence"] != summary["mission_evidence"]
    ):
        raise ValueError("completed live route terminal does not reproduce")
    return {
        "verified": True,
        "status": summary["status"],
        "executed_steps": len(rows),
        "decision_cycles": summary["decision_cycles"],
        "last_source_sequence": summary["last_source_sequence"],
        "remaining_steps": "not_precomputed",
        "trace_sha256": summary["trace_sha256"],
    }
