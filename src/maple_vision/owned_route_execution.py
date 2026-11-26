"""Interruptible action execution for one verified project-owned route."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .lab import canonical, digest
from .scene import SceneRenderer
from .sim_types import SimAction
from .simulation import Simulator


@dataclass(frozen=True)
class RouteExecutionStep:
    source_sequence: int
    labels: tuple[str, ...]
    image: Image.Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_trace(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if not rows:
        raise ValueError("route trace is empty")
    return rows


def _event_labels(review: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    labels: dict[int, list[str]] = {}
    for item in review.get("navigation", []):
        labels.setdefault(item["sequence"], []).append("layer_reached")
    for item in review.get("casts", []):
        labels.setdefault(item["source_sequence"], []).append(f"skill_dispatched:{item['key']}")
    return {sequence: tuple(value) for sequence, value in labels.items()}


class OwnedRouteExecution:
    """Apply a verified plan one action at a time to a fresh simulator."""

    def __init__(self, output: Path, review: dict[str, Any]) -> None:
        if (
            review.get("scope") != "project_owned_synthetic_route"
            or review.get("status") != "completed"
            or review.get("source_files", {}).get("trace.jsonl") != _sha256(output / "trace.jsonl")
        ):
            raise ValueError("stepwise execution requires one bound completed route")
        if (output / "route-execution.jsonl").exists() or (
            output / "route-execution.json"
        ).exists():
            raise FileExistsError("route execution evidence already exists")
        self.output = output
        self.review = review
        self.trace = _read_trace(output / "trace.jsonl")
        self.simulator = Simulator.restore(
            json.loads((output / "initial-state.json").read_text(encoding="utf-8"))
        )
        self.renderer = SceneRenderer(map_pack=self.simulator.map_pack)
        self.labels = _event_labels(review)
        self.index = 0
        self.active = True
        self.status = "running"
        self.reason: str | None = None
        self.summary: dict[str, Any] | None = None

    @property
    def last_source_sequence(self) -> int | None:
        return None if self.index == 0 else self.trace[self.index - 1]["sequence"]

    def step(self) -> RouteExecutionStep:
        if not self.active:
            raise ValueError("route execution is not active")
        row = self.trace[self.index]
        before = self.simulator.snapshot()
        if row["sequence"] != before.sequence:
            raise ValueError("route trace sequence disagrees with fresh simulator")
        frame = self.renderer.render(before)
        if hashlib.sha256(frame.pixels).hexdigest() != row["frame_sha256"]:
            raise ValueError("route trace frame disagrees with fresh simulator")
        action = SimAction(**row["execution"]["applied_action"])
        outcome = self.simulator.step(action)
        expected = row["execution"]
        if (
            outcome.accepted != expected["accepted"]
            or outcome.reason != expected["reason"]
            or list(outcome.events) != expected["events"]
            or digest(self.simulator.save()) != row["after_state_sha256"]
        ):
            raise ValueError("stepwise route action does not reproduce its plan")
        evidence = {
            "schema_version": 1,
            "source_sequence": before.sequence,
            "applied_action": asdict(action),
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "events": list(outcome.events),
            "before_frame_sha256": row["frame_sha256"],
            "after_state_sha256": row["after_state_sha256"],
        }
        journal = self.output / "route-execution.jsonl"
        with journal.open("ab") as stream:
            stream.write(canonical(evidence) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.index += 1
        if self.index == len(self.trace):
            self._finalize("completed", None)
        return RouteExecutionStep(
            before.sequence,
            self.labels.get(before.sequence, ()),
            frame.to_image(),
        )

    def cancel(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason:
            raise ValueError("route cancellation requires a reason")
        if self.active:
            self._finalize("taken_over", reason)
        assert self.summary is not None
        return self.summary

    def _finalize(self, status: str, reason: str | None) -> None:
        self.active = False
        self.status = status
        self.reason = reason
        journal = self.output / "route-execution.jsonl"
        self.summary = {
            "schema_version": 1,
            "scope": "project_owned_stepwise_simulator_execution",
            "status": status,
            "reason": reason,
            "executed_steps": self.index,
            "planned_steps": len(self.trace),
            "last_source_sequence": self.last_source_sequence,
            "source_trace_sha256": _sha256(self.output / "trace.jsonl"),
            "journal_sha256": _sha256(journal),
            "final_state_sha256": digest(self.simulator.save()),
        }
        (self.output / "route-execution.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def verify_route_execution(output: Path, review: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reexecute only the durably recorded prefix and reject any later action."""
    if review is None:
        review = json.loads((output / "player-review.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "route-execution.json").read_text(encoding="utf-8"))
    if review.get("execution") != summary:
        raise ValueError("player review execution summary changed")
    if (
        summary.get("scope") != "project_owned_stepwise_simulator_execution"
        or summary.get("status") not in {"completed", "taken_over"}
        or summary.get("source_trace_sha256") != _sha256(output / "trace.jsonl")
        or summary.get("journal_sha256") != _sha256(output / "route-execution.jsonl")
    ):
        raise ValueError("invalid stepwise route execution summary")
    planned = _read_trace(output / "trace.jsonl")
    executed = _read_trace(output / "route-execution.jsonl")
    if (
        len(executed) != summary.get("executed_steps")
        or len(planned) != summary.get("planned_steps")
        or not 1 <= len(executed) <= len(planned)
    ):
        raise ValueError("invalid stepwise route execution denominator")
    simulator = Simulator.restore(
        json.loads((output / "initial-state.json").read_text(encoding="utf-8"))
    )
    renderer = SceneRenderer(map_pack=simulator.map_pack)
    for index, item in enumerate(executed):
        source = planned[index]
        before = simulator.snapshot()
        if item["source_sequence"] != source["sequence"] or before.sequence != source["sequence"]:
            raise ValueError("stepwise route execution is not a contiguous prefix")
        frame = renderer.render(before)
        action = SimAction(**item["applied_action"])
        outcome = simulator.step(action)
        if (
            item["applied_action"] != source["execution"]["applied_action"]
            or item["before_frame_sha256"] != source["frame_sha256"]
            or hashlib.sha256(frame.pixels).hexdigest() != item["before_frame_sha256"]
            or item["accepted"] != outcome.accepted
            or item["reason"] != outcome.reason
            or item["events"] != list(outcome.events)
            or item["after_state_sha256"] != digest(simulator.save())
            or item["after_state_sha256"] != source["after_state_sha256"]
        ):
            raise ValueError("stepwise route execution evidence does not reproduce")
    if summary["last_source_sequence"] != executed[-1]["source_sequence"]:
        raise ValueError("stepwise route terminal sequence changed")
    if summary["final_state_sha256"] != digest(simulator.save()):
        raise ValueError("stepwise route final state changed")
    if summary["status"] == "completed" and len(executed) != len(planned):
        raise ValueError("completed stepwise route omitted planned actions")
    if summary["status"] == "taken_over" and len(executed) == len(planned):
        raise ValueError("taken-over stepwise route has no cancelled suffix")
    return {
        "verified": True,
        "status": summary["status"],
        "executed_steps": len(executed),
        "planned_steps": len(planned),
        "last_source_sequence": executed[-1]["source_sequence"],
        "cancelled_steps": len(planned) - len(executed),
        "journal_sha256": summary["journal_sha256"],
    }
