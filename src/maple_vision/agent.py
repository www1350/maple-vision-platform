"""Bounded visual task policy. No simulator object or ground-truth state enters here."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .contracts import Detection, Status
from .scene import PALETTE, ImageFrame
from .sim_types import CLASS_SPECS, ClassSpec, SimAction
from .vision import (
    Detector,
    PixelPerception,
    TerrainSource,
    VisualObservation,
    validate_terrain_source,
)


class HybridPerception:
    """Model objects plus explicit fixture UI; terrain source is selectable.

    The default keeps the original fixture terrain parser. In model mode all
    seven object classes, including platforms and ladders, come only from the
    detector. Neither mode is a general real-game UI parser.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        terrain_source: TerrainSource = "fixture",
        map_markers: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.terrain_source = validate_terrain_source(terrain_source, detector)
        self.detector = detector
        self.hud = PixelPerception(map_markers=map_markers)

    def observe(self, frame: ImageFrame) -> VisualObservation:
        if self.terrain_source == "model":
            context = self.hud.observe_hud(frame)
            detections = self.detector.detect(frame)
            if not isinstance(detections, tuple) or any(
                not isinstance(d, Detection) for d in detections
            ):
                raise ValueError("model detections must be an immutable tuple of Detection values")
            if any(d.class_id not in PALETTE for d in detections):
                raise ValueError("model detection class is not a supported fixture object class")
            if any(d.bbox[2] > frame.width or d.bbox[3] > frame.height for d in detections):
                raise ValueError("model detection box must be inside the input frame")
            source_id = "onnx-all-objects+fixture-hud-v1"
        else:
            context = self.hud.observe(frame)
            objects = tuple(
                d for d in self.detector.detect(frame) if d.class_id not in {"platform", "ladder"}
            )
            terrain = tuple(d for d in context.detections if d.class_id in {"platform", "ladder"})
            detections = objects + terrain
            source_id = "onnx-objects+fixture-terrain-hud-v1"
        players = [item for item in detections if item.class_id == "player"]
        confidence = players[0].confidence if len(players) == 1 else 0.0
        known = len(players) == 1 and context.map_id is not None and context.hp_ratio is not None
        if self.terrain_source == "model":
            known = known and context.mp_ratio is not None
        return replace(
            context,
            detections=detections,
            confidence=confidence,
            status=Status.KNOWN if known else Status.UNKNOWN,
            source_id=source_id,
        )


@dataclass(frozen=True)
class GuardDecision:
    state: str
    reason: str


class Supervisor:
    def __init__(self, max_unknown_steps: int = 12, max_age_ms: int = 250) -> None:
        if type(max_unknown_steps) is not int or not 1 <= max_unknown_steps <= 1000:
            raise ValueError("max_unknown_steps must be in [1, 1000]")
        if type(max_age_ms) is not int or not 0 <= max_age_ms <= 60_000:
            raise ValueError("invalid observation age")
        self.max_unknown_steps = max_unknown_steps
        self.max_age_ms = max_age_ms
        self.unknown_steps = 0
        self.last_sequence = -1
        self.last_timestamp_ms = -1
        self.last_now_ms = -1
        self.stopped = False
        self.reason = ""

    def inspect(self, observation: VisualObservation, now_ms: int) -> GuardDecision:
        if self.stopped:
            return GuardDecision("stopped", self.reason)
        if type(now_ms) is not int or now_ms < self.last_now_ms:
            self.stopped, self.reason = True, "clock_rollback"
            return GuardDecision("stopped", self.reason)
        self.last_now_ms = now_ms
        reason = ""
        if observation.timestamp_ms > now_ms:
            reason = "future_observation"
        elif now_ms - observation.timestamp_ms > self.max_age_ms:
            reason = "stale_observation"
        elif observation.sequence <= self.last_sequence:
            reason = "duplicate_or_reordered_observation"
        elif observation.timestamp_ms < self.last_timestamp_ms:
            reason = "observation_timestamp_rollback"
        elif observation.status is not Status.KNOWN or observation.confidence < 0.5:
            reason = "unknown_observation"
        elif len([d for d in observation.detections if d.class_id == "player"]) != 1:
            reason = "ambiguous_player"
        elif observation.hp_ratio is None or observation.mp_ratio is None:
            reason = "unknown_resources"
        elif observation.hp_ratio <= 0:
            self.stopped, self.reason = True, "player_dead"
            return GuardDecision("stopped", self.reason)
        if reason:
            self.unknown_steps += 1
            if self.unknown_steps >= self.max_unknown_steps:
                self.stopped, self.reason = True, reason
                return GuardDecision("stopped", reason)
            return GuardDecision("paused", reason)
        self.last_sequence = observation.sequence
        self.last_timestamp_ms = observation.timestamp_ms
        self.unknown_steps = 0
        return GuardDecision("running", "fresh_visual_observation")


def center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass(frozen=True)
class AgentDecision:
    action: SimAction
    task: str
    state: str
    reason: str


class VisualAgent:
    """Complete a bounded fixture mission using visible objects, map marker and bars.

    ClassSpec is static fixture knowledge. It contains no simulator state. Mission
    success is checked independently by the validation runner, not by this policy.
    """

    def __init__(
        self,
        job: str = "assassin",
        maps: tuple[str, ...] = ("meadow", "cavern"),
        *,
        class_spec: ClassSpec | None = None,
    ) -> None:
        if job not in CLASS_SPECS:
            raise ValueError("unknown fixture job")
        if (
            not isinstance(maps, tuple)
            or not 1 <= len(maps) <= 8
            or any(
                not isinstance(item, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", item) is None
                for item in maps
            )
            or len(set(maps)) != len(maps)
        ):
            raise ValueError("mission maps must be a tuple of 1..8 distinct bounded map IDs")
        if class_spec is not None and (
            not isinstance(class_spec, ClassSpec) or class_spec.id != job
        ):
            raise ValueError("policy class profile must match job")
        self.spec = CLASS_SPECS[job] if class_spec is None else class_spec
        self.maps = maps
        self.map_index = 0
        self.phase = "fight"
        self.state = "running"
        self._phase_steps = 0
        self._empty_frames = 0
        self._last_attack_ms = -10_000
        self._last_potion_ms = -10_000
        self._last_skill_ms = -10_000
        self._buff_ms = -10_000
        self._jumped_maps: set[str] = set()
        self._last_position: tuple[float, float] | None = None
        self._last_action = SimAction()
        self._facing = 1
        self._stuck_steps = 0
        self.recoveries = 0
        self._descent_direction = 0
        self._last_monster_count: int | None = None
        self._attack_attempts = 0

    def _result(self, action: SimAction, reason: str) -> AgentDecision:
        self._last_action = action
        if self.phase == "fight" and action.kind in {"attack", "jump_attack"}:
            self._attack_attempts += 1
        if action.direction and action.kind in {"move", "attack", "jump", "jump_attack"}:
            self._facing = action.direction
        return AgentDecision(action, self.phase, self.state, reason)

    def _advance(self, phase: str) -> None:
        self.phase, self._phase_steps, self._empty_frames = phase, 0, 0

    def decide(self, observation: VisualObservation) -> AgentDecision:
        if self.state != "running":
            return self._result(SimAction(), self.state)
        players = [d for d in observation.detections if d.class_id == "player"]
        if observation.status is not Status.KNOWN or len(players) != 1:
            return self._result(SimAction(), "waiting_for_player")
        if observation.map_id != self.maps[self.map_index]:
            if (
                self.phase == "travel"
                and self.map_index + 1 < len(self.maps)
                and observation.map_id == self.maps[self.map_index + 1]
            ):
                self.map_index += 1
                self._advance("fight")
            else:
                self.state = "failed"
                return self._result(SimAction(), "unexpected_map")
        self._phase_steps += 1
        if self._phase_steps > 1000:
            self.state = "failed"
            return self._result(SimAction(), "task_timeout")
        player = players[0]
        px, py = center(player)
        position = (px, py)
        if self._last_action.kind == "move" and self._last_position is not None:
            distance = abs(px - self._last_position[0]) + abs(py - self._last_position[1])
            self._stuck_steps = self._stuck_steps + 1 if distance < 0.5 else 0
        self._last_position = position
        if self._stuck_steps >= 8:
            self._stuck_steps = 0
            if self.recoveries >= 2:
                self.state = "failed"
                return self._result(SimAction(), "recovery_budget_exhausted")
            self.recoveries += 1
            return self._result(
                SimAction("jump", self._last_action.direction), "bounded_stuck_recovery"
            )
        now = observation.timestamp_ms
        hp, mp = observation.hp_ratio, observation.mp_ratio
        if (
            hp is not None
            and hp < 0.5
            and now - self._last_skill_ms >= 500
            and self.spec.heal_amount
            and mp is not None
            and mp > 0.2
        ):
            self._last_skill_ms = now
            return self._result(SimAction("heal"), "health_below_threshold")
        if now - self._last_potion_ms >= 500:
            if hp is not None and hp < 0.4:
                self._last_potion_ms = now
                return self._result(SimAction("potion_hp"), "health_below_threshold")
            if mp is not None and mp < 0.25:
                self._last_potion_ms = now
                return self._result(SimAction("potion_mp"), "mana_below_threshold")
        if self.phase == "fight":
            monsters = [d for d in observation.detections if d.class_id == "monster"]
            if len(monsters) != self._last_monster_count:
                self._last_monster_count = len(monsters)
                self._attack_attempts = 0
            if self._attack_attempts >= 12:
                self.state = "failed"
                return self._result(SimAction(), "combat_progress_timeout")
            if not monsters:
                self._empty_frames += 1
                if self._empty_frames >= 4:
                    self._advance("collect")
                return self._result(SimAction(), "confirming_clear_scene")
            self._empty_frames = 0
            if (
                now - self._buff_ms >= 5000
                and now - self._last_skill_ms >= 500
                and mp is not None
                and mp > 0.3
            ):
                self._buff_ms = self._last_skill_ms = now
                return self._result(SimAction("buff"), "maintain_fixture_buff")
            target = min(monsters, key=lambda item: abs(center(item)[0] - px))
            tx, ty = center(target)
            direction = 1 if tx >= px else -1
            reach = 65 if self.spec.id == "dragon_knight" else self.spec.attack_range * 0.9
            if self._descent_direction and player.bbox[3] < target.bbox[3] - 2:
                return self._result(
                    SimAction("move", self._descent_direction), "finish_platform_descent"
                )
            self._descent_direction = 0
            if abs(ty - py) > 38:
                supports = [
                    d
                    for d in observation.detections
                    if d.class_id == "platform"
                    and abs(player.bbox[3] - d.bbox[1]) <= 2
                    and d.bbox[0] < player.bbox[2]
                    and d.bbox[2] > player.bbox[0]
                ]
                if supports and player.bbox[3] < target.bbox[3] - 2:
                    same_level = [
                        d
                        for d in observation.detections
                        if d.class_id == "platform" and abs(d.bbox[1] - supports[0].bbox[1]) < 2
                    ]
                    left, right = (
                        min(d.bbox[0] for d in same_level),
                        max(d.bbox[2] for d in same_level),
                    )
                    self._descent_direction = -1 if px - left < right - px else 1
                    return self._result(
                        SimAction("move", self._descent_direction), "leave_elevated_platform"
                    )
                return self._result(SimAction(), "wait_for_jump_descent")
            if abs(tx - px) > reach:
                return self._result(SimAction("move", direction), "approach_visible_monster")
            if now - self._last_attack_ms < self.spec.attack_cooldown_ms:
                return self._result(SimAction(), "attack_cooldown")
            self._last_attack_ms = now
            # A stationary first jump avoids inventing a second airborne jump.
            on_ground = any(
                d.class_id == "platform"
                and d.bbox[0] <= px <= d.bbox[2]
                and abs(player.bbox[3] - d.bbox[1]) <= 4
                for d in observation.detections
            )
            if (
                self.spec.id == "assassin"
                and observation.map_id not in self._jumped_maps
                and on_ground
            ):
                # Face first; the jump itself has no horizontal velocity.
                if self._facing != direction:
                    return self._result(SimAction("attack", direction), "face_visible_monster")
                self._jumped_maps.add(observation.map_id)
                return self._result(SimAction("jump_attack"), "fixture_jump_attack")
            skill = "area" if self.spec.id == "dragon_knight" else "basic"
            return self._result(SimAction("attack", direction, skill), "attack_visible_monster")
        if self.phase == "collect":
            drops = [d for d in observation.detections if d.class_id == "loot"]
            if not drops:
                self._empty_frames += 1
                if self._empty_frames >= 4:
                    self._advance("talk")
                return self._result(SimAction(), "confirming_no_visible_loot")
            self._empty_frames = 0
            target = min(drops, key=lambda item: abs(center(item)[0] - px))
            tx, _ = center(target)
            if abs(tx - px) <= 20:
                return self._result(SimAction("loot"), "collect_visible_drop")
            return self._result(SimAction("move", 1 if tx > px else -1), "approach_visible_drop")
        if self.phase == "talk":
            if observation.dialogue_visible:
                if self.map_index == len(self.maps) - 1:
                    self.state = "completed"
                    self._advance("done")
                else:
                    self._advance("travel")
                return self._result(SimAction(), "dialogue_observed")
            npcs = [d for d in observation.detections if d.class_id == "npc"]
            if npcs:
                tx, _ = center(min(npcs, key=lambda item: abs(center(item)[0] - px)))
                if abs(tx - px) <= 25:
                    return self._result(SimAction("interact"), "interact_with_visible_npc")
                return self._result(SimAction("move", 1 if tx > px else -1), "approach_visible_npc")
        if self.phase == "travel":
            portals = [d for d in observation.detections if d.class_id == "portal"]
            if portals:
                tx, _ = center(min(portals, key=lambda item: abs(center(item)[0] - px)))
                if abs(tx - px) <= 22:
                    return self._result(SimAction("portal"), "enter_visible_portal")
                return self._result(
                    SimAction("move", 1 if tx > px else -1), "approach_visible_portal"
                )
        return self._result(SimAction(), "waiting_for_task_target")


def verify_visual_effect(
    before: VisualObservation, after: VisualObservation, action: SimAction
) -> str:
    """Only claim an observed change. This does not prove hidden combat/loot semantics."""
    if before.status is not Status.KNOWN or after.status is not Status.KNOWN:
        return "unknown"
    if after.sequence <= before.sequence or after.timestamp_ms < before.timestamp_ms:
        return "unknown"
    if action.kind == "idle":
        return "not_applicable"
    if action.kind == "portal":
        if before.map_id is None or after.map_id is None:
            return "unknown"
        return (
            "observed"
            if before.map_id is not None
            and after.map_id is not None
            and before.map_id != after.map_id
            else "pending"
        )
    if action.kind == "interact":
        return "observed" if not before.dialogue_visible and after.dialogue_visible else "pending"
    if action.kind in {"potion_hp", "heal"}:
        if (
            before.hp_ratio is not None
            and after.hp_ratio is not None
            and after.hp_ratio > before.hp_ratio
        ):
            return "observed"
        return "pending"
    if action.kind == "potion_mp":
        if (
            before.mp_ratio is not None
            and after.mp_ratio is not None
            and after.mp_ratio > before.mp_ratio
        ):
            return "observed"
        return "pending"
    if action.kind in {"move", "jump", "jump_attack", "climb"}:
        first = [d for d in before.detections if d.class_id == "player"]
        second = [d for d in after.detections if d.class_id == "player"]
        if len(first) == len(second) == 1 and center(first[0]) != center(second[0]):
            return "observed"
    # Sprite disappearance can be caused by occlusion; never label it a verified kill.
    return "unknown"
