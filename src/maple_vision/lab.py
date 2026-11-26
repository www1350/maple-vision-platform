"""Offline closed-loop validation, fault injection and deterministic failure replay."""

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .agent import AgentDecision, HybridPerception, Supervisor, VisualAgent, verify_visual_effect
from .animation_pack import MAX_SOURCE_BYTES
from .catalog import strict_json
from .class_package import MAX_CLASS_PACKAGE_BYTES, ClassPackage
from .input_bindings import MAX_INPUT_BINDINGS_BYTES, InputBindingProfile
from .map_pack import MapPack
from .mission_verification import MissionOracle, compare_branch_decisions
from .navigation_graph import (
    MovementCapabilities,
    PlatformGeometry,
    PlatformGraph,
    build_platform_graph,
)
from .onnx_detector import OnnxDetector
from .profiles import LoadedClassProfile, parse_loaded_class_profile
from .runtime_evidence import (
    load_run_runtime,
    runtime_summary_fields,
    validate_runtime_capabilities,
    validate_runtime_step,
)
from .runtime_pipeline import MAX_RUNTIME_CONFIG_BYTES, RuntimeConfig
from .scene import PALETTE, ImageFrame, SceneRenderer
from .scene_animation import MAX_SCENE_ANIMATION_BYTES, SceneAnimation, load_scene_animation
from .sim_types import CLASS_SPECS, RULESET, ClassSpec, SimAction
from .simulation import Simulator
from .sprite_pack import SpritePack, verify_dataset_sprite_pack
from .state_estimation import BeliefState, TemporalStateEstimator
from .task_runtime import RuntimeDecision, TaskRuntime
from .validation_resources import ProcessResourceMonitor
from .vision import Detector, PixelPerception, TerrainSource, validate_terrain_source

FAULTS = (
    "none",
    "brief_dropout",
    "persistent_dropout",
    "duplicate_frames",
    "freeze_input",
    "loot_false_negative",
)
_MAX_MAP_EVIDENCE_BYTES = 1024 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def runtime_identity() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in ("maple-vision-platform", "Pillow", "numpy", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": digest(sources),
    }


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    job: str = "assassin"
    max_steps: int = 1000
    fault: str = "none"
    record_stride: int = 4

    def __post_init__(self) -> None:
        self._validate_fields()
        if self.job not in CLASS_SPECS:
            raise ValueError("unknown job or fault")

    def _validate_fields(self) -> None:
        for name, value, low, high in (
            ("seed", self.seed, 0, 2**32 - 1),
            ("max_steps", self.max_steps, 1, 100_000),
            ("record_stride", self.record_stride, 1, 1000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.fault not in FAULTS:
            raise ValueError("unknown job or fault")


@dataclass(frozen=True)
class CapabilityRunConfig(RunConfig):
    """External class identity requires a separately supplied verified package."""

    def __post_init__(self) -> None:
        self._validate_fields()
        if (
            not isinstance(self.job, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", self.job) is None
        ):
            raise ValueError("invalid external class identity")


def _recorded_config(summary: dict[str, Any]) -> RunConfig:
    factory = CapabilityRunConfig if "class_package" in summary else RunConfig
    return factory(**summary["config"])


def _class_jobs(
    packages: tuple[ClassPackage, ...] | None,
) -> tuple[tuple[str, ClassPackage | None], ...]:
    if packages is None:
        return tuple((job, None) for job in CLASS_SPECS)
    if (
        not isinstance(packages, tuple)
        or not 1 <= len(packages) <= 16
        or any(type(package) is not ClassPackage for package in packages)
        or len({package.class_id for package in packages}) != len(packages)
    ):
        raise ValueError("class packages must be 1..16 unique verified external classes")
    return tuple((package.class_id, package) for package in packages)


class _Frames:
    """Fault adapter has no access to policy. A repeated image retains its old identity."""

    def __init__(self, fault: str) -> None:
        self.fault = fault
        self.frozen: ImageFrame | None = None

    def apply(self, frame: ImageFrame) -> ImageFrame:
        if frame.sequence == 19:
            self.frozen = frame
        if (
            self.fault == "brief_dropout"
            and 20 <= frame.sequence < 26
            or self.fault == "persistent_dropout"
            and frame.sequence >= 20
        ):
            return ImageFrame(
                frame.sequence,
                frame.timestamp_ms,
                frame.width,
                frame.height,
                bytes(len(frame.pixels)),
            )
        if (
            self.fault == "duplicate_frames"
            and 20 <= frame.sequence < 26
            and self.frozen is not None
        ):
            return self.frozen
        if self.fault == "loot_false_negative":
            pixels = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
            mask = np.all(pixels == PALETTE["loot"], axis=2)
            if np.any(mask):
                hidden = pixels.copy()
                hidden[mask] = (0, 0, 0)
                return ImageFrame(
                    frame.sequence,
                    frame.timestamp_ms,
                    frame.width,
                    frame.height,
                    hidden.tobytes(),
                )
        return frame


def run_episode(
    output: Path | None = None,
    config: RunConfig | None = None,
    detector: Detector | None = None,
    *,
    class_profile: LoadedClassProfile | None = None,
    sprite_pack: SpritePack | None = None,
    map_pack: MapPack | None = None,
    terrain_source: TerrainSource = "fixture",
    runtime: RuntimeConfig | None = None,
    animation: SceneAnimation | None = None,
    class_package: ClassPackage | None = None,
    input_profile: InputBindingProfile | None = None,
) -> dict[str, Any]:
    return _execute_episode(
        output,
        config,
        detector,
        class_profile=class_profile,
        sprite_pack=sprite_pack,
        map_pack=map_pack,
        terrain_source=terrain_source,
        runtime=runtime,
        animation=animation,
        class_package=class_package,
        input_profile=input_profile,
    )


def _perception_settings(
    detector: Detector | None,
    *,
    terrain_source: TerrainSource = "fixture",
    map_pack: MapPack | None = None,
) -> dict[str, Any]:
    validate_terrain_source(terrain_source, detector)
    if detector is None:
        settings: dict[str, Any] = {
            "pipeline": "pixel-fixture",
            "implementation": "maple_vision.vision.PixelPerception",
            "model_manifest": None,
            "loaded_model_sha256": None,
            "inference_options": None,
            "verifiable_model_binding": True,
        }
        if map_pack is not None:
            settings["map_pack_sha256"] = map_pack.content_sha256
        return settings
    native = detector if type(detector) is OnnxDetector else None
    if native is not None and native.loaded_model_sha256 != native.manifest.model_sha256:
        raise ValueError("model manifest is not bound to the loaded model hash")
    settings = {
        "pipeline": "onnx-all-objects+fixture-hud-v1"
        if terrain_source == "model"
        else "onnx-objects+fixture-terrain-hud-v1",
        "implementation": f"{type(detector).__module__}.{type(detector).__qualname__}",
        "model_manifest": asdict(detector.manifest) if hasattr(detector, "manifest") else None,
        "loaded_model_sha256": native.loaded_model_sha256 if native is not None else None,
        "inference_options": {
            "confidence_threshold": native.confidence_threshold,
            "iou_threshold": native.iou_threshold,
        }
        if native is not None
        else None,
        "verifiable_model_binding": native is not None,
    }
    if terrain_source == "model":
        # Preserve the default v0.2 settings byte-for-byte. The opt-in source has
        # both a new pipeline identity and an explicit, checked source setting.
        settings["terrain_source"] = terrain_source
    if map_pack is not None:
        settings["map_pack_sha256"] = map_pack.content_sha256
    return settings


def _mission_requirements(map_pack: MapPack | None) -> tuple[tuple[str, ...], int]:
    """Static oracle expectations; hidden world state never enters the visual policy."""
    if map_pack is None:
        return ("meadow", "cavern"), 6
    return map_pack.map_ids, sum(len(item.monster_x) for item in map_pack.maps)


def _mission_platform_graphs(
    runtime: RuntimeConfig | None,
    map_pack: MapPack | None,
    class_spec: object,
    step_ms: int,
) -> dict[str, PlatformGraph]:
    if runtime is None or runtime.mission is None:
        return {}
    stages = tuple(stage for stage in runtime.mission.stages if stage.navigation is not None)
    if not stages:
        return {}
    if map_pack is None:
        raise ValueError("navigation missions require an external map pack")
    walk_speed = getattr(class_spec, "walk_speed", None)
    jump_speed = getattr(class_spec, "jump_speed", None)
    if type(walk_speed) is not int or type(jump_speed) is not int:
        raise ValueError("navigation mission class lacks bounded movement capabilities")
    graphs = {}
    for stage in stages:
        assert stage.navigation is not None
        capabilities = MovementCapabilities(
            walk_speed=walk_speed,
            jump_speed=jump_speed,
            step_ms=step_ms,
            can_jump=stage.navigation.required_edge_kind != "ladder",
        )
        graphs[stage.map_id] = build_platform_graph(
            PlatformGeometry.from_map_pack(map_pack, stage.map_id), capabilities
        )
    return graphs


def _episode_budget(map_pack: MapPack | None, *, fault: bool = False) -> int:
    if map_pack is None:
        return 500 if fault else 1000
    return max(1000, 500 * len(map_pack.map_ids))


def _execute_episode(
    output: Path | None = None,
    config: RunConfig | None = None,
    detector: Detector | None = None,
    *,
    class_profile: LoadedClassProfile | None = None,
    sprite_pack: SpritePack | None = None,
    map_pack: MapPack | None = None,
    initial_state: dict[str, Any] | None = None,
    trace_consumer: Callable[[dict[str, Any]], None] | None = None,
    terrain_source: TerrainSource = "fixture",
    runtime: RuntimeConfig | None = None,
    animation: SceneAnimation | None = None,
    class_package: ClassPackage | None = None,
    input_profile: InputBindingProfile | None = None,
) -> dict[str, Any]:
    if runtime is not None and type(runtime) is not RuntimeConfig:
        raise ValueError("runtime must be a structured RuntimeConfig or None")
    if class_package is not None and (
        type(class_package) is not ClassPackage or class_profile is not None or runtime is None
    ):
        raise ValueError("external class packages require structured runtime and no legacy profile")
    if input_profile is not None and type(input_profile) is not InputBindingProfile:
        raise ValueError("input profile must be a verified InputBindingProfile or None")
    config = config or (
        RunConfig() if class_package is None else CapabilityRunConfig(job=class_package.class_id)
    )
    bundle_schema = 1 if runtime is None else 2
    spec = None if class_profile is None else class_profile.class_spec
    simulator = (
        Simulator(
            config.seed,
            config.job,
            map_id="meadow" if map_pack is None else map_pack.entry_map,
            class_spec=spec,
            map_pack=map_pack,
            class_package=class_package,
        )
        if initial_state is None
        else Simulator.restore(initial_state)
    )
    if map_pack is not None and simulator.map_pack != map_pack:
        raise ValueError("requested map pack disagrees with embedded simulator content")
    if simulator.class_package != class_package:
        raise ValueError("requested class package disagrees with embedded simulator content")
    map_pack = simulator.map_pack
    settings = _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    mission_maps, expected_drops = _mission_requirements(map_pack)
    if runtime is None and map_pack is not None and any(len(m.portals) > 1 for m in map_pack.maps):
        raise ValueError("branching map packs require the structured runtime")
    mission_oracle = None
    if runtime is not None and runtime.mission is not None:
        runtime.mission.bind_map_pack(map_pack)
        mission_oracle = MissionOracle(runtime.mission, map_pack)
    markers = None if map_pack is None else map_pack.markers
    renderer = SceneRenderer(sprite_pack=sprite_pack, map_pack=map_pack, animation=animation)
    fault_frames = _Frames(config.fault)
    if output is not None:
        if output.exists():
            raise FileExistsError(f"run output already exists: {output}")
        output.mkdir(parents=True)
        (output / "frames").mkdir()
    perception = (
        PixelPerception(map_markers=markers)
        if detector is None
        else HybridPerception(detector, terrain_source=terrain_source, map_markers=markers)
    )
    platform_graphs = _mission_platform_graphs(
        runtime, map_pack, simulator.class_spec, simulator.step_ms
    )
    policy: VisualAgent | TaskRuntime
    policy = (
        VisualAgent(config.job, mission_maps, class_spec=spec)
        if runtime is None
        else TaskRuntime(
            config.job,
            runtime.mission.maps if runtime.mission is not None else mission_maps,
            class_spec=spec,
            config=runtime.tasks,
            mission=runtime.mission,
            portal_centers={
                (m.id, p.destination): p.x + p.w / 2
                for m in map_pack.maps
                if len(m.portals) > 1
                for p in m.portals
            }
            if map_pack is not None
            else None,
            platform_graphs=platform_graphs,
            **({"capabilities": simulator.capabilities} if class_package is not None else {}),
        )
    )
    supervisor = Supervisor()
    estimator = None if runtime is None else TemporalStateEstimator(runtime.state)
    initial = simulator.save()
    if output:
        _write_json(output / "initial-state.json", initial)
        if runtime is not None:
            _write_json(output / "runtime-config.json", runtime.manifest())
        if map_pack is not None:
            _write_json(output / "map-pack.json", map_pack.manifest())
        if sprite_pack is not None:
            sprite_pack.write_sources(output)
            _write_json(output / "sprite-pack.json", sprite_pack.manifest())
        if animation is not None:
            animation.write_sources(output)
        if class_package is not None:
            _write_json(output / "class-package.json", class_package.manifest())
        if input_profile is not None:
            _write_json(output / "input-bindings.json", input_profile.manifest())
    trace: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    actions: Counter[str] = Counter()
    events: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    input_dispatch_reasons: Counter[str] = Counter()
    input_dispatch_accepted = 0
    paused_steps = 0
    visited: set[str] = set()
    dialogue_maps: set[str] = set()
    collected_drops: set[tuple[str, str]] = set()
    cleared: dict[str, bool] = {}
    wall_start = time.perf_counter()
    frame = fault_frames.apply(renderer.render(simulator.snapshot()))
    observation = perception.observe(frame)
    belief: BeliefState | None = (
        None if estimator is None else estimator.update(observation, simulator.timestamp_ms)
    )
    runtime_decision: RuntimeDecision | None = None
    runtime_interruption: RuntimeDecision | None = None
    runtime_policy_evaluated = False
    status, reason = "failed", "step_budget_exhausted"
    terminal: dict[str, Any] = {
        "cause": "step_budget_exhausted",
        "guard": None,
        "decision": None,
    }
    for _ in range(config.max_steps):
        before = simulator.snapshot()
        visited.add(before.map_id)
        cleared[before.map_id] = not before.monsters and not before.drops
        guard = supervisor.inspect(observation, before.timestamp_ms)
        if guard.state == "stopped":
            if isinstance(policy, TaskRuntime):
                runtime_interruption = policy.interrupt(
                    guard.reason, before.timestamp_ms, terminal=True
                )
                runtime_decision = None
                runtime_policy_evaluated = False
            status, reason = "stopped", guard.reason
            terminal = {"cause": "supervisor", "guard": asdict(guard), "decision": None}
            break
        if guard.state == "paused":
            paused_steps += 1
            if isinstance(policy, TaskRuntime):
                runtime_decision = policy.interrupt(guard.reason, before.timestamp_ms)
                decision = AgentDecision(
                    runtime_decision.action,
                    runtime_decision.task,
                    runtime_decision.state,
                    runtime_decision.reason,
                )
            else:
                decision = AgentDecision(SimAction(), policy.phase, "paused", guard.reason)
            runtime_policy_evaluated = False
        else:
            if isinstance(policy, TaskRuntime):
                assert belief is not None
                runtime_decision = policy.decide(belief, before.timestamp_ms)
                decision = AgentDecision(
                    runtime_decision.action,
                    runtime_decision.task,
                    runtime_decision.state,
                    runtime_decision.reason,
                )
                runtime_policy_evaluated = True
            else:
                decision = policy.decide(observation)
        if decision.state in {"failed", "completed", "stopped"}:
            status, reason = decision.state, decision.reason
            terminal = {
                "cause": "policy",
                "guard": asdict(guard),
                "decision": asdict(decision),
            }
            break
        action = decision.action
        dispatch = None
        authorized_action = action
        if isinstance(policy, TaskRuntime):
            assert runtime_decision is not None
            authorized_action = SimAction()
            if runtime_decision.intent is not None:
                dispatch = policy.scheduler.dispatch(runtime_decision.intent, before.timestamp_ms)
                if dispatch.accepted:
                    authorized_action = dispatch.action
                else:
                    policy.interrupt(f"dispatch_denied:{dispatch.reason}", before.timestamp_ms)
        input_action = authorized_action
        input_dispatch = None
        if input_profile is not None:
            input_action, input_dispatch = input_profile.dispatch(
                authorized_action, config.job, class_package
            )
            if input_dispatch is not None:
                input_dispatch_reasons[input_dispatch["reason"]] += 1
                input_dispatch_accepted += int(input_dispatch["accepted"])
        applied = (
            SimAction()
            if config.fault == "freeze_input" and before.sequence >= 20
            else input_action
        )
        actions[action.kind] += 1
        if output and before.sequence % config.record_stride == 0:
            (output / "frames" / f"{before.sequence:06d}.png").write_bytes(frame.png_bytes())
            if len(previews) < 600:
                previews.append(frame.to_image())
        outcome = simulator.step(applied)
        receipt: dict[str, Any] | None = None
        if isinstance(policy, TaskRuntime) and dispatch is not None and dispatch.accepted:
            policy.record_execution(
                dispatch.intent_id, outcome.accepted, outcome.reason, before.timestamp_ms
            )
            receipt = {
                "intent_id": dispatch.intent_id,
                "timestamp_ms": before.timestamp_ms,
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "applied_action": asdict(applied),
            }
        events.update(outcome.events)
        if mission_oracle is not None:
            mission_oracle.observe(before, outcome.snapshot, outcome.events)
        if "loot" in outcome.events and before.map_id == outcome.snapshot.map_id:
            remaining = {drop.id for drop in outcome.snapshot.drops}
            collected_drops.update(
                (before.map_id, drop.id) for drop in before.drops if drop.id not in remaining
            )
        if "dialogue" in outcome.events:
            dialogue_maps.add(before.map_id)
        following_frame = fault_frames.apply(renderer.render(outcome.snapshot))
        following = perception.observe(following_frame)
        effect = verify_visual_effect(observation, following, action)
        effects[effect] += 1
        execution_evidence = {
            "applied_action": asdict(applied),
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "events": list(outcome.events),
        }
        if input_profile is not None:
            execution_evidence["input_dispatch"] = input_dispatch
        row = {
            "schema_version": bundle_schema,
            "sequence": before.sequence,
            "frame_sha256": hashlib.sha256(frame.pixels).hexdigest(),
            "observation": asdict(observation),
            "guard": asdict(guard),
            "decision": asdict(decision),
            "execution": execution_evidence,
            "visual_effect": effect,
            "after_state_sha256": digest(simulator.save()),
        }
        if isinstance(policy, TaskRuntime):
            assert belief is not None and runtime_decision is not None
            row["runtime"] = {
                "schema_version": 1,
                "belief": belief.to_dict(),
                "decision": runtime_decision.to_dict(),
                "policy_evaluated": runtime_policy_evaluated,
                "dispatch": None if dispatch is None else dispatch.to_dict(),
                "receipt": receipt,
                "state": policy.snapshot(),
            }
        if output:
            trace.append(row)
        if trace_consumer is not None:
            trace_consumer(row)
        observation, frame = following, following_frame
        if estimator is not None:
            belief = estimator.update(following, simulator.timestamp_ms)
    if isinstance(policy, TaskRuntime) and terminal["cause"] == "step_budget_exhausted":
        runtime_interruption = policy.interrupt(reason, simulator.timestamp_ms, terminal=True)
        runtime_decision = None
        runtime_policy_evaluated = False
    final = simulator.snapshot()
    visited.add(final.map_id)
    cleared[final.map_id] = not final.monsters and not final.drops
    oracle_pass = (
        final.player.alive
        and set(cleared) == set(mission_maps)
        and all(cleared.values())
        and dialogue_maps == set(mission_maps)
        and events["monster_defeated"] == expected_drops
        and len(collected_drops) == expected_drops
    )
    mission_result = None if mission_oracle is None else mission_oracle.result(final)
    branch_check = None
    if mission_result is not None:
        oracle_pass = mission_result["completed"]
        if mission_oracle is not None and mission_oracle.program.conditional:
            assert isinstance(policy, TaskRuntime)
            runtime_branches = policy.snapshot()["mission_branching"]["decisions"]
            branch_check = compare_branch_decisions(runtime_branches, mission_result["branches"])
            oracle_pass = oracle_pass and branch_check["matched"]
    if status == "completed" and not oracle_pass:
        status = "failed"
        reason = (
            "conditional_branch_check_failed"
            if branch_check is not None and not branch_check["matched"]
            else "independent_goal_check_failed"
        )
    elapsed = time.perf_counter() - wall_start
    if canonical(settings) != canonical(
        _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    ):
        raise ValueError("detector settings changed during the episode")
    summary: dict[str, Any] = {
        "schema_version": bundle_schema,
        "scope": "synthetic_fixture",
        "ruleset": RULESET,
        "config": asdict(config),
        "detector": "pixel-fixture" if detector is None else "onnx",
        "status": status,
        "reason": reason,
        "steps": final.sequence,
        "simulated_seconds": final.timestamp_ms / 1000,
        "wall_seconds": round(elapsed, 6),
        "paused_steps": paused_steps,
        "recoveries": policy.recoveries,
        "actions": dict(sorted(actions.items())),
        "events": dict(sorted(events.items())),
        "visual_effects": dict(sorted(effects.items())),
        "independent_goal_check": oracle_pass,
        "maps_visited": sorted(visited),
        "dialogue_maps": sorted(dialogue_maps),
        "final_coins": final.player.coins,
        "drops_collected": len(collected_drops),
        "alive": final.player.alive,
        "final_state_sha256": digest(simulator.save()),
        "python": platform.python_version(),
        "runtime_identity": runtime_identity(),
        "model_manifest": settings["model_manifest"],
        "class_profile": None if class_profile is None else asdict(class_profile),
        "perception_evidence": {
            "schema_version": 1,
            "settings": settings,
            "terminal": {**terminal, "observation": asdict(observation)},
        },
    }
    if mission_result is not None:
        summary["mission_evidence"] = mission_result
    if branch_check is not None:
        summary["conditional_branch_check"] = branch_check

    if runtime is not None:
        assert isinstance(policy, TaskRuntime) and belief is not None
        summary["runtime_evidence"] = {
            "schema_version": 1,
            "settings": runtime.manifest(),
            "terminal": {
                "schema_version": 1,
                "belief": belief.to_dict(),
                "decision": None if runtime_decision is None else runtime_decision.to_dict(),
                "policy_evaluated": runtime_policy_evaluated,
                "interruption": None
                if runtime_interruption is None
                else runtime_interruption.to_dict(),
                "state": policy.snapshot(),
            },
        }
    if sprite_pack is not None:
        summary["sprite_pack"] = sprite_pack.manifest()
    if animation is not None:
        summary["animation"] = animation.manifest()
    if class_package is not None:
        summary["class_package"] = class_package.manifest()
    if input_profile is not None:
        summary["input_profile"] = input_profile.manifest()
        attempts = sum(input_dispatch_reasons.values())
        summary["input_dispatch_summary"] = {
            "scope": "offline_simulated_skill_dispatch",
            "attempts": attempts,
            "accepted": input_dispatch_accepted,
            "denied": attempts - input_dispatch_accepted,
            "reasons": dict(sorted(input_dispatch_reasons.items())),
        }
    if map_pack is not None:
        summary["map_pack"] = map_pack.manifest()
    if output:
        trace_path = output / "trace.jsonl"
        with trace_path.open("x", encoding="utf-8") as stream:
            for row in trace:
                stream.write(canonical(row).decode() + "\n")
        _write_json(output / "final-state.json", simulator.save())
        _write_json(output / "summary.json", summary)
        (output / "final.png").write_bytes(frame.png_bytes())
        if previews:
            previews[0].save(
                output / "preview.gif",
                save_all=True,
                append_images=previews[1:],
                duration=config.record_stride * simulator.step_ms,
                loop=0,
                optimize=False,
            )
        files = {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": bundle_schema,
            "kind": "offline-run-bundle",
            "files": files,
            "recording": "sampled_png_and_preview_gif; full_action_trace",
            "replay_scope": (
                "deterministic_simulator_action_replay; not_counterfactual_video_control"
            ),
        }
        if runtime is not None:
            manifest["runtime"] = runtime.manifest()
        _write_json(output / "manifest.json", manifest)
        summary["output"] = str(output)
    return summary


def _run_evidence_path(output: Path, relative: object) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("invalid run path")
    target = (output / relative).resolve()
    if (
        not target.is_relative_to(output.resolve())
        or ".." in Path(relative).parts
        or Path(relative).is_absolute()
    ):
        raise ValueError("run evidence path escapes directory")
    return target


def _read_run_evidence(output: Path, relative: str) -> bytes:
    target = _run_evidence_path(output, relative)
    budgets = {
        "map-pack.json": _MAX_MAP_EVIDENCE_BYTES,
        "runtime-config.json": MAX_RUNTIME_CONFIG_BYTES,
        "scene-animation.json": MAX_SCENE_ANIMATION_BYTES,
        "class-package.json": MAX_CLASS_PACKAGE_BYTES * 2,
        "input-bindings.json": MAX_INPUT_BINDINGS_BYTES,
    }
    budget = budgets.get(relative)
    if relative.startswith("animation_objects/"):
        budget = MAX_SOURCE_BYTES
    if budget is None:
        return target.read_bytes()
    label = {
        "map-pack.json": "map pack",
        "runtime-config.json": "runtime config",
        "scene-animation.json": "scene animation",
        "class-package.json": "class package",
        "input-bindings.json": "input bindings",
    }.get(relative, "animation source")
    if (output / relative).is_symlink():
        raise ValueError(f"invalid run {label} evidence")
    with target.open("rb") as source:
        contents = source.read(budget + 1)
    if len(contents) > budget:
        raise ValueError(f"run {label} evidence exceeds byte budget")
    return contents


def _load_run_animation(
    output: Path, summary: dict[str, Any], files: dict[str, Any]
) -> SceneAnimation | None:
    paths = {"scene-animation.json"} | {
        path for path in files if path.startswith("animation_objects/")
    }
    if "animation" not in summary:
        if any(path in files for path in paths) or any(
            (output / path).exists() or (output / path).is_symlink()
            for path in ("scene-animation.json", "animation_objects")
        ):
            raise ValueError("run has unmanifested animation evidence")
        return None
    if "scene-animation.json" not in files:
        raise ValueError("run omits animation scene sidecar")
    animation = load_scene_animation(output)
    if canonical(animation.manifest()) != canonical(summary["animation"]):
        raise ValueError("run animation metadata disagrees with summary")
    expected = {"scene-animation.json"} | {s.source_file for s in animation.pack.sources}
    if paths != expected:
        raise ValueError("run animation source evidence differs from declared pack")
    return animation


def _load_run_class_package(
    output: Path,
    summary: dict[str, Any],
    files: dict[str, Any],
    simulator: Simulator,
) -> ClassPackage | None:
    package = simulator.class_package
    path = output / "class-package.json"
    if package is None:
        if (
            "class_package" in summary
            or "class-package.json" in files
            or path.exists()
            or path.is_symlink()
        ):
            raise ValueError("run contains unmanifested class package evidence")
        return None
    if "class-package.json" not in files:
        raise ValueError("run omits class package evidence")
    sidecar = ClassPackage.from_manifest(
        strict_json(_read_run_evidence(output, "class-package.json").decode("utf-8"))
    )
    declared = ClassPackage.from_manifest(summary.get("class_package"))
    if sidecar != package or declared != package:
        raise ValueError("run class package disagrees with embedded simulator rules")
    return package


def _load_run_input_profile(
    output: Path, summary: dict[str, Any], files: dict[str, Any]
) -> InputBindingProfile | None:
    path = output / "input-bindings.json"
    declared = summary.get("input_profile")
    if declared is None:
        if "input-bindings.json" in files or path.exists() or path.is_symlink():
            raise ValueError("run contains unmanifested input binding evidence")
        return None
    if "input-bindings.json" not in files:
        raise ValueError("run omits input binding evidence")
    sidecar = InputBindingProfile.from_manifest(
        strict_json(_read_run_evidence(output, "input-bindings.json").decode("utf-8"))
    )
    summary_profile = InputBindingProfile.from_manifest(declared)
    if sidecar != summary_profile:
        raise ValueError("run input binding sidecar disagrees with summary")
    return sidecar


def verify_run(output: Path) -> dict[str, Any]:
    """Check evidence integrity then replay every applied action and resulting state."""
    manifest = strict_json((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") not in {1, 2}
        or manifest.get("kind") != "offline-run-bundle"
    ):
        raise ValueError("unsupported run manifest")
    files = manifest.get("files")
    required = {
        "initial-state.json",
        "final-state.json",
        "summary.json",
        "trace.jsonl",
        "final.png",
    }
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("run manifest omits required evidence")
    for relative, expected in files.items():
        contents = _read_run_evidence(output, relative)
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"run evidence checksum mismatch: {relative}")
    initial = strict_json((output / "initial-state.json").read_text(encoding="utf-8"))
    summary = strict_json((output / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(initial, dict) or not isinstance(summary, dict):
        raise ValueError("run state and summary must be objects")
    runtime = load_run_runtime(output, manifest, summary)
    config = _recorded_config(summary)
    simulator = Simulator.restore(initial)
    class_package = _load_run_class_package(output, summary, files, simulator)
    input_profile = _load_run_input_profile(output, summary, files)
    if class_package is not None and runtime is None:
        raise ValueError("external class evidence requires structured runtime")
    if runtime is not None:
        validate_runtime_capabilities(summary["runtime_evidence"]["terminal"], class_package)
    profile_metadata = summary.get("class_profile")
    if profile_metadata is not None:
        profile = parse_loaded_class_profile(profile_metadata)
        if "class_profile" not in initial or profile.class_spec != simulator.class_spec:
            raise ValueError("run profile provenance disagrees with embedded simulator rules")
    elif "class_profile" in initial:
        raise ValueError("run omits provenance for its injected class profile")
    map_pack = simulator.map_pack
    mission_maps, expected_drops = _mission_requirements(map_pack)
    mission_oracle = None
    if runtime is not None and runtime.mission is not None:
        runtime.mission.bind_map_pack(map_pack)
        mission_oracle = MissionOracle(runtime.mission, map_pack)
    map_manifest_path = output / "map-pack.json"
    if map_pack is not None:
        if "map-pack.json" not in files:
            raise ValueError("run manifest omits embedded map pack evidence")
        if (
            map_manifest_path.is_symlink()
            or map_manifest_path.stat().st_size > _MAX_MAP_EVIDENCE_BYTES
        ):
            raise ValueError("invalid run map pack manifest")
        embedded = MapPack.from_manifest(strict_json(map_manifest_path.read_text(encoding="utf-8")))
        summary_pack = MapPack.from_manifest(summary.get("map_pack"))
        if embedded != map_pack or summary_pack != map_pack:
            raise ValueError("run map pack metadata disagrees with embedded simulator content")
    elif (
        summary.get("map_pack") is not None
        or "map-pack.json" in files
        or map_manifest_path.exists()
    ):
        raise ValueError("run contains unmanifested map pack evidence")
    if (
        simulator.seed != config.seed
        or simulator.job != config.job
        or simulator.snapshot().sequence != 0
        or simulator.snapshot().map_id != mission_maps[0]
        or simulator.step_ms != 50
    ):
        raise ValueError("run config disagrees with initial simulator state")
    if map_pack is not None or runtime is not None:
        legacy_spec = simulator.class_spec if "class_profile" in initial else None
        if legacy_spec is not None and not isinstance(legacy_spec, ClassSpec):
            raise ValueError("legacy profile is not a fixture ClassSpec")
        expected_initial = Simulator(
            config.seed,
            config.job,
            map_id="meadow" if map_pack is None else map_pack.entry_map,
            class_spec=legacy_spec,
            map_pack=map_pack,
            class_package=class_package,
        ).save()
        if canonical(simulator.save()) != canonical(expected_initial):
            raise ValueError("run initial state is not the declared fresh seeded world")
    sprite_pack = None
    sprite_manifest_path = output / "sprite-pack.json"
    if "sprite-pack.json" in files:
        if sprite_manifest_path.is_symlink() or sprite_manifest_path.stat().st_size > 512 * 1024:
            raise ValueError("invalid run sprite manifest")
        sprite_metadata = strict_json(sprite_manifest_path.read_text(encoding="utf-8"))
        sprite_pack = verify_dataset_sprite_pack(output, sprite_metadata)
        if summary.get("sprite_pack") != sprite_pack.manifest():
            raise ValueError("run sprite metadata disagrees with summary")
        if any(entry.source_file not in files for entry in sprite_pack.entries):
            raise ValueError("run manifest omits sprite source evidence")
    elif (
        summary.get("sprite_pack") is not None
        or sprite_manifest_path.exists()
        or (output / "sprite_objects").exists()
    ):
        raise ValueError("run contains unmanifested sprite evidence")
    animation = _load_run_animation(output, summary, files)
    renderer = SceneRenderer(sprite_pack=sprite_pack, map_pack=map_pack, animation=animation)
    fault_frames = _Frames(config.fault)
    count = 0
    actions: Counter[str] = Counter()
    events: Counter[str] = Counter()
    visited: set[str] = set()
    dialogue_maps: set[str] = set()
    collected_drops: set[tuple[str, str]] = set()
    cleared: dict[str, bool] = {}
    seen_intents: set[str] = set()
    input_dispatch_reasons: Counter[str] = Counter()
    input_dispatch_accepted = 0

    def check_image(path: Path, expected: ImageFrame) -> None:
        actual = ImageFrame.from_file(path)
        if (actual.width, actual.height, actual.pixels) != (
            expected.width,
            expected.height,
            expected.pixels,
        ):
            raise ValueError("recorded image disagrees with replay pixels")

    with (output / "trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if count >= config.max_steps:
                raise ValueError("replay trace exceeds the configured step budget")
            row = strict_json(line)
            if (
                not isinstance(row, dict)
                or type(row.get("schema_version")) is not int
                or row.get("schema_version") != manifest["schema_version"]
                or type(row.get("sequence")) is not int
                or row.get("sequence") != simulator.snapshot().sequence
            ):
                raise ValueError("replay sequence mismatch")
            snapshot = simulator.snapshot()
            visited.add(snapshot.map_id)
            cleared[snapshot.map_id] = not snapshot.monsters and not snapshot.drops
            frame = fault_frames.apply(renderer.render(snapshot))
            if hashlib.sha256(frame.pixels).hexdigest() != row.get("frame_sha256"):
                raise ValueError("replay frame mismatch")
            if snapshot.sequence % config.record_stride == 0:
                relative = f"frames/{snapshot.sequence:06d}.png"
                if relative not in files:
                    raise ValueError("missing sampled frame evidence")
                check_image(output / relative, frame)
            execution = row["execution"]
            requested = SimAction(**row["decision"]["action"])
            authorized = validate_runtime_step(
                row, runtime, now_ms=snapshot.timestamp_ms, seen_intents=seen_intents
            )
            if runtime is not None:
                validate_runtime_capabilities(row["runtime"], class_package)
            applied = SimAction(**execution["applied_action"])
            input_action = authorized
            expected_input_dispatch = None
            if input_profile is not None:
                input_action, expected_input_dispatch = input_profile.dispatch(
                    authorized, config.job, class_package
                )
                if canonical(execution.get("input_dispatch")) != canonical(expected_input_dispatch):
                    raise ValueError("recorded input dispatch disagrees with bound profile")
                if expected_input_dispatch is not None:
                    input_dispatch_reasons[expected_input_dispatch["reason"]] += 1
                    input_dispatch_accepted += int(expected_input_dispatch["accepted"])
            elif "input_dispatch" in execution:
                raise ValueError("trace has input dispatch without a bound profile")
            expected_action = (
                SimAction()
                if config.fault == "freeze_input" and snapshot.sequence >= 20
                else input_action
            )
            if applied != expected_action:
                raise ValueError("applied action disagrees with recorded fault configuration")
            actions[requested.kind] += 1
            outcome = simulator.step(applied)
            events.update(outcome.events)
            if mission_oracle is not None:
                mission_oracle.observe(snapshot, outcome.snapshot, outcome.events)
            if "loot" in outcome.events and snapshot.map_id == outcome.snapshot.map_id:
                remaining = {drop.id for drop in outcome.snapshot.drops}
                collected_drops.update(
                    (snapshot.map_id, drop.id)
                    for drop in snapshot.drops
                    if drop.id not in remaining
                )
            if "dialogue" in outcome.events:
                dialogue_maps.add(snapshot.map_id)
            if (outcome.accepted, outcome.reason, list(outcome.events)) != (
                execution["accepted"],
                execution["reason"],
                execution["events"],
            ):
                raise ValueError("replay execution mismatch")
            if digest(simulator.save()) != row["after_state_sha256"]:
                raise ValueError("replay state mismatch")
            count += 1
    final = strict_json((output / "final-state.json").read_text(encoding="utf-8"))
    snapshot = simulator.snapshot()
    visited.add(snapshot.map_id)
    cleared[snapshot.map_id] = not snapshot.monsters and not snapshot.drops
    check_image(output / "final.png", fault_frames.apply(renderer.render(snapshot)))
    if (
        count != summary["steps"]
        or simulator.save() != final
        or digest(final) != summary["final_state_sha256"]
    ):
        raise ValueError("replay final state mismatch")
    oracle_pass = (
        snapshot.player.alive
        and set(cleared) == set(mission_maps)
        and all(cleared.values())
        and dialogue_maps == set(mission_maps)
        and events["monster_defeated"] == expected_drops
        and len(collected_drops) == expected_drops
    )
    if mission_oracle is not None:
        mission_result = mission_oracle.result(snapshot)
        if canonical(summary.get("mission_evidence")) != canonical(mission_result):
            raise ValueError("mission objective witnesses disagree with action replay")
        oracle_pass = mission_result["completed"]
        if mission_oracle.program.conditional:
            terminal_state = summary["runtime_evidence"]["terminal"]["state"]
            branching = terminal_state.get("mission_branching")
            if not isinstance(branching, dict):
                raise ValueError("conditional mission omits runtime branch evidence")
            branch_check = compare_branch_decisions(
                branching.get("decisions"), mission_result["branches"]
            )
            if canonical(summary.get("conditional_branch_check")) != canonical(branch_check):
                raise ValueError("conditional branch check disagrees with replay evidence")
            oracle_pass = oracle_pass and branch_check["matched"]
        elif "conditional_branch_check" in summary:
            raise ValueError("branch check requires a conditional mission")
    elif "mission_evidence" in summary:
        raise ValueError("mission evidence requires an embedded mission program")
    elif "conditional_branch_check" in summary:
        raise ValueError("branch check requires an embedded conditional mission")

    if summary.get("status") not in {"completed", "failed", "stopped"}:
        raise ValueError("invalid run status")
    if summary["status"] == "completed" and not oracle_pass:
        raise ValueError("run falsely reports successful mission")
    derived = {
        "independent_goal_check": oracle_pass,
        "actions": dict(sorted(actions.items())),
        "events": dict(sorted(events.items())),
        "maps_visited": sorted(visited),
        "dialogue_maps": sorted(dialogue_maps),
        "final_coins": snapshot.player.coins,
        "drops_collected": len(collected_drops),
        "alive": snapshot.player.alive,
        "simulated_seconds": snapshot.timestamp_ms / 1000,
    }
    if input_profile is not None:
        attempts = sum(input_dispatch_reasons.values())
        derived["input_dispatch_summary"] = {
            "scope": "offline_simulated_skill_dispatch",
            "attempts": attempts,
            "accepted": input_dispatch_accepted,
            "denied": attempts - input_dispatch_accepted,
            "reasons": dict(sorted(input_dispatch_reasons.items())),
        }
    if count > config.max_steps or any(summary.get(key) != value for key, value in derived.items()):
        raise ValueError("run summary disagrees with replay evidence")
    return {
        "verified": True,
        "steps_replayed": count,
        "scope": "deterministic_simulator",
        "files_checked": len(files),
        "independent_goal_check": oracle_pass,
        "perception_reproduced": False,
        "runtime_identity_matches": summary.get("runtime_identity") == runtime_identity(),
    }


def _compare_observation(actual: object, expected: object, location: str) -> None:
    """Tolerate only tiny numeric measurement drift; identities/order stay exact."""
    if type(actual) is float:
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise ValueError(f"perception observation mismatch at {location}")
        if not math.isfinite(expected) or not math.isclose(
            actual, expected, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"perception observation mismatch at {location}")
    elif isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError(f"perception observation fields mismatch at {location}")
        for key, value in actual.items():
            _compare_observation(value, expected[key], f"{location}.{key}")
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"perception detection count mismatch at {location}")
        for index, (value, recorded) in enumerate(zip(actual, expected, strict=True)):
            _compare_observation(value, recorded, f"{location}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"perception observation mismatch at {location}")


def verify_perception(
    output: Path,
    detector: Detector | None = None,
    *,
    terrain_source: TerrainSource = "fixture",
) -> dict[str, Any]:
    """Actually re-infer every frame and re-run supervisor/policy, including termination.

    This reads the bundle without modifying it. Versioned terminal evidence and
    model options are required; old bundles remain supported by ``verify_run``.
    Only the built-in pixel pipeline or the hash-bound OnnxDetector is eligible for
    a model reproduction claim. Arbitrary custom detectors have no proven identity.
    """
    validate_terrain_source(terrain_source, detector)
    manifest_bytes = _run_evidence_path(output, "manifest.json").read_bytes()
    replay = verify_run(output)
    if _run_evidence_path(output, "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("run manifest changed during action replay preflight")
    manifest = strict_json(manifest_bytes.decode("utf-8"))
    summary = strict_json(_run_evidence_path(output, "summary.json").read_text(encoding="utf-8"))
    runtime = load_run_runtime(output, manifest, summary)
    evidence = summary.get("perception_evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema_version", "settings", "terminal"}
        or type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
    ):
        raise ValueError(
            "run lacks supported terminal perception evidence; regenerate it with the current "
            "run_episode, or use action-only verify_run for this legacy bundle"
        )
    if detector is not None and type(detector) is not OnnxDetector:
        raise ValueError("custom detector identity cannot be verified; supply OnnxDetector")
    map_pack = (
        None if summary.get("map_pack") is None else MapPack.from_manifest(summary["map_pack"])
    )
    settings = _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    if detector is not None:
        assert isinstance(detector, OnnxDetector)
        if detector.loaded_model_sha256 != detector.manifest.model_sha256:
            raise ValueError("supplied model manifest is not bound to the loaded model hash")
    if canonical(settings) != canonical(evidence["settings"]):
        raise ValueError(
            "supplied terrain mode, model/hash, full manifest or inference options do not match run"
        )
    if canonical(summary.get("model_manifest")) != canonical(settings["model_manifest"]):
        raise ValueError("run model manifest disagrees with perception evidence")
    if summary.get("detector") != ("pixel-fixture" if detector is None else "onnx"):
        raise ValueError("run detector does not match perception evidence")
    terminal = evidence["terminal"]
    if not isinstance(terminal, dict) or set(terminal) != {
        "cause",
        "guard",
        "decision",
        "observation",
    }:
        raise ValueError("run terminal perception evidence is incomplete")
    config = _recorded_config(summary)
    initial = strict_json(
        _run_evidence_path(output, "initial-state.json").read_text(encoding="utf-8")
    )
    profile = (
        None
        if summary.get("class_profile") is None
        else parse_loaded_class_profile(summary["class_profile"])
    )
    sprite_pack = (
        verify_dataset_sprite_pack(output, summary["sprite_pack"])
        if summary.get("sprite_pack") is not None
        else None
    )
    animation = _load_run_animation(output, summary, manifest["files"])
    class_package = _load_run_class_package(
        output, summary, manifest["files"], Simulator.restore(initial)
    )
    input_profile = _load_run_input_profile(output, summary, manifest["files"])
    count = 0
    with _run_evidence_path(output, "trace.jsonl").open(encoding="utf-8") as stream:

        def compare_trace(actual: dict[str, Any]) -> None:
            nonlocal count
            line = stream.readline()
            if not line:
                raise ValueError("perception replay produced an unrecorded/skipped trace step")
            expected = strict_json(line)
            normalized = json.loads(canonical(actual))
            if not isinstance(expected, dict) or set(expected) != set(normalized):
                raise ValueError("perception trace fields mismatch")
            _compare_observation(
                normalized["observation"], expected["observation"], f"step[{count}]"
            )
            for key in normalized.keys() - {"observation"}:
                if canonical(normalized[key]) != canonical(expected[key]):
                    raise ValueError(f"perception/policy trace mismatch at step {count}: {key}")
            count += 1

        reproduced = _execute_episode(
            config=config,
            detector=detector,
            class_profile=profile,
            sprite_pack=sprite_pack,
            map_pack=map_pack,
            initial_state=initial,
            trace_consumer=compare_trace,
            terrain_source=terrain_source,
            runtime=runtime,
            animation=animation,
            class_package=class_package,
            input_profile=input_profile,
        )
        if stream.readline():
            raise ValueError("perception replay terminated before all recorded steps")
    actual_terminal = json.loads(canonical(reproduced["perception_evidence"]["terminal"]))
    _compare_observation(
        actual_terminal["observation"], terminal["observation"], "terminal.observation"
    )
    for key in ("cause", "guard", "decision"):
        if canonical(actual_terminal[key]) != canonical(terminal[key]):
            raise ValueError(f"terminal perception/policy mismatch: {key}")
    # Wall time and implementation identity can differ while outputs reproduce.
    # All semantic summary fields, including fault recovery and stopping, must match.
    for key in (
        "schema_version",
        "scope",
        "ruleset",
        "config",
        "detector",
        "status",
        "reason",
        "steps",
        "simulated_seconds",
        "paused_steps",
        "recoveries",
        "actions",
        "events",
        "visual_effects",
        "independent_goal_check",
        "maps_visited",
        "dialogue_maps",
        "final_coins",
        "drops_collected",
        "alive",
        "final_state_sha256",
        "model_manifest",
        "class_profile",
        "map_pack",
        "runtime_evidence",
        "animation",
        "class_package",
        "input_profile",
        "input_dispatch_summary",
        "mission_evidence",
        "conditional_branch_check",
    ):
        if canonical(reproduced.get(key)) != canonical(summary.get(key)):
            raise ValueError(f"perception replay summary mismatch: {key}")
    if _run_evidence_path(output, "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("run manifest changed during perception verification")
    for relative, expected_hash in manifest["files"].items():
        if hashlib.sha256(_read_run_evidence(output, relative)).hexdigest() != expected_hash:
            raise ValueError(f"run evidence changed during perception verification: {relative}")
    identity = runtime_identity()
    result = {
        "schema_version": 1,
        "verified": True,
        "scope": "synthetic_fixture_perception_and_policy_reexecution",
        "action_replay_verified": replay["verified"],
        "perception_reproduced": True,
        "policy_reproduced": True,
        "trace_steps_compared": count,
        "frames_reinferred": count + 1,
        "terminal_observation_compared": True,
        "terminal_guard_compared": terminal["guard"] is not None,
        "terminal_decision_compared": terminal["decision"] is not None,
        "model_hash_verified": detector is not None,
        "perception_settings": settings,
        "observation_tolerances": {"absolute": 1e-6, "relative": 1e-6},
        "decision_comparison": "exact; includes guard, action, reason, task and terminal state",
        "recorded_status": summary["status"],
        "independent_goal_check": replay["independent_goal_check"],
        "files_unchanged": True,
        "runtime_identity_matches": summary.get("runtime_identity") == identity,
        "recorded_runtime_identity": summary.get("runtime_identity"),
        "verification_runtime_identity": identity,
    }
    if runtime is not None:
        result["structured_runtime_reproduced"] = True
        result["runtime_settings"] = runtime.manifest()
        result["belief_frames_recomputed"] = count + 1
        result["terminal_runtime_compared"] = True
    return result


def _record_episode(
    output: Path,
    config: RunConfig,
    detector: Detector | None,
    *,
    failure_name: str,
    terrain_source: TerrainSource = "fixture",
    map_pack: MapPack | None = None,
    runtime: RuntimeConfig | None = None,
    animation: SceneAnimation | None = None,
    class_package: ClassPackage | None = None,
) -> dict[str, Any]:
    """Capture exactly once; retain the actual failure rather than executing a rerun."""
    staging = output / "active-episode"
    # run_episode refuses an existing path, so cleanup only targets the directory
    # created by this particular call in this already-new validation output.
    result = run_episode(
        staging,
        config,
        detector,
        terrain_source=terrain_source,
        map_pack=map_pack,
        runtime=runtime,
        animation=animation,
        class_package=class_package,
    )
    if result["status"] == "completed":
        shutil.rmtree(staging)
        result.pop("output", None)
        retained: str | None = None
    else:
        destination = output / failure_name
        if destination.exists():
            raise FileExistsError(f"failure evidence already exists: {destination}")
        staging.rename(destination)
        result["output"] = str(destination)
        retained = failure_name
    result["episode_capture"] = {
        "mode": "original_single_execution",
        "record_stride": config.record_stride,
        "full_action_observation_terminal_trace": True,
        "retained_bundle": retained,
    }
    return result


def _capture_contract() -> dict[str, Any]:
    return {
        "mode": "single_execution_staging",
        "record_stride": 16,
        "full_action_observation_terminal_trace": True,
        "successful_normal_bundles": "discarded_after_summary",
        "failed_bundles": "retained_original_execution_not_rerun",
        "wall_and_resources_include_capture_io": True,
        "bounded_storage": "one_active_episode_plus_retained_failures_and_requested_fault_bundles",
    }


def run_suite(
    output: Path,
    seeds: int = 3,
    detector: Detector | None = None,
    *,
    terrain_source: TerrainSource = "fixture",
    map_pack: MapPack | None = None,
    runtime: RuntimeConfig | None = None,
    animation: SceneAnimation | None = None,
    class_packages: tuple[ClassPackage, ...] | None = None,
) -> dict[str, Any]:
    if type(seeds) is not int or not 1 <= seeds <= 1000:
        raise ValueError("seeds must be in [1, 1000]")
    if runtime is not None and type(runtime) is not RuntimeConfig:
        raise ValueError("runtime must be a structured RuntimeConfig or None")
    jobs = _class_jobs(class_packages)
    if class_packages is not None and runtime is None:
        raise ValueError("external class packages require structured runtime")
    settings = _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    if output.exists():
        raise FileExistsError(f"suite output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    source_options: dict[str, Any] = (
        {} if terrain_source == "fixture" else {"terrain_source": "model"}
    )
    if map_pack is not None:
        source_options["map_pack"] = map_pack
    if runtime is not None:
        source_options["runtime"] = runtime
    if animation is not None:
        source_options["animation"] = animation
    for job, class_package in jobs:
        config_type = RunConfig if class_package is None else CapabilityRunConfig
        episode_options: dict[str, Any] = dict(source_options)
        if class_package is not None:
            episode_options["class_package"] = class_package
        for seed in range(seeds):
            config = config_type(seed, job, max_steps=_episode_budget(map_pack), record_stride=16)
            result = _record_episode(
                output,
                config,
                detector,
                failure_name=f"failure-{job}-{seed}",
                **episode_options,
            )
            rows.append(result)
    fault_job, fault_class_package = jobs[0]
    fault_config_type = RunConfig if fault_class_package is None else CapabilityRunConfig
    for fault in FAULTS[1:]:
        result = run_episode(
            output / fault,
            fault_config_type(
                job=fault_job,
                fault=fault,
                max_steps=_episode_budget(map_pack, fault=True),
                record_stride=16,
            ),
            detector,
            terrain_source=terrain_source,
            map_pack=map_pack,
            runtime=runtime,
            animation=animation,
            class_package=fault_class_package,
        )
        result["episode_capture"] = {
            "mode": "original_single_execution",
            "record_stride": 16,
            "full_action_observation_terminal_trace": True,
            "retained_bundle": fault,
        }
        result["replay"] = verify_run(output / fault)
        rows.append(result)
    expected = {
        "none": "completed",
        "brief_dropout": "completed",
        "duplicate_frames": "completed",
        "persistent_dropout": "stopped",
        "freeze_input": "failed",
        "loot_false_negative": "failed",
    }
    passed = all(row["status"] == expected[row["config"]["fault"]] for row in rows)
    if canonical(settings) != canonical(
        _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    ):
        raise ValueError("detector settings changed during the suite")
    report = {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "passed": passed,
        "episodes": len(rows),
        "normal_episodes": seeds * len(jobs),
        "actual_wall_seconds": round(time.monotonic() - started, 6),
        "episode_wall_seconds": round(sum(row["wall_seconds"] for row in rows), 6),
        "simulated_seconds": sum(row["simulated_seconds"] for row in rows),
        "real_recording_acceptance": "not_run",
        "24h_wall_soak": "not_run",
        "perception_settings": settings,
        "model_manifest": settings["model_manifest"],
        "episode_capture": _capture_contract(),
        "results": rows,
    }
    if map_pack is not None:
        report["map_pack"] = map_pack.manifest()
    report.update(runtime_summary_fields(runtime))
    if animation is not None:
        report["animation"] = animation.manifest()
    if class_packages is not None:
        report["class_packages"] = [package.manifest() for package in class_packages]
    _write_json(output / "suite.json", report)
    return report


def run_soak(
    output: Path,
    seconds: int = 60,
    detector: Detector | None = None,
    *,
    resource_interval_seconds: float = 1.0,
    max_resource_samples: int = 1024,
    terrain_source: TerrainSource = "fixture",
    map_pack: MapPack | None = None,
    runtime: RuntimeConfig | None = None,
    animation: SceneAnimation | None = None,
    class_packages: tuple[ClassPackage, ...] | None = None,
) -> dict[str, Any]:
    """Wall-clock soak, checkpointed after each mission (up to one episode overrun).

    Simulated time is separately reported. An accelerated hour is never counted as
    an hour of process uptime. Each episode is captured once; the original failing
    bundle is retained, while only this run's successful staging bundle is deleted.
    """
    if type(seconds) is not int or not 1 <= seconds <= 7 * 24 * 3600:
        raise ValueError("soak seconds must be in [1, 604800]")
    if runtime is not None and type(runtime) is not RuntimeConfig:
        raise ValueError("runtime must be a structured RuntimeConfig or None")
    jobs = _class_jobs(class_packages)
    if class_packages is not None and runtime is None:
        raise ValueError("external class packages require structured runtime")
    resources = ProcessResourceMonitor(resource_interval_seconds, max_resource_samples)
    settings = _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
    if output.exists():
        raise FileExistsError(f"soak output already exists: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    completed = 0
    simulated = 0.0
    passed = True
    report: dict[str, Any] = {}
    run_error: str | None = None
    source_options: dict[str, Any] = (
        {} if terrain_source == "fixture" else {"terrain_source": "model"}
    )
    if map_pack is not None:
        source_options["map_pack"] = map_pack
    if runtime is not None:
        source_options["runtime"] = runtime
    if animation is not None:
        source_options["animation"] = animation
    resources.start()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            while time.monotonic() - start < seconds:
                job, class_package = jobs[completed % len(jobs)]
                config_type = RunConfig if class_package is None else CapabilityRunConfig
                config = config_type(
                    seed=completed, job=job, max_steps=_episode_budget(map_pack), record_stride=16
                )
                episode_options: dict[str, Any] = dict(source_options)
                if class_package is not None:
                    episode_options["class_package"] = class_package
                result = _record_episode(
                    output,
                    config,
                    detector,
                    failure_name=f"failure-{completed + 1}",
                    **episode_options,
                )
                if canonical(settings) != canonical(
                    _perception_settings(detector, terrain_source=terrain_source, map_pack=map_pack)
                ):
                    raise ValueError("detector settings changed during the soak")
                stream.write(canonical(result).decode() + "\n")
                stream.flush()
                completed += 1
                simulated += result["simulated_seconds"]
                if result["status"] != "completed":
                    passed = False
                    break
                _write_json(
                    output / "checkpoint.json",
                    {
                        "completed_episodes": completed,
                        "actual_wall_seconds": time.monotonic() - start,
                        "requested_wall_seconds": seconds,
                        "simulated_seconds": simulated,
                        "perception_settings": settings,
                        "episode_capture": _capture_contract(),
                        "resources": resources.report(),
                        **runtime_summary_fields(runtime),
                        **({"animation": animation.manifest()} if animation is not None else {}),
                        **(
                            {"class_packages": [package.manifest() for package in class_packages]}
                            if class_packages is not None
                            else {}
                        ),
                    },
                )
    except BaseException as exc:
        passed = False
        run_error = type(exc).__name__
        raise
    finally:
        resource_report = resources.close()
        actual_wall = time.monotonic() - start
        completed_request = passed and actual_wall >= seconds and completed > 0
        report = {
            "schema_version": 1,
            "scope": "synthetic_fixture",
            "episodes": completed,
            "requested_wall_seconds": seconds,
            "actual_wall_seconds": round(actual_wall, 6),
            "simulated_seconds": simulated,
            "passed": completed_request,
            "24h_wall_soak": "passed"
            if completed_request and seconds >= 86400 and actual_wall >= 86400
            else "not_completed",
            "perception_settings": settings,
            "model_manifest": settings["model_manifest"],
            "resources": resource_report,
            "run_error": run_error,
            "episode_capture": _capture_contract(),
            "partial_episode_bundle": "active-episode"
            if (output / "active-episode").exists()
            else None,
        }
        if map_pack is not None:
            report["map_pack"] = map_pack.manifest()
        report.update(runtime_summary_fields(runtime))
        if animation is not None:
            report["animation"] = animation.manifest()
        if class_packages is not None:
            report["class_packages"] = [package.manifest() for package in class_packages]
        _write_json(output / "soak.json", report)
    return report


def chat_draft(incoming: str, reply: str) -> dict[str, object]:
    """Human-reviewable local text, with no network or message-sending backend."""
    if (
        not isinstance(incoming, str)
        or not isinstance(reply, str)
        or not incoming.strip()
        or not reply.strip()
    ):
        raise ValueError("incoming and reply must be nonempty text")
    if max(len(incoming), len(reply)) > 2000:
        raise ValueError("chat draft text exceeds 2000 characters")
    return {
        "schema_version": 1,
        "incoming": incoming,
        "draft": reply,
        "requires_human_confirmation": True,
        "sent": False,
    }
