"""Offline closed-loop validation, fault injection and deterministic failure replay."""

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .agent import AgentDecision, HybridPerception, Supervisor, VisualAgent, verify_visual_effect
from .catalog import strict_json
from .onnx_detector import OnnxDetector
from .profiles import LoadedClassProfile, parse_loaded_class_profile
from .scene import ImageFrame, SceneRenderer
from .sim_types import CLASS_SPECS, RULESET, SimAction
from .simulation import Simulator
from .sprite_pack import SpritePack, verify_dataset_sprite_pack
from .validation_resources import ProcessResourceMonitor
from .vision import Detector, PixelPerception

FAULTS = ("none", "brief_dropout", "persistent_dropout", "duplicate_frames", "freeze_input")


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
        for name, value, low, high in (
            ("seed", self.seed, 0, 2**32 - 1),
            ("max_steps", self.max_steps, 1, 100_000),
            ("record_stride", self.record_stride, 1, 1000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.job not in CLASS_SPECS or self.fault not in FAULTS:
            raise ValueError("unknown job or fault")


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
        return frame


def run_episode(
    output: Path | None = None,
    config: RunConfig | None = None,
    detector: Detector | None = None,
    *,
    class_profile: LoadedClassProfile | None = None,
    sprite_pack: SpritePack | None = None,
) -> dict[str, Any]:
    return _execute_episode(
        output, config, detector, class_profile=class_profile, sprite_pack=sprite_pack
    )


def _perception_settings(detector: Detector | None) -> dict[str, Any]:
    if detector is None:
        return {
            "pipeline": "pixel-fixture",
            "implementation": "maple_vision.vision.PixelPerception",
            "model_manifest": None,
            "loaded_model_sha256": None,
            "inference_options": None,
            "verifiable_model_binding": True,
        }
    native = detector if type(detector) is OnnxDetector else None
    if native is not None and native.loaded_model_sha256 != native.manifest.model_sha256:
        raise ValueError("model manifest is not bound to the loaded model hash")
    return {
        "pipeline": "onnx-objects+fixture-terrain-hud-v1",
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


def _execute_episode(
    output: Path | None = None,
    config: RunConfig | None = None,
    detector: Detector | None = None,
    *,
    class_profile: LoadedClassProfile | None = None,
    sprite_pack: SpritePack | None = None,
    initial_state: dict[str, Any] | None = None,
    trace_consumer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config = config or RunConfig()
    spec = None if class_profile is None else class_profile.class_spec
    simulator = (
        Simulator(config.seed, config.job, class_spec=spec)
        if initial_state is None
        else Simulator.restore(initial_state)
    )
    settings = _perception_settings(detector)
    renderer, fault_frames = SceneRenderer(sprite_pack=sprite_pack), _Frames(config.fault)
    if output is not None:
        if output.exists():
            raise FileExistsError(f"run output already exists: {output}")
        output.mkdir(parents=True)
        (output / "frames").mkdir()
    perception = PixelPerception() if detector is None else HybridPerception(detector)
    policy, supervisor = VisualAgent(config.job, class_spec=spec), Supervisor()
    initial = simulator.save()
    if output:
        _write_json(output / "initial-state.json", initial)
        if sprite_pack is not None:
            sprite_pack.write_sources(output)
            _write_json(output / "sprite-pack.json", sprite_pack.manifest())
    trace: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    actions: Counter[str] = Counter()
    events: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    paused_steps = 0
    visited: set[str] = set()
    dialogue_maps: set[str] = set()
    collected_drops: set[tuple[str, str]] = set()
    cleared: dict[str, bool] = {}
    wall_start = time.perf_counter()
    frame = fault_frames.apply(renderer.render(simulator.snapshot()))
    observation = perception.observe(frame)
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
            status, reason = "stopped", guard.reason
            terminal = {"cause": "supervisor", "guard": asdict(guard), "decision": None}
            break
        if guard.state == "paused":
            paused_steps += 1
            decision = AgentDecision(SimAction(), policy.phase, "paused", guard.reason)
        else:
            decision = policy.decide(observation)
        if decision.state in {"failed", "completed"}:
            status, reason = decision.state, decision.reason
            terminal = {
                "cause": "policy",
                "guard": asdict(guard),
                "decision": asdict(decision),
            }
            break
        action = decision.action
        applied = (
            SimAction() if config.fault == "freeze_input" and before.sequence >= 20 else action
        )
        actions[action.kind] += 1
        if output and before.sequence % config.record_stride == 0:
            (output / "frames" / f"{before.sequence:06d}.png").write_bytes(frame.png_bytes())
            if len(previews) < 600:
                previews.append(frame.to_image())
        outcome = simulator.step(applied)
        events.update(outcome.events)
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
        row = {
            "schema_version": 1,
            "sequence": before.sequence,
            "frame_sha256": hashlib.sha256(frame.pixels).hexdigest(),
            "observation": asdict(observation),
            "guard": asdict(guard),
            "decision": asdict(decision),
            "execution": {
                "applied_action": asdict(applied),
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "events": list(outcome.events),
            },
            "visual_effect": effect,
            "after_state_sha256": digest(simulator.save()),
        }
        if output:
            trace.append(row)
        if trace_consumer is not None:
            trace_consumer(row)
        observation, frame = following, following_frame
    final = simulator.snapshot()
    visited.add(final.map_id)
    cleared[final.map_id] = not final.monsters and not final.drops
    oracle_pass = (
        final.player.alive
        and set(cleared) == {"meadow", "cavern"}
        and all(cleared.values())
        and dialogue_maps == {"meadow", "cavern"}
        and events["monster_defeated"] == 6
        and len(collected_drops) == 6
    )
    if status == "completed" and not oracle_pass:
        status, reason = "failed", "independent_goal_check_failed"
    elapsed = time.perf_counter() - wall_start
    if canonical(settings) != canonical(_perception_settings(detector)):
        raise ValueError("detector settings changed during the episode")
    summary: dict[str, Any] = {
        "schema_version": 1,
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
    if sprite_pack is not None:
        summary["sprite_pack"] = sprite_pack.manifest()
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
        _write_json(
            output / "manifest.json",
            {
                "schema_version": 1,
                "kind": "offline-run-bundle",
                "files": files,
                "recording": "sampled_png_and_preview_gif; full_action_trace",
                "replay_scope": (
                    "deterministic_simulator_action_replay; not_counterfactual_video_control"
                ),
            },
        )
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


def verify_run(output: Path) -> dict[str, Any]:
    """Check evidence integrity then replay every applied action and resulting state."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
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
        target = _run_evidence_path(output, relative)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"run evidence checksum mismatch: {relative}")
    initial = json.loads((output / "initial-state.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    config = RunConfig(**summary["config"])
    simulator = Simulator.restore(initial)
    profile_metadata = summary.get("class_profile")
    if profile_metadata is not None:
        profile = parse_loaded_class_profile(profile_metadata)
        if initial.get("schema_version") != 2 or profile.class_spec != simulator.class_spec:
            raise ValueError("run profile provenance disagrees with embedded simulator rules")
    elif initial.get("schema_version") == 2:
        raise ValueError("run omits provenance for its injected class profile")
    if (
        simulator.seed != config.seed
        or simulator.job != config.job
        or simulator.snapshot().sequence != 0
        or simulator.snapshot().map_id != "meadow"
        or simulator.step_ms != 50
    ):
        raise ValueError("run config disagrees with initial simulator state")
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
    renderer, fault_frames = SceneRenderer(sprite_pack=sprite_pack), _Frames(config.fault)
    count = 0
    actions: Counter[str] = Counter()
    events: Counter[str] = Counter()
    visited: set[str] = set()
    dialogue_maps: set[str] = set()
    collected_drops: set[tuple[str, str]] = set()
    cleared: dict[str, bool] = {}

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
            row = json.loads(line)
            if (
                row.get("schema_version") != 1
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
            applied = SimAction(**execution["applied_action"])
            expected_action = (
                SimAction()
                if config.fault == "freeze_input" and snapshot.sequence >= 20
                else requested
            )
            if applied != expected_action:
                raise ValueError("applied action disagrees with recorded fault configuration")
            actions[requested.kind] += 1
            outcome = simulator.step(applied)
            events.update(outcome.events)
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
    final = json.loads((output / "final-state.json").read_text(encoding="utf-8"))
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
        and set(cleared) == {"meadow", "cavern"}
        and all(cleared.values())
        and dialogue_maps == {"meadow", "cavern"}
        and events["monster_defeated"] == 6
        and len(collected_drops) == 6
    )
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


def verify_perception(output: Path, detector: Detector | None = None) -> dict[str, Any]:
    """Actually re-infer every frame and re-run supervisor/policy, including termination.

    This reads the bundle without modifying it. Versioned terminal evidence and
    model options are required; old bundles remain supported by ``verify_run``.
    Only the built-in pixel pipeline or the hash-bound OnnxDetector is eligible for
    a model reproduction claim. Arbitrary custom detectors have no proven identity.
    """
    manifest_bytes = _run_evidence_path(output, "manifest.json").read_bytes()
    replay = verify_run(output)
    if _run_evidence_path(output, "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("run manifest changed during action replay preflight")
    manifest = strict_json(manifest_bytes.decode("utf-8"))
    summary = strict_json(_run_evidence_path(output, "summary.json").read_text(encoding="utf-8"))
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
    settings = _perception_settings(detector)
    if detector is not None:
        assert isinstance(detector, OnnxDetector)
        if detector.loaded_model_sha256 != detector.manifest.model_sha256:
            raise ValueError("supplied model manifest is not bound to the loaded model hash")
    if canonical(settings) != canonical(evidence["settings"]):
        raise ValueError("supplied model/hash, full manifest or inference options do not match run")
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
    config = RunConfig(**summary["config"])
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
            initial_state=initial,
            trace_consumer=compare_trace,
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
    ):
        if canonical(reproduced.get(key)) != canonical(summary.get(key)):
            raise ValueError(f"perception replay summary mismatch: {key}")
    if _run_evidence_path(output, "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("run manifest changed during perception verification")
    for relative, expected_hash in manifest["files"].items():
        target = _run_evidence_path(output, relative)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"run evidence changed during perception verification: {relative}")
    identity = runtime_identity()
    return {
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


def _record_episode(
    output: Path,
    config: RunConfig,
    detector: Detector | None,
    *,
    failure_name: str,
) -> dict[str, Any]:
    """Capture exactly once; retain the actual failure rather than executing a rerun."""
    staging = output / "active-episode"
    # run_episode refuses an existing path, so cleanup only targets the directory
    # created by this particular call in this already-new validation output.
    result = run_episode(staging, config, detector)
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


def run_suite(output: Path, seeds: int = 3, detector: Detector | None = None) -> dict[str, Any]:
    if type(seeds) is not int or not 1 <= seeds <= 1000:
        raise ValueError("seeds must be in [1, 1000]")
    if output.exists():
        raise FileExistsError(f"suite output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    settings = _perception_settings(detector)
    rows: list[dict[str, Any]] = []
    for job in CLASS_SPECS:
        for seed in range(seeds):
            config = RunConfig(seed, job, record_stride=16)
            result = _record_episode(output, config, detector, failure_name=f"failure-{job}-{seed}")
            rows.append(result)
    for fault in FAULTS[1:]:
        result = run_episode(
            output / fault, RunConfig(fault=fault, max_steps=500, record_stride=16), detector
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
    }
    passed = all(row["status"] == expected[row["config"]["fault"]] for row in rows)
    if canonical(settings) != canonical(_perception_settings(detector)):
        raise ValueError("detector settings changed during the suite")
    report = {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "passed": passed,
        "episodes": len(rows),
        "normal_episodes": seeds * len(CLASS_SPECS),
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
    _write_json(output / "suite.json", report)
    return report


def run_soak(
    output: Path,
    seconds: int = 60,
    detector: Detector | None = None,
    *,
    resource_interval_seconds: float = 1.0,
    max_resource_samples: int = 1024,
) -> dict[str, Any]:
    """Wall-clock soak, checkpointed after each mission (up to one episode overrun).

    Simulated time is separately reported. An accelerated hour is never counted as
    an hour of process uptime. Each episode is captured once; the original failing
    bundle is retained, while only this run's successful staging bundle is deleted.
    """
    if type(seconds) is not int or not 1 <= seconds <= 7 * 24 * 3600:
        raise ValueError("soak seconds must be in [1, 604800]")
    resources = ProcessResourceMonitor(resource_interval_seconds, max_resource_samples)
    settings = _perception_settings(detector)
    if output.exists():
        raise FileExistsError(f"soak output already exists: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    completed = 0
    simulated = 0.0
    passed = True
    report: dict[str, Any] = {}
    run_error: str | None = None
    resources.start()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            while time.monotonic() - start < seconds:
                job = tuple(CLASS_SPECS)[completed % len(CLASS_SPECS)]
                config = RunConfig(seed=completed, job=job, record_stride=16)
                result = _record_episode(
                    output, config, detector, failure_name=f"failure-{completed + 1}"
                )
                if canonical(settings) != canonical(_perception_settings(detector)):
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
