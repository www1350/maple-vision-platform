"""Offline closed-loop validation, fault injection and deterministic failure replay."""

import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .agent import AgentDecision, HybridPerception, Supervisor, VisualAgent, verify_visual_effect
from .catalog import strict_json
from .profiles import LoadedClassProfile, parse_loaded_class_profile
from .scene import ImageFrame, SceneRenderer
from .sim_types import CLASS_SPECS, RULESET, SimAction
from .simulation import Simulator
from .sprite_pack import SpritePack, verify_dataset_sprite_pack
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
    config = config or RunConfig()
    spec = None if class_profile is None else class_profile.class_spec
    simulator = Simulator(config.seed, config.job, class_spec=spec)
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
    for _ in range(config.max_steps):
        before = simulator.snapshot()
        visited.add(before.map_id)
        cleared[before.map_id] = not before.monsters and not before.drops
        guard = supervisor.inspect(observation, before.timestamp_ms)
        if guard.state == "stopped":
            status, reason = "stopped", guard.reason
            break
        if guard.state == "paused":
            paused_steps += 1
            decision = AgentDecision(SimAction(), policy.phase, "paused", guard.reason)
        else:
            decision = policy.decide(observation)
        if decision.state in {"failed", "completed"}:
            status, reason = decision.state, decision.reason
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
        "model_manifest": asdict(detector.manifest)
        if detector is not None and hasattr(detector, "manifest")
        else None,
        "class_profile": None if class_profile is None else asdict(class_profile),
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
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("invalid run path")
        target = (output / relative).resolve()
        if (
            not target.is_relative_to(output.resolve())
            or ".." in Path(relative).parts
            or Path(relative).is_absolute()
        ):
            raise ValueError("run evidence path escapes directory")
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


def run_suite(output: Path, seeds: int = 3) -> dict[str, Any]:
    if type(seeds) is not int or not 1 <= seeds <= 1000:
        raise ValueError("seeds must be in [1, 1000]")
    if output.exists():
        raise FileExistsError(f"suite output already exists: {output}")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for job in CLASS_SPECS:
        for seed in range(seeds):
            config = RunConfig(seed, job)
            result = run_episode(config=config)
            rows.append(result)
            if result["status"] != "completed":
                run_episode(output / f"failure-{job}-{seed}", config)
    for fault in FAULTS[1:]:
        result = run_episode(output / fault, RunConfig(fault=fault, max_steps=500))
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
    report = {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "passed": passed,
        "episodes": len(rows),
        "normal_episodes": seeds * len(CLASS_SPECS),
        "actual_wall_seconds": round(sum(row["wall_seconds"] for row in rows), 6),
        "simulated_seconds": sum(row["simulated_seconds"] for row in rows),
        "real_recording_acceptance": "not_run",
        "24h_wall_soak": "not_run",
        "results": rows,
    }
    _write_json(output / "suite.json", report)
    return report


def run_soak(output: Path, seconds: int = 60) -> dict[str, Any]:
    """Wall-clock soak, checkpointed after each mission (up to one episode overrun).

    Simulated time is separately reported. An accelerated hour is never counted as
    an hour of process uptime. A failure stops the soak and preserves a replay.
    """
    if type(seconds) is not int or not 1 <= seconds <= 7 * 24 * 3600:
        raise ValueError("soak seconds must be in [1, 604800]")
    if output.exists():
        raise FileExistsError(f"soak output already exists: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    completed = 0
    simulated = 0.0
    passed = True
    report: dict[str, Any] = {}
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            while time.monotonic() - start < seconds:
                job = tuple(CLASS_SPECS)[completed % len(CLASS_SPECS)]
                config = RunConfig(seed=completed, job=job)
                result = run_episode(config=config)
                stream.write(canonical(result).decode() + "\n")
                stream.flush()
                completed += 1
                simulated += result["simulated_seconds"]
                if result["status"] != "completed":
                    passed = False
                    run_episode(output / f"failure-{completed}", config)
                    break
                _write_json(
                    output / "checkpoint.json",
                    {
                        "completed_episodes": completed,
                        "actual_wall_seconds": time.monotonic() - start,
                        "requested_wall_seconds": seconds,
                        "simulated_seconds": simulated,
                    },
                )
    finally:
        actual_wall = time.monotonic() - start
        report = {
            "schema_version": 1,
            "scope": "synthetic_fixture",
            "episodes": completed,
            "requested_wall_seconds": seconds,
            "actual_wall_seconds": round(actual_wall, 6),
            "simulated_seconds": simulated,
            "passed": passed and actual_wall >= seconds and completed > 0,
            "24h_wall_soak": "passed" if passed and actual_wall >= 86400 else "not_completed",
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
