"""Exercise an installed wheel in a clean, runtime-only environment outside the checkout.

Only this script's standard-library imports run in the build environment. All package
checks run as subprocesses in a separate virtual environment, and stdout is one JSON
report. It intentionally does not train a model; CI has a separate opt-in CPU smoke.
The runtime checks include local sync, lifecycle data, pixel re-inference, and
the packaged terrain configuration and model-only terrain route without native extras.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import venv
from pathlib import Path
from typing import Any


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_json(text: str) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"CLI emitted a non-finite JSON constant: {value}")

    value = json.loads(text, parse_constant=invalid_constant)
    require(isinstance(value, dict), "CLI output must be exactly one JSON object")
    return value


def smoke(wheel: Path, checkout: Path) -> dict[str, Any]:
    started = time.perf_counter()
    require(wheel.is_file() and wheel.suffix == ".whl", "wheel must be an existing .whl file")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="maple-wheel-") as temporary:
        scratch = Path(temporary).resolve()
        require(not scratch.is_relative_to(checkout), "smoke scratch must be outside checkout")
        environment = scratch / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        executable = binaries / ("maple-vision.exe" if os.name == "nt" else "maple-vision")
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONHOME", None)
        clean_env["PYTHONNOUSERSITE"] = "1"
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_env["ORT_DISABLE_TELEMETRY"] = "1"

        def execute(arguments: list[str]) -> str:
            result = subprocess.run(
                arguments,
                cwd=scratch,
                env=clean_env,
                text=True,
                capture_output=True,
                check=False,
                timeout=600,
            )
            if result.returncode:
                raise RuntimeError(
                    f"command failed ({result.returncode}): {arguments}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            return result.stdout

        def cli(*arguments: str) -> dict[str, Any]:
            return strict_json(execute([str(executable), *arguments]))

        execute(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "-r",
                str(checkout / "requirements-runtime.txt"),
            ]
        )
        execute(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                str(wheel),
            ]
        )
        execute([str(python), "-m", "pip", "check"])
        execute([str(executable), "--help"])
        window_check = binaries / (
            "maple-vision-window-check.exe" if os.name == "nt" else "maple-vision-window-check"
        )
        require(window_check.is_file(), "installed desktop window check launcher missing")
        window_help = execute([str(window_check), "--help"])
        require(
            "--title-exact" in window_help and "--handle" in window_help,
            "desktop window check help is incomplete",
        )
        checks["desktop_window_entrypoint"] = {"installed": True, "read_only_help": True}
        observation_check = binaries / (
            "maple-vision-window-observe.exe" if os.name == "nt" else "maple-vision-window-observe"
        )
        require(observation_check.is_file(), "installed desktop observation launcher missing")
        observation_help = execute([str(observation_check), "--help"])
        require(
            "--perception" in observation_help and "synthetic-fixture" in observation_help,
            "desktop observation help does not expose its fixture-only scope",
        )
        checks["desktop_observation_entrypoint"] = {
            "installed": True,
            "fixture_scope_explicit": True,
        }
        installed = strict_json(
            execute(
                [
                    str(python),
                    "-c",
                    "import json, importlib.metadata as m, importlib.util; "
                    "from pathlib import Path; from importlib.resources import files; "
                    "import maple_vision; "
                    "resource=files('maple_vision').joinpath('resources/tiny-yolo.yaml'); "
                    "terrain=files('maple_vision').joinpath('resources/terrain-yolo.yaml'); "
                    "print(json.dumps({'module':str(Path(maple_vision.__file__).resolve()), "
                    "'version':m.version('maple-vision-platform'), "
                    "'runtime':{n:m.version(n) for n in ('numpy','Pillow')}, "
                    "'model_config_packaged':resource.is_file() "
                    "and 'backbone:' in resource.read_text(), "
                    "'terrain_config_packaged':terrain.is_file() "
                    "and 'reg_max: 16' in terrain.read_text(), "
                    "'training_not_installed':importlib.util.find_spec('torch') is None, "
                    "'inference_not_installed':importlib.util.find_spec('onnxruntime') is None}))",
                ]
            )
        )
        require(not Path(installed["module"]).is_relative_to(checkout), "import used checkout")
        require(installed["model_config_packaged"], "tiny YOLO config missing from installed wheel")
        require(
            installed["terrain_config_packaged"], "terrain YOLO config missing from installed wheel"
        )
        require(installed["training_not_installed"], "isolated smoke unexpectedly contains torch")
        require(installed["inference_not_installed"], "isolated smoke unexpectedly contains ORT")
        checks["installed_package"] = installed
        practice = binaries / (
            "maple-vision-practice.exe" if os.name == "nt" else "maple-vision-practice"
        )
        require(practice.is_file(), "installed practice launcher missing")
        trial = strict_json(
            execute(
                [
                    str(python),
                    "-c",
                    """
import json
import tempfile
from pathlib import Path
from maple_vision.owned_route_playback import finalize_route_playback, verify_route_playback
from maple_vision.owned_session import OwnedSession
session = OwnedSession()
session.confirm("J")
route_dir = Path(tempfile.mkdtemp()) / "route"
execution = session.start_live_layered_route(route_dir)
require_empty = (route_dir / "trace.jsonl").read_text() == ""
shown = []
while execution.last_source_sequence != 72:
    item = execution.step()
    assert item is not None
    if item.source_sequence % 4 == 0 or item.labels:
        shown.append((item.source_sequence, item.image))
if shown[-1][0] != item.source_sequence:
    shown.append((item.source_sequence, item.image))
executed = execution.cancel("manual_takeover")
assert execution.review is not None
finalize_route_playback(
    route_dir,
    execution.review,
    tuple(shown),
    status="taken_over",
    reason="manual_takeover",
)
route_playback = verify_route_playback(route_dir)
session.arm()
receipt = session.press("J", focused=True)
session.release("J")
print(json.dumps({
    "receipt": receipt,
    "route": execution.review,
    "route_playback": route_playback,
    "started_without_plan": require_empty,
    "state": session.state,
}))
""",
                ]
            )
        )
        require(
            trial["state"] == "completed"
            and trial["receipt"]["hit_segments"] == 2
            and trial["route"]["status"] == "taken_over"
            and len(trial["route"]["navigation"]) == 1
            and [item["key"] for item in trial["route"]["casts"]] == ["J"]
            and trial["route_playback"]["status"] == "taken_over"
            and trial["route_playback"]["timeline_events"] == 3
            and trial["route_playback"]["execution"]["executed_steps"] == 73
            and trial["route_playback"]["execution"]["remaining_steps"] == "not_precomputed"
            and trial["started_without_plan"],
            "packaged practice trial failed",
        )
        checks["owned_practice_session"] = trial
        execute([str(executable), "model", "audit-terrain", "--help"])
        terrain = strict_json(
            execute(
                [
                    str(python),
                    "-c",
                    """
import json
from maple_vision.agent import HybridPerception
from maple_vision.contracts import Detection
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.vision import PixelPerception
import maple_vision.terrain_audit

objects = (Detection('player', (25, 146, 39, 168), 0.9),
           Detection('platform', (20, 168, 300, 172), 0.8))
class Detector:
    def detect(self, frame):
        return objects
def forbidden(self, frame):
    raise AssertionError('model terrain must not call palette object detection')
PixelPerception.detect = forbidden
observation = HybridPerception(Detector(), terrain_source='model').observe(
    SceneRenderer().render(Simulator().snapshot()))
assert observation.detections == objects
assert observation.source_id == 'onnx-all-objects+fixture-hud-v1'
print(json.dumps({'pipeline': observation.source_id, 'fixture_object_detection_called': False,
                  'object_count': len(observation.detections), 'native_extras_required': False}))
""",
                ]
            )
        )
        require(terrain["object_count"] == 2, "installed model-only terrain lost detector objects")
        checks["model_terrain_without_native_extras"] = terrain

        trace = scratch / "p0-trace.jsonl"
        p0 = cli("demo", "--frames", "3", "--output", str(trace))
        for field in (
            "frames_processed",
            "intents_emitted",
            "executions_skipped",
            "verifications_not_applicable",
        ):
            require(p0.get(field) == 3, f"P0 count mismatch: {field}")
        rows = [strict_json(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        require(len(rows) == 3 and all(row["schema_version"] == 1 for row in rows), "P0 trace")
        checks["p0_demo"] = {key: value for key, value in p0.items() if key != "output"}
        legacy = scratch / "legacy.sqlite"
        cli("db", "init", str(legacy))
        legacy_check = cli("db", "check", str(legacy))
        require(legacy_check["schema_version"] == 1, "legacy v1 API changed")
        require(legacy_check["source_records"] == 0, "legacy database must start empty")
        checks["p0_database"] = {key: value for key, value in legacy_check.items() if key != "path"}

        knowledge = scratch / "knowledge.sqlite"
        source = scratch / "source.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": [
                        {
                            "kind": "items",
                            "entity_id": "wheel.fixture",
                            "game_version": "wheel-smoke-v1",
                            "source_uri": "urn:maple-vision:wheel-smoke",
                            "collected_at": "2026-11-24T00:00:00Z",
                            "permission": "self_authored",
                            "license": "CC0-1.0",
                            "scope": "synthetic_fixture",
                            "verification_status": "VERIFIED",
                            "payload": {"label": "standalone wheel fixture"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        require(cli("knowledge", "init", str(knowledge))["schema_version"] == 2, "catalog schema")
        require(cli("knowledge", "import", str(knowledge), str(source))["imported"] == 1, "import")
        queried = cli(
            "knowledge", "query", str(knowledge), "items", "wheel.fixture", "wheel-smoke-v1"
        )
        require(len(queried["records"]) == 1, "catalog query must expose the verified fixture")
        snapshot = scratch / "snapshot"
        manifest = cli("knowledge", "snapshot", str(knowledge), str(snapshot))
        verified_snapshot = cli("knowledge", "verify-snapshot", str(snapshot))
        require(verified_snapshot.get("verified") is True, "snapshot verification status")
        require(
            all(verified_snapshot.get(key) == value for key, value in manifest.items()),
            "snapshot verification content",
        )
        checks["catalog_snapshot"] = manifest["summary"]

        sync_source = scratch / "sync-source.json"
        sync_record = json.loads(source.read_text(encoding="utf-8"))["records"][0]
        sync_record["entity_id"] = "wheel.synced-fixture"
        sync_record["payload"] = {"label": "standalone wheel sync fixture"}
        sync_source.write_text(
            json.dumps({"schema_version": 1, "records": [sync_record]}), encoding="utf-8"
        )
        sync_hash = hashlib.sha256(sync_source.read_bytes()).hexdigest()
        sync_plan = scratch / "sync-plan.json"
        planned = cli("knowledge", "plan-sync", str(knowledge), str(sync_source), str(sync_plan))
        require(planned["status"] == "planned", "sync plan status")
        require(planned["delta_counts"]["added"] == 1, "sync plan must add one fixture")
        require(planned["source_sha256"] == sync_hash, "sync plan source identity")
        require(
            not cli(
                "knowledge",
                "query",
                str(knowledge),
                "items",
                "wheel.synced-fixture",
                "wheel-smoke-v1",
            )["records"],
            "planning must not apply the new record",
        )
        sync_source.unlink()
        applied = cli("knowledge", "apply-sync", str(knowledge), str(sync_plan))
        require(applied["status"] == "applied" and applied["imported"] == 1, "sync application")
        require(applied["plan_id"] == planned["plan_id"], "sync applied a different plan")
        synced = cli(
            "knowledge", "query", str(knowledge), "items", "wheel.synced-fixture", "wheel-smoke-v1"
        )
        require(
            len(synced["records"]) == 1
            and strict_json(synced["records"][0]["payload_json"]) == sync_record["payload"],
            "sync result must be queryable from the installed catalog",
        )
        repeated = cli("knowledge", "apply-sync", str(knowledge), str(sync_plan))
        require(
            repeated["status"] == "already_applied" and repeated["imported"] == 0,
            "sync reapplication must be idempotent",
        )
        require(repeated["records_total"] == 2, "sync must retain the original fixture")
        checks["catalog_sync"] = {
            "plan_id": planned["plan_id"],
            "source_sha256": sync_hash,
            "delta_counts": planned["delta_counts"],
            "status": applied["status"],
            "imported": applied["imported"],
            "records_total": repeated["records_total"],
            "reapply_status": repeated["status"],
            "source_removed_before_apply": True,
            "verified_query": True,
        }

        dataset = scratch / "dataset"
        created = cli(
            "dataset", "create", str(dataset), "--sessions", "6", "--frames", "2", "--seed", "7"
        )
        require(created["validated"] and created["frames"] == 12, "dataset generation")
        validated = cli("dataset", "check", str(dataset))
        require(validated["sessions"] == 6 and validated["frames"] == 12, "dataset validation")
        checks["dataset"] = validated

        lifecycle = scratch / "lifecycle-dataset"
        lifecycle_created = cli(
            "dataset",
            "create",
            str(lifecycle),
            "--profile",
            "lifecycle",
            "--sessions",
            "3",
            "--frames",
            "12",
            "--seed",
            "7",
        )
        require(
            lifecycle_created["schema_version"] == 2
            and lifecycle_created["validated"]
            and lifecycle_created["frames"] == 36,
            "lifecycle dataset generation",
        )
        lifecycle_checked = cli("dataset", "check", str(lifecycle))
        require(
            lifecycle_checked["schema_version"] == 2
            and lifecycle_checked["sessions"] == 3
            and lifecycle_checked["frames"] == 36
            and lifecycle_checked["validated"],
            "lifecycle dataset validation",
        )
        phases = ("spawn", "combat", "loot", "cleared", "dialogue", "transition")
        phase_counts = {
            split: lifecycle_checked["coverage"]["by_split"][split]["phase_counts"]
            for split in ("train", "val", "test")
        }
        require(
            all(counts.get(phase, 0) > 0 for counts in phase_counts.values() for phase in phases),
            "lifecycle minimum data must cover all phases in every split",
        )
        require(
            lifecycle_checked["visual_agent_success_evaluated"] is False,
            "lifecycle data validation must not claim visual policy success",
        )
        checks["lifecycle_dataset"] = {
            "schema_version": lifecycle_checked["schema_version"],
            "profile": "lifecycle",
            "sessions": lifecycle_checked["sessions"],
            "frames": lifecycle_checked["frames"],
            "validated": lifecycle_checked["validated"],
            "phase_counts_by_split": phase_counts,
            "visual_agent_success_evaluated": False,
        }
        dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        sprite = dataset / dataset_manifest["samples"][0]["image"]
        registry = scratch / "registry"
        entry = cli(
            "assets",
            "import",
            str(registry),
            str(sprite),
            "--entity-id",
            "wheel.scene",
            "--game-version",
            "wheel-smoke-v1",
            "--source-uri",
            "urn:maple-vision:wheel-smoke",
            "--license",
            "CC0-1.0",
            "--permission",
            "self_authored",
        )
        require(
            cli("assets", "verify", str(registry), entry["asset_id"]) == entry, "asset verification"
        )
        checks["asset_registry"] = {"verified": True, "media_type": entry["media_type"]}

        mission_dir = scratch / "mission"
        mission = cli(
            "lab", "run", "--output", str(mission_dir), "--job", "assassin", "--seed", "0"
        )
        require(mission["status"] == "completed", "installed visual mission did not complete")
        require(mission["independent_goal_check"], "installed mission independent oracle failed")
        require(
            mission["maps_visited"] == ["cavern", "meadow"],
            "installed mission did not visit both maps",
        )
        replay = cli("lab", "replay", str(mission_dir))
        require(
            replay["verified"] and replay["steps_replayed"] == mission["steps"], "installed replay"
        )
        checks["mission"] = {
            key: mission[key]
            for key in (
                "status",
                "steps",
                "scope",
                "detector",
                "independent_goal_check",
                "maps_visited",
            )
        }
        checks["replay"] = replay
        before_reinference = {
            path.relative_to(mission_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in mission_dir.rglob("*")
            if path.is_file()
        }
        reinference = cli("lab", "reinfer", str(mission_dir))
        require(
            reinference["verified"]
            and reinference["perception_reproduced"]
            and reinference["policy_reproduced"]
            and reinference["runtime_identity_matches"],
            "installed pixel perception/policy re-inference",
        )
        require(
            reinference["trace_steps_compared"] == mission["steps"]
            and reinference["frames_reinferred"] == mission["steps"] + 1
            and reinference["terminal_observation_compared"]
            and reinference["terminal_decision_compared"],
            "installed re-inference must include every frame and terminal policy",
        )
        require(reinference["model_hash_verified"] is False, "pixel smoke is not ONNX validation")
        after_reinference = {
            path.relative_to(mission_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in mission_dir.rglob("*")
            if path.is_file()
        }
        require(
            before_reinference == after_reinference, "re-inference modified the original bundle"
        )
        checks["reinference"] = reinference
        structured_dir = scratch / "structured-mission"
        structured = cli(
            "lab",
            "run",
            "--output",
            str(structured_dir),
            "--runtime",
            "structured",
            "--record-stride",
            "32",
        )
        require(
            structured["schema_version"] == 2
            and structured["status"] == "completed"
            and structured["independent_goal_check"],
            "installed structured mission failed",
        )
        structured_reinfer = cli("lab", "reinfer", str(structured_dir))
        require(
            structured_reinfer["verified"]
            and structured_reinfer["structured_runtime_reproduced"]
            and structured_reinfer["belief_frames_recomputed"] == structured["steps"] + 1
            and structured_reinfer["terminal_runtime_compared"]
            and structured_reinfer["runtime_identity_matches"],
            "installed structured pipeline did not reproduce every belief and task",
        )
        checks["structured_runtime"] = {
            "status": structured["status"],
            "steps": structured["steps"],
            "independent_goal_check": structured["independent_goal_check"],
            "reinference": structured_reinfer,
        }
        # Author-supplied JSON is an input, while every implementation import and
        # CLI invocation remains in the isolated installed environment.
        map_source = scratch / "map-pack.json"
        shutil.copyfile(checkout / "examples/maps/three-map-fixture.json", map_source)
        map_check = cli("maps", "check", str(map_source))
        require(map_check["map_ids"] == ["grove", "quarry", "ruins"], "external map route")
        map_mission = scratch / "map-mission"
        external = cli(
            "lab",
            "run",
            "--output",
            str(map_mission),
            "--map-pack",
            str(map_source),
            "--record-stride",
            "64",
        )
        require(
            external["status"] == "completed" and external["independent_goal_check"],
            "installed three-map mission failed",
        )
        require(external["drops_collected"] == 9, "external oracle must derive nine monsters")
        map_dataset = scratch / "map-dataset"
        created = cli(
            "dataset",
            "create",
            str(map_dataset),
            "--profile",
            "lifecycle",
            "--sessions",
            "3",
            "--frames",
            "18",
            "--map-pack",
            str(map_source),
        )
        require(created["validated"], "installed external dataset validation failed")
        require(
            strict_json((map_dataset / "manifest.json").read_text(encoding="utf-8"))[
                "schema_version"
            ]
            == 3,
            "external dataset must use schema3",
        )
        map_source.unlink()
        require(cli("dataset", "check", str(map_dataset))["validated"], "portable map dataset")
        require(cli("lab", "replay", str(map_mission))["verified"], "portable map action replay")
        map_reinfer = cli("lab", "reinfer", str(map_mission))
        require(
            map_reinfer["perception_reproduced"]
            and map_reinfer["policy_reproduced"]
            and map_reinfer["frames_reinferred"] == external["steps"] + 1,
            "portable map perception reexecution",
        )
        checks["external_maps"] = {
            "map_ids": map_check["map_ids"],
            "content_sha256": map_check["map_pack"]["content_sha256"],
            "status": external["status"],
            "independent_goal_check": external["independent_goal_check"],
            "steps": external["steps"],
            "drops_collected": external["drops_collected"],
            "dataset_frames": created["frames"],
            "dataset_schema_version": 3,
            "source_deleted_before_verification": True,
            "frames_reinferred": map_reinfer["frames_reinferred"],
        }
        domain_source = scratch / "domain-source.json"
        shutil.copyfile(checkout / "fixtures/knowledge-domain-demo.json", domain_source)
        domain_database = scratch / "domain-catalog.sqlite"
        cli("knowledge", "init", str(domain_database))
        cli("knowledge", "import", str(domain_database), str(domain_source))
        domain_snapshot = scratch / "domain-source-snapshot"
        cli("knowledge", "snapshot", str(domain_database), str(domain_snapshot))
        domains = scratch / "domains"
        projection = cli("knowledge", "project-domains", str(domain_snapshot), str(domains))
        shutil.rmtree(domain_snapshot)
        domain_source.unlink()
        domain_database.unlink()
        require(cli("knowledge", "verify-domains", str(domains))["verified"], "domain replay")
        levels = cli(
            "knowledge",
            "query-domain",
            str(domains),
            "skill_levels",
            "fixture-v1",
            "--scope",
            "synthetic_fixture",
            "--entity-id",
            "throw",
        )["records"]
        require([row["mp_cost"] for row in levels] == [1, 2], "normalized skill levels")
        checks["catalog_domains"] = projection["summary"]

        asset_index = scratch / "asset-index.sqlite"
        cli("assets", "index", str(registry), str(asset_index), entry["asset_id"])
        require(
            cli("assets", "index-check", str(registry), str(asset_index))["verified"],
            "installed asset index",
        )
        animations = scratch / "animation-pack"
        shutil.copytree(checkout / "examples/animations/fixture", animations)
        animation_check = cli("animations", "check", str(animations))
        first_clip = animation_check["animation_pack"]["clips"][0]
        binding = scratch / "bindings.json"
        binding.write_text(json.dumps({first_clip["class_id"]: first_clip["clip_id"]}), "utf-8")
        animation_scene = scratch / "animation-scene"
        cli("animations", "scene", str(animations), str(binding), str(animation_scene))
        cli(
            "animations",
            "sample",
            str(animation_scene),
            "--clip",
            first_clip["clip_id"],
            "--elapsed-ms",
            "150",
            "--output",
            str(scratch / "animation.png"),
        )
        animated_data = scratch / "animation-dataset"
        animation_created = cli(
            "dataset",
            "create",
            str(animated_data),
            "--profile",
            "lifecycle",
            "--sessions",
            "3",
            "--frames",
            "12",
            "--animation-scene",
            str(animation_scene),
        )
        shutil.rmtree(animations)
        shutil.rmtree(animation_scene)
        require(
            animation_created["schema_version"] == 4
            and cli("dataset", "check", str(animated_data))["validated"],
            "portable schema4",
        )
        checks["animation_content"] = {
            "pack_verified": True,
            "native_sample_written": True,
            "dataset_schema": 4,
            "frames": animation_created["frames"],
        }

        class_source = scratch / "archer.json"
        shutil.copyfile(checkout / "examples/classes/archer.json", class_source)
        compiled = cli("classes", "check", str(class_source))
        class_run = scratch / "class-mission"
        class_result = cli(
            "lab",
            "run",
            "--output",
            str(class_run),
            "--class-pack",
            str(class_source),
            "--record-stride",
            "32",
        )
        class_source.unlink()
        require(
            class_result["status"] == "completed" and class_result["independent_goal_check"],
            "installed compiled-class mission",
        )
        class_reinfer = cli("lab", "reinfer", str(class_run))
        require(
            class_reinfer["structured_runtime_reproduced"]
            and class_reinfer["frames_reinferred"] == class_result["steps"] + 1,
            "installed external-class reexecution",
        )
        layered = scratch / "layered-bundle"
        cli("lab", "export-bundle", str(class_run), str(layered))
        shutil.rmtree(class_run)
        bundle_check = cli("lab", "verify-bundle", str(layered), "--reinfer")
        require(bundle_check["verified"], "portable layered replay without original run")
        checks["external_class_and_bundle"] = {
            "class_id": compiled["capabilities"]["class_id"],
            "status": class_result["status"],
            "steps": class_result["steps"],
            "frames_reinferred": class_reinfer["frames_reinferred"],
            "layered_bundle": bundle_check,
        }
        # Exercise python -m as well as the console entry point, without sending messages.
        draft = strict_json(
            execute(
                [
                    str(python),
                    "-m",
                    "maple_vision",
                    "chat-draft",
                    "--incoming",
                    "hello",
                    "--reply",
                    "local draft",
                ]
            )
        )
        require(
            draft["sent"] is False and draft["requires_human_confirmation"], "chat draft contract"
        )
        checks["module_entrypoint"] = {"passed": True}
    return {
        "schema_version": 1,
        "passed": True,
        "wheel": wheel.name,
        "wheel_sha256": wheel_hash,
        "python": platform.python_version(),
        "platform": platform.system(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "checks": checks,
        "scope": "isolated_installed_runtime; training_and_accuracy_are_separate_gates",
    }


def main() -> int:
    checkout = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="wheel to install; default: sole dist/*.whl")
    parser.add_argument("--output", type=Path, default=checkout / "artifacts" / "wheel-smoke.json")
    args = parser.parse_args()
    try:
        wheel = args.wheel
        if wheel is None:
            wheels = sorted((checkout / "dist").glob("*.whl"))
            require(len(wheels) == 1, "dist must contain exactly one wheel, or specify --wheel")
            wheel = wheels[0]
        report = smoke(wheel.resolve(), checkout)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        traceback.print_exc(file=sys.stderr)
        report = {"schema_version": 1, "passed": False, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
