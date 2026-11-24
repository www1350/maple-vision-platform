"""Exercise an installed wheel in a clean, runtime-only environment outside the checkout.

Only this script's standard-library imports run in the build environment. All package
checks run as subprocesses in a separate virtual environment, and stdout is one JSON
report. It intentionally does not train a model; CI has a separate opt-in CPU smoke.
"""

import argparse
import hashlib
import json
import os
import platform
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
        installed = strict_json(
            execute(
                [
                    str(python),
                    "-c",
                    "import json, importlib.metadata as m, importlib.util; "
                    "from pathlib import Path; from importlib.resources import files; "
                    "import maple_vision; "
                    "resource=files('maple_vision').joinpath('resources/tiny-yolo.yaml'); "
                    "print(json.dumps({'module':str(Path(maple_vision.__file__).resolve()), "
                    "'version':m.version('maple-vision-platform'), "
                    "'runtime':{n:m.version(n) for n in ('numpy','Pillow')}, "
                    "'model_config_packaged':resource.is_file() "
                    "and 'backbone:' in resource.read_text(), "
                    "'training_not_installed':importlib.util.find_spec('torch') is None, "
                    "'inference_not_installed':importlib.util.find_spec('onnxruntime') is None}))",
                ]
            )
        )
        require(not Path(installed["module"]).is_relative_to(checkout), "import used checkout")
        require(installed["model_config_packaged"], "tiny YOLO config missing from installed wheel")
        require(installed["training_not_installed"], "isolated smoke unexpectedly contains torch")
        require(installed["inference_not_installed"], "isolated smoke unexpectedly contains ORT")
        checks["installed_package"] = installed

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

        dataset = scratch / "dataset"
        created = cli(
            "dataset", "create", str(dataset), "--sessions", "6", "--frames", "2", "--seed", "7"
        )
        require(created["validated"] and created["frames"] == 12, "dataset generation")
        validated = cli("dataset", "check", str(dataset))
        require(validated["sessions"] == 6 and validated["frames"] == 12, "dataset validation")
        checks["dataset"] = validated
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
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        traceback.print_exc(file=sys.stderr)
        report = {"schema_version": 1, "passed": False, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
