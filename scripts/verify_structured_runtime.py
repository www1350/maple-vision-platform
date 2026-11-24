"""Reproduce v0.5 temporal state/task acceptance without training any new model.

Pixel tests cover built-in and external maps. The previously accepted lifecycle
ONNX model is an integration regression on its original maps, not a new accuracy
or promotion experiment. Every failure keeps the original single-run bundle.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from maple_vision.lab import (
    FAULTS,
    RunConfig,
    run_episode,
    run_soak,
    run_suite,
    runtime_identity,
    verify_perception,
    verify_run,
)
from maple_vision.map_pack import load_map_pack
from maple_vision.onnx_detector import OnnxDetector
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import CLASS_SPECS

ACCEPTED_MODEL = "7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def verify(output: Path, source: Path, model_dir: Path, soak_seconds: int = 60) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"acceptance output already exists: {output}")
    if type(soak_seconds) is not int or not 1 <= soak_seconds <= 600:
        raise ValueError("acceptance soak_seconds must be in [1, 600]")
    pack = load_map_pack(source)
    model = model_dir / "model.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != ACCEPTED_MODEL:
        raise ValueError("integration regression requires the accepted v0.2 model")
    detector = OnnxDetector(model, model_dir / "manifest.json")
    runtime = RuntimeConfig()
    identity = runtime_identity()
    output.mkdir(parents=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "synthetic_structured_runtime_engineering_acceptance",
        "passed": False,
        "runtime_identity": identity,
        "runtime_settings": runtime.manifest(),
        "platform": platform.platform(),
        "new_model_training": False,
        "new_model_promotion": False,
        "real_domain_acceptance": "not_run",
        "24h_wall_soak": "not_run",
        "results": {},
    }
    try:
        jobs = []
        for label, maps, inference in (
            ("builtin-pixel", None, None),
            ("external-pixel", pack, None),
            ("accepted-hybrid", None, detector),
        ):
            for job in CLASS_SPECS:
                bundle = output / label / job
                summary = run_episode(
                    bundle,
                    RunConfig(job=job, max_steps=1500, record_stride=16),
                    inference,
                    map_pack=maps,
                    runtime=runtime,
                )
                replay = verify_run(bundle)
                reinfer = verify_perception(bundle, inference)
                row = {
                    "name": f"{label}/{job}",
                    "run": summary,
                    "action_replay": replay,
                    "reinfer": reinfer,
                }
                jobs.append(row)
                report["results"]["jobs"] = jobs
                if (
                    summary["status"] != "completed"
                    or not summary["independent_goal_check"]
                    or not reinfer["structured_runtime_reproduced"]
                    or reinfer["belief_frames_recomputed"] != summary["steps"] + 1
                ):
                    raise ValueError(f"structured mission failed: {row['name']}")
        suites = []
        for label, maps in (("builtin", None), ("external", pack)):
            destination = output / f"{label}-suite"
            suite = run_suite(destination, seeds=3, map_pack=maps, runtime=runtime)
            fault_reinference = {
                fault: verify_perception(destination / fault) for fault in FAULTS[1:]
            }
            suites.append({"name": label, "suite": suite, "fault_reinfer": fault_reinference})
            report["results"]["suites"] = suites
            if not suite["passed"]:
                raise ValueError(f"structured expected-outcome suite failed: {label}")
        soak = run_soak(output / "soak", seconds=soak_seconds, map_pack=pack, runtime=runtime)
        report["results"]["short_soak"] = soak
        if not soak["passed"]:
            raise ValueError("structured short soak failed")
        if runtime_identity() != identity:
            raise ValueError("implementation changed during formal acceptance")
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["actual_wall_seconds"] = round(time.perf_counter() - started, 6)
        write_json(output / "acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--map-pack", type=Path, default=Path("examples/maps/three-map-fixture.json")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("examples/lifecycle-model"))
    parser.add_argument("--soak-seconds", type=int, default=60)
    args = parser.parse_args()
    result = verify(args.output, args.map_pack, args.model_dir, args.soak_seconds)
    print(json.dumps({"passed": result["passed"], "report": str(args.output / "acceptance.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
