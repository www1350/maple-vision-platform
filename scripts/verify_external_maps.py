"""Reproduce the bounded v0.4 synthetic map acceptance with the pixel baseline.

Every output is a new directory. This never trains, loads or promotes a learned
model. Integrity checks can read every dataset split; only val is benchmarked.
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from maple_vision.dataset import generate_dataset
from maple_vision.evaluation import benchmark_dataset
from maple_vision.lab import (
    RunConfig,
    run_episode,
    run_soak,
    run_suite,
    runtime_identity,
    verify_perception,
    verify_run,
)
from maple_vision.map_pack import load_map_pack
from maple_vision.sim_types import CLASS_SPECS


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def verify(output: Path, source: Path, soak_seconds: int = 60) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"acceptance output already exists: {output}")
    if type(soak_seconds) is not int or not 1 <= soak_seconds <= 600:
        raise ValueError("acceptance soak_seconds must be in [1, 600]")
    pack = load_map_pack(source)
    output.mkdir(parents=True)
    started = time.perf_counter()
    identity = runtime_identity()
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "synthetic_external_map_pixel_pipeline",
        "passed": False,
        "map_pack": pack.manifest(),
        "runtime_identity": identity,
        "platform": platform.platform(),
        "learned_model_evaluated": False,
        "unseen_map_model_generalization_claimed": False,
        "24h_wall_soak": "not_run",
        "results": {},
    }
    try:
        # The path actually supplied to the loader disappears before all work.
        temporary_source = output / "temporary-input.json"
        shutil.copyfile(source, temporary_source)
        pack = load_map_pack(temporary_source)
        temporary_source.unlink()
        report["original_loader_path_removed"] = True
        rows = []
        for job in CLASS_SPECS:
            bundle = output / job
            result = run_episode(
                bundle,
                RunConfig(job=job, max_steps=max(1000, 500 * len(pack.map_ids)), record_stride=16),
                map_pack=pack,
            )
            replay = verify_run(bundle)
            reinfer = verify_perception(bundle)
            rows.append({"run": result, "action_replay": replay, "reinfer": reinfer})
            if result["status"] != "completed" or not result["independent_goal_check"]:
                raise ValueError(f"external map mission failed: {job}; original bundle retained")
        report["results"]["jobs"] = rows
        suite = run_suite(output / "suite", seeds=3, map_pack=pack)
        report["results"]["suite"] = suite
        if not suite["passed"]:
            raise ValueError("external map suite failed; original evidence retained")
        data = generate_dataset(
            output / "dataset",
            sessions=18,
            frames_per_session=9 * len(pack.map_ids),
            seed=20261125,
            profile="lifecycle",
            map_pack=pack,
        )
        report["results"]["dataset"] = data
        report["dataset_manifest_sha256"] = hashlib.sha256(
            (output / "dataset/manifest.json").read_bytes()
        ).hexdigest()
        benchmark = benchmark_dataset(
            output / "dataset", output / "pixel-validation.json", split="val"
        )
        groups = benchmark["grouped_metrics"]
        if not isinstance(groups, dict) or set(groups["by_map"]) != set(pack.map_ids):
            raise ValueError("benchmark omitted an authored map group")
        report["results"]["pixel_validation"] = benchmark
        report["results"]["soak"] = run_soak(output / "soak", seconds=soak_seconds, map_pack=pack)
        if not report["results"]["soak"]["passed"]:
            raise ValueError("external map short soak failed; original evidence retained")
        if runtime_identity() != identity:
            raise ValueError("implementation changed during the acceptance run")
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
    parser.add_argument("--soak-seconds", type=int, default=60)
    args = parser.parse_args()
    report = verify(args.output, args.map_pack, args.soak_seconds)
    print(json.dumps({"passed": report["passed"], "report": str(args.output / "acceptance.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
