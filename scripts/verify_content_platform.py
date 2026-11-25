"""Reproduce external-class, domain-catalog, animation and layered replay acceptance.

All sources are self-authored fixtures. This entry trains no model and performs
no new model scoring. A short wall-clock soak is explicitly separate from 24h.
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image

from maple_vision.animation_pack import (
    AnimationClip,
    AnimationFrame,
    AnimationPack,
    AnimationSource,
)
from maple_vision.asset_index import index_assets, verify_asset_index
from maple_vision.asset_registry import import_asset
from maple_vision.catalog import create_snapshot, import_catalog, initialize_catalog
from maple_vision.catalog_domains import build_domain_snapshot, query_domain, verify_domain_snapshot
from maple_vision.class_package import load_class_package
from maple_vision.dataset import generate_dataset, validate_dataset
from maple_vision.failure_bundle import export_failure_bundle, verify_failure_bundle
from maple_vision.lab import (
    FAULTS,
    CapabilityRunConfig,
    run_episode,
    run_soak,
    run_suite,
    runtime_identity,
    verify_perception,
    verify_run,
)
from maple_vision.map_pack import load_map_pack
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.scene import PALETTE
from maple_vision.scene_animation import (
    SceneAnimation,
    export_scene_animation,
    load_scene_animation,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def authored_animation(root: Path) -> SceneAnimation:
    """Two different real RGB frames retain the documented fixture color bounds."""
    root.mkdir()
    frames = []
    entries = []
    for index, duration in enumerate((30, 70)):
        image = Image.new("RGBA", (14, 22), (*PALETTE["player"], 255))
        if index:
            image.putpixel((0, 0), (0, 0, 0, 0))
            image.putpixel((13, 21), (0, 0, 0, 0))
        source = root / f"frame-{index}.png"
        image.save(source)
        entry = import_asset(
            root / "registry",
            source,
            entity_id="acceptance.player",
            game_version="fixture-v1",
            source_uri="urn:maple-vision:content-acceptance:self-authored",
            license="CC0-1.0",
            permission="self_authored",
        )
        entries.append(entry)
        frames.append(
            AnimationFrame(AnimationSource(entry, source.read_bytes()), duration, (7, 22))
        )
    index_assets(root / "index.sqlite", root / "registry", tuple(e.asset_id for e in entries))
    require(len(verify_asset_index(root / "index.sqlite", root / "registry")) == 2, "asset index")
    clip = AnimationClip(
        "acceptance.player",
        "acceptance.player",
        "fixture-v1",
        "player",
        "idle",
        "loop",
        tuple(frames),
    )
    scene = SceneAnimation(AnimationPack("synthetic_fixture", (clip,)), (("player", clip.clip_id),))
    export_scene_animation(scene, root / "scene")
    require(load_scene_animation(root / "scene") == scene, "animation portability")
    return scene


def verify(output: Path, checkout: Path, soak_seconds: int = 60) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("acceptance output already exists")
    if type(soak_seconds) is not int or not 1 <= soak_seconds <= 600:
        raise ValueError("short acceptance soak must be 1..600 seconds")
    identity = runtime_identity()
    started = time.perf_counter()
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "synthetic_content_platform_engineering_acceptance",
        "passed": False,
        "runtime_identity": identity,
        "platform": platform.platform(),
        "new_model_training": False,
        "new_model_scoring": False,
        "new_model_promotion": False,
        "real_domain_acceptance": "not_run",
        "24h_wall_soak": "not_run",
        "results": {},
    }
    try:
        database = output / "catalog.sqlite"
        initialize_catalog(database)
        import_catalog(database, checkout / "fixtures/knowledge-domain-demo.json")
        create_snapshot(database, output / "source-snapshot")
        domains = build_domain_snapshot(output / "source-snapshot", output / "domains")
        shutil.rmtree(output / "source-snapshot")
        database.unlink()
        require(verify_domain_snapshot(output / "domains") == domains, "domain reprojection")
        require(domains["summary"]["mapped_source_records"] == 14, "domain exact denominator")
        levels = query_domain(
            output / "domains",
            "skill_levels",
            "fixture-v1",
            scope="synthetic_fixture",
            entity_id="throw",
        )
        require([row["mp_cost"] for row in levels] == [1, 2], "typed skill-level query")
        report["results"]["catalog_domains"] = domains
        scene = authored_animation(output / "authoring")
        report["results"]["animation"] = scene.manifest()
        dataset = output / "animation-dataset"
        maps = load_map_pack(checkout / "examples/maps/three-map-fixture.json")
        created = generate_dataset(
            dataset,
            sessions=6,
            frames_per_session=18,
            seed=602511,
            profile="lifecycle",
            map_pack=maps,
            animation=scene,
        )
        require(created["schema_version"] == 4 and created["validated"], "animation dataset")
        require(validate_dataset(dataset)["frames"] == 108, "dataset exact frame denominator")
        report["results"]["animation_dataset"] = created
        packages = tuple(
            load_class_package(checkout / f"examples/classes/{job}.json")
            for job in ("archer", "bandit", "blood_knight")
        )
        report["class_packages"] = [package.manifest() for package in packages]
        runtime = RuntimeConfig()
        jobs = []
        for label, map_pack in (("builtin", None), ("external", maps)):
            for package in packages:
                destination = output / label / package.class_id
                result = run_episode(
                    destination,
                    CapabilityRunConfig(job=package.class_id, max_steps=1500, record_stride=32),
                    class_package=package,
                    map_pack=map_pack,
                    animation=scene,
                    runtime=runtime,
                )
                replay = verify_run(destination)
                reinfer = verify_perception(destination)
                require(
                    result["status"] == "completed" and result["independent_goal_check"],
                    f"mission failed: {label}/{package.class_id}",
                )
                require(
                    reinfer["frames_reinferred"] == result["steps"] + 1
                    and reinfer["structured_runtime_reproduced"],
                    "full pipeline reexecution",
                )
                jobs.append(
                    {
                        "name": f"{label}/{package.class_id}",
                        "run": result,
                        "action_replay": replay,
                        "reinfer": reinfer,
                    }
                )
                report["results"]["jobs"] = jobs
        shutil.rmtree(output / "authoring")
        report["results"]["animation_dataset_after_source_deletion"] = validate_dataset(dataset)
        require(verify_run(output / "external/archer")["verified"], "portable animation run")
        suites = []
        for label, map_pack in (("builtin", None), ("external", maps)):
            destination = output / f"{label}-suite"
            suite = run_suite(
                destination,
                seeds=3,
                map_pack=map_pack,
                class_packages=packages,
                animation=scene,
                runtime=runtime,
            )
            require(suite["passed"] and suite["episodes"] == 13, "13-case expected outcomes")
            faults = {fault: verify_perception(destination / fault) for fault in FAULTS[1:]}
            suites.append({"name": label, "suite": suite, "fault_reinfer": faults})
            report["results"]["suites"] = suites
        layered = output / "failure-bundle"
        exported = export_failure_bundle(output / "external-suite/freeze_input", layered)
        portable = output / "failure-bundle-portable"
        layered.rename(portable)
        checked = verify_failure_bundle(portable, reinfer=True)
        report["results"]["failure_bundle"] = {"export": exported, "portable_verify": checked}
        require(checked["verified"], "layered failure reprojection")
        report["results"]["short_soak"] = run_soak(
            output / "soak",
            seconds=soak_seconds,
            map_pack=maps,
            class_packages=packages,
            animation=scene,
            runtime=runtime,
        )
        require(report["results"]["short_soak"]["passed"], "short real wall-clock soak")
        require(runtime_identity() == identity, "implementation changed during acceptance")
        report["passed"] = True
    finally:
        report["actual_wall_seconds"] = round(time.perf_counter() - started, 6)
        report["source_tree_unchanged"] = runtime_identity() == identity
        write_json(output / "content-acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soak-seconds", type=int, default=60)
    args = parser.parse_args()
    report = verify(args.output, Path(__file__).resolve().parents[1], args.soak_seconds)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output": str(args.output),
                "wall_seconds": report["actual_wall_seconds"],
                "report_sha256": hashlib.sha256(
                    (args.output / "content-acceptance.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
