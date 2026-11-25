"""Offline knowledge, vision, simulation and reproducible validation commands."""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import RuntimeConfig, load_config
from .contracts import SCHEMA_VERSION
from .knowledge import initialize, open_readonly
from .pipeline import (
    FreshnessEstimator,
    NoOpController,
    NoOpPerception,
    NoOpPolicy,
    NoOpVerifier,
    Pipeline,
)
from .sources import DemoSource
from .telemetry import write_tick


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maple-vision", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run deterministic NoOp foundation pipeline")
    demo.add_argument("--config", type=Path)
    demo.add_argument("--frames", type=int)
    demo.add_argument("--output", type=Path, default=Path("p0-trace.jsonl"))
    database = commands.add_parser("db", help="initialize or inspect a knowledge database")
    db_commands = database.add_subparsers(dest="db_command", required=True)
    for command in ("init", "check"):
        subparser = db_commands.add_parser(command)
        subparser.add_argument("path", type=Path)
    knowledge = commands.add_parser("knowledge", help="versioned offline catalog and snapshots")
    knowledge_commands = knowledge.add_subparsers(dest="operation", required=True)
    for operation in (
        "init",
        "check",
        "import",
        "query",
        "snapshot",
        "verify-snapshot",
        "plan-sync",
        "apply-sync",
    ):
        subparser = knowledge_commands.add_parser(operation)
        subparser.add_argument("path", type=Path)
        if operation in {"import", "plan-sync"}:
            subparser.add_argument("source", type=Path)
            if operation == "plan-sync":
                subparser.add_argument("output", type=Path)
        elif operation == "apply-sync":
            subparser.add_argument("plan", type=Path)
        elif operation == "query":
            subparser.add_argument("kind")
            subparser.add_argument("entity_id")
            subparser.add_argument("game_version")
            subparser.add_argument("--allow-unverified", action="store_true")
        elif operation == "snapshot":
            subparser.add_argument("output", type=Path)
    fetch = knowledge_commands.add_parser(
        "fetch-plan", help="capture a permitted HTTPS JSON export for offline review"
    )
    fetch.add_argument("path", type=Path)
    fetch.add_argument("url")
    fetch.add_argument("output", type=Path)
    fetch.add_argument("--permission", required=True)
    fetch.add_argument("--license", required=True)
    fetch.add_argument("--sha256")
    for operation in ("project-domains", "verify-domains", "query-domain"):
        subparser = knowledge_commands.add_parser(operation)
        subparser.add_argument("path", type=Path)
        if operation == "project-domains":
            subparser.add_argument("output", type=Path)
        elif operation == "query-domain":
            subparser.add_argument("table")
            subparser.add_argument("game_version")
            subparser.add_argument("--scope", required=True)
            subparser.add_argument("--entity-id")
            subparser.add_argument("--limit", type=int, default=1000)
    assets = commands.add_parser("assets", help="authorized local asset registry")
    asset_commands = assets.add_subparsers(dest="operation", required=True)
    for operation in ("import", "list", "verify"):
        subparser = asset_commands.add_parser(operation)
        subparser.add_argument("registry", type=Path)
        if operation == "import":
            subparser.add_argument("source", type=Path)
            for flag in ("entity-id", "game-version", "source-uri", "license", "permission"):
                subparser.add_argument(f"--{flag}", required=True)
        elif operation == "verify":
            subparser.add_argument("asset_id")
    for operation in ("index", "index-check", "index-query"):
        subparser = asset_commands.add_parser(operation)
        subparser.add_argument("registry", type=Path)
        subparser.add_argument("database", type=Path)
        if operation == "index":
            subparser.add_argument("asset_ids", nargs="+")
        elif operation == "index-query":
            subparser.add_argument("entity_id")
            subparser.add_argument("game_version")
            subparser.add_argument("--after-asset-id")
            subparser.add_argument("--limit", type=int, default=256)
    animations = commands.add_parser("animations", help="versioned local animation content")
    animation_commands = animations.add_subparsers(dest="operation", required=True)
    for operation in ("build", "check", "scene", "sample"):
        subparser = animation_commands.add_parser(operation)
        subparser.add_argument("path", type=Path)
        if operation in {"build", "scene"}:
            subparser.add_argument("mapping", type=Path)
            subparser.add_argument("output", type=Path)
        if operation == "sample":
            subparser.add_argument("--clip", required=True)
            subparser.add_argument("--elapsed-ms", type=int, default=0)
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--size", type=int, nargs=2, default=(96, 96))
            subparser.add_argument("--position", type=int, nargs=2, default=(48, 64))
            subparser.add_argument("--flip-x", action="store_true")
    maps = commands.add_parser("maps", help="validate a bounded offline synthetic map pack")
    map_commands = maps.add_subparsers(dest="operation", required=True)
    map_check = map_commands.add_parser("check")
    map_check.add_argument("path", type=Path)
    classes = commands.add_parser("classes", help="validate and compile local class packages")
    class_commands = classes.add_subparsers(dest="operation", required=True)
    class_check = class_commands.add_parser("check")
    class_check.add_argument("path", type=Path)
    dataset = commands.add_parser(
        "dataset", help="generate or verify session-split fixture datasets"
    )
    dataset_commands = dataset.add_subparsers(dest="operation", required=True)
    for operation in ("create", "check"):
        subparser = dataset_commands.add_parser(operation)
        subparser.add_argument("path", type=Path)
        if operation == "create":
            subparser.add_argument("--sessions", type=int, default=9)
            subparser.add_argument(
                "--frames", type=int, help="frames per session; default legacy=8, lifecycle=48"
            )
            subparser.add_argument("--seed", type=int, default=0)
            subparser.add_argument("--profile", choices=("legacy", "lifecycle"), default="legacy")
            subparser.add_argument("--asset-registry", type=Path)
            subparser.add_argument("--sprite-map", type=Path)
            subparser.add_argument("--map-pack", type=Path)
            subparser.add_argument("--animation-scene", type=Path)
    vision = commands.add_parser("vision", help="observe local images or benchmark a dataset")
    vision_commands = vision.add_subparsers(dest="operation", required=True)
    for operation in ("observe", "benchmark"):
        subparser = vision_commands.add_parser(operation)
        subparser.add_argument("path", type=Path)
        subparser.add_argument("--model", type=Path)
        subparser.add_argument("--manifest", type=Path)
        if operation == "benchmark":
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--split", choices=("train", "val", "test"), default="test")
            subparser.add_argument(
                "--terrain-metrics",
                action="store_true",
                help="include full-width floor recall and matched terrain edge errors",
            )
        else:
            subparser.add_argument(
                "--terrain-source", choices=("fixture", "model"), default="fixture"
            )
            subparser.add_argument("--map-pack", type=Path)
    ocr = vision_commands.add_parser("ocr", help="read text locally with installed Tesseract")
    ocr.add_argument("path", type=Path)
    ocr.add_argument("--language", default="eng")
    ocr.add_argument("--binary", type=Path)
    ocr.add_argument("--tessdata-dir", type=Path)
    ocr.add_argument("--timeout", type=float, default=5.0)
    ocr.add_argument("--crop", type=int, nargs=4)
    recording = commands.add_parser(
        "recording", help="import and observe authorized local recordings"
    )
    recording_commands = recording.add_subparsers(dest="operation", required=True)
    recording_import = recording_commands.add_parser("import")
    recording_import.add_argument("source", type=Path)
    recording_import.add_argument("output", type=Path)
    recording_import.add_argument("--source-uri", required=True)
    recording_import.add_argument("--permission", required=True)
    recording_import.add_argument("--license", required=True)
    recording_import.add_argument("--presentation-time", action="store_true")
    recording_import.add_argument("--fps", type=float)
    recording_import.add_argument("--max-frames", type=int, default=300)
    recording_import.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="zero-based source frame at which this bounded import window begins",
    )
    label_command = recording_commands.add_parser("label-template")
    label_command.add_argument("path", type=Path)
    label_command.add_argument("output", type=Path)
    audit = recording_commands.add_parser("audit-route")
    for name in ("path", "labels", "graph", "output"):
        audit.add_argument(name, type=Path)
    strike = recording_commands.add_parser("measure-strike")
    for name in ("path", "labels", "output"):
        strike.add_argument(name, type=Path)
    for name in ("start", "end", "player-index", "monster-index", "level"):
        strike.add_argument("--" + name, type=int, required=True)
    for name in ("skill-id", "weapon", "reviewer"):
        strike.add_argument("--" + name, required=True)
    strike.add_argument("--facing", choices=("left", "right"), required=True)
    strike.add_argument(
        "--feedback", choices=("damage_visible", "miss_text_visible", "no_feedback"), required=True
    )
    strike.add_argument(
        "--feedback-frame",
        type=int,
        action="append",
        default=[],
        help="reviewed frame containing a distinct feedback event; repeat in time order",
    )
    strike.add_argument(
        "--movement-frame", type=int, help="reviewed first visible movement after final feedback"
    )
    strike.add_argument(
        "--weapon-speed-grade",
        type=int,
        choices=range(2, 10),
        help="reviewed weapon speed grade (2 fastest through 9 slowest)",
    )
    strike.add_argument(
        "--speed-effect",
        action="append",
        default=[],
        help="reviewed attack-speed effect; repeat for each active effect",
    )
    strike.add_argument(
        "--input-profile",
        type=Path,
        help="reviewed local key-binding profile to bind into the evidence",
    )
    strike.add_argument(
        "--input-key",
        help="configured key used for this cast; requires --input-profile",
    )
    verify_strike_command = recording_commands.add_parser("verify-strike")
    for name in ("path", "labels", "report"):
        verify_strike_command.add_argument(name, type=Path)
    strike_summary = recording_commands.add_parser("summarize-strikes")
    strike_inputs = strike_summary.add_mutually_exclusive_group(required=True)
    strike_inputs.add_argument("--report", type=Path, action="append")
    strike_inputs.add_argument(
        "--verified-case",
        type=Path,
        nargs=3,
        action="append",
        metavar=("RECORDING", "LABELS", "REPORT"),
    )
    strike_summary.add_argument("--output", type=Path, required=True)
    event_review = recording_commands.add_parser("review-events")
    event_review.add_argument("path", type=Path)
    event_review.add_argument("output", type=Path)
    event_review.add_argument("--reviewer", required=True)
    event_review.add_argument(
        "--event",
        action="append",
        nargs=3,
        required=True,
        metavar=("KIND", "START", "END"),
        help="reviewed visible event kind and inclusive frame interval; repeat as needed",
    )
    verify_events_command = recording_commands.add_parser("verify-events")
    for name in ("path", "report"):
        verify_events_command.add_argument(name, type=Path)
    window_summary = recording_commands.add_parser("summarize-windows")
    window_summary.add_argument("--session", type=Path, action="append", required=True)
    window_summary.add_argument("--output", type=Path, required=True)
    window_verify = recording_commands.add_parser("verify-windows")
    window_verify.add_argument("report", type=Path)
    window_verify.add_argument("--session", type=Path, action="append", required=True)
    score = recording_commands.add_parser("benchmark")
    score.add_argument(
        "--case",
        dest="cases",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("RECORDING", "LABELS"),
    )
    score.add_argument("--development-dataset", type=Path, action="append", default=[])
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--confidence", type=float, default=0.25)
    score.add_argument("--model", type=Path)
    score.add_argument("--manifest", type=Path)
    recording_check = recording_commands.add_parser("check")
    recording_check.add_argument("path", type=Path)
    recording_observe = recording_commands.add_parser("observe")
    recording_observe.add_argument("path", type=Path)
    recording_observe.add_argument("--output", type=Path, required=True)
    recording_observe.add_argument("--model", type=Path)
    recording_observe.add_argument("--manifest", type=Path)
    model = commands.add_parser("model", help="local-only CPU YOLO training and ONNX export")
    model_commands = model.add_subparsers(dest="operation", required=True)
    train = model_commands.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "resources" / "tiny-yolo.yaml"
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--image-size", type=int, default=160)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    export = model_commands.add_parser("export")
    export.add_argument("weights", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--image-size", type=int, default=160)
    export.add_argument(
        "--scope",
        choices=("synthetic_fixture", "authorized_recordings"),
        default="synthetic_fixture",
    )
    audit = model_commands.add_parser(
        "audit-terrain", help="audit a local detection head before training; train/val only"
    )
    audit.add_argument("dataset", type=Path)
    audit.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "resources" / "terrain-yolo.yaml"
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "val"), default="val")
    audit.add_argument("--image-size", type=int, default=320)
    audit.add_argument("--warmup", type=int, default=3)
    audit.add_argument("--iterations", type=int, default=20)
    audit.add_argument("--seed", type=int, default=42)
    lab = commands.add_parser("lab", help="simulator-only visual missions, fault tests and replay")
    lab_commands = lab.add_subparsers(dest="operation", required=True)
    run = lab_commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--job", help="builtin class, or the exact class ID in --class-pack")
    run.add_argument("--class-pack", type=Path)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-steps", type=int, default=1000)
    run.add_argument("--record-stride", type=int, default=4)
    run.add_argument(
        "--fault",
        choices=("none", "brief_dropout", "persistent_dropout", "duplicate_frames", "freeze_input"),
        default="none",
    )
    run.add_argument("--model", type=Path)
    run.add_argument("--manifest", type=Path)
    run.add_argument("--knowledge-db", type=Path)
    run.add_argument("--game-version", default="fixture-v1")
    run.add_argument("--asset-registry", type=Path)
    run.add_argument("--sprite-map", type=Path)
    run.add_argument(
        "--input-bindings",
        type=Path,
        help="versioned role key profile for offline simulated skill dispatch",
    )
    suite = lab_commands.add_parser("suite")
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--seeds", type=int, default=3)
    replay = lab_commands.add_parser("replay")
    replay.add_argument("path", type=Path)
    reinfer = lab_commands.add_parser(
        "reinfer", help="re-run every recorded perception and policy decision with matching inputs"
    )
    reinfer.add_argument("path", type=Path)
    export_bundle = lab_commands.add_parser(
        "export-bundle", help="capture a self-contained layered replay or failure bundle"
    )
    export_bundle.add_argument("path", type=Path)
    export_bundle.add_argument("output", type=Path)
    export_bundle.add_argument("--video", action="store_true")
    verify_bundle = lab_commands.add_parser(
        "verify-bundle", help="replay and independently reproject every recorded evidence layer"
    )
    verify_bundle.add_argument("path", type=Path)
    for bundle_command in (export_bundle, verify_bundle):
        bundle_command.add_argument("--reinfer", action="store_true")
    soak = lab_commands.add_parser("soak")
    soak.add_argument("--output", type=Path, required=True)
    soak.add_argument("--seconds", type=int, default=60)
    soak.add_argument("--resource-interval", type=float, default=1.0)
    soak.add_argument("--max-resource-samples", type=int, default=1024)
    for model_command in (suite, soak, reinfer, export_bundle, verify_bundle):
        model_command.add_argument("--model", type=Path)
        model_command.add_argument("--manifest", type=Path)
    for class_command in (suite, soak):
        class_command.add_argument("--class-pack", type=Path, action="append")
    for model_command in (run, suite, soak, reinfer, export_bundle, verify_bundle):
        model_command.add_argument(
            "--terrain-source", choices=("fixture", "model"), default="fixture"
        )
    for map_command in (run, suite, soak):
        map_command.add_argument("--map-pack", type=Path)
        map_command.add_argument("--mission-program", type=Path)
        map_command.add_argument("--animation-scene", type=Path)
        map_command.add_argument(
            "--runtime",
            choices=("legacy", "structured"),
            help="default: structured with --class-pack, otherwise legacy",
        )
    chat = commands.add_parser(
        "chat-draft", help="prepare a local human-reviewed reply; sends nothing"
    )
    chat.add_argument("--incoming", required=True)
    chat.add_argument("--reply", required=True)
    bench = commands.add_parser("workbench", help="start the local simulator workbench")
    bench.add_argument("--workspace", type=Path, required=True)
    bench.add_argument("--examples", type=Path)
    bench.add_argument("--port", type=int, default=8765)
    world = map_commands.add_parser("route", help="plan a weighted route between maps")
    world.add_argument("path", type=Path)
    world.add_argument("--map-pack", action="store_true")
    world.add_argument("source")
    world.add_argument("target")
    platform = map_commands.add_parser("platform-route", help="search a fixed-step movement graph")
    platform.add_argument("geometry", type=Path)
    platform.add_argument("movement", type=Path)
    platform.add_argument("--start", type=float, nargs=2, required=True)
    platform.add_argument("--target", required=True)
    return parser


def _demo(config: RuntimeConfig, output: Path) -> dict[str, str | int]:
    pipeline = Pipeline(
        DemoSource(config.frames, config.frame_interval_ms),
        NoOpPerception(),
        FreshnessEstimator(config.max_observation_age_ms),
        NoOpPolicy(),
        NoOpController(),
        NoOpVerifier(),
    )
    count = 0
    run_id = uuid4().hex
    # Exclusive creation preserves prior evidence. Parent directories must exist.
    with output.open("x", encoding="utf-8") as stream:
        while (result := pipeline.tick()) is not None:
            write_tick(stream, run_id, result)
            count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "frames_processed": count,
        "intents_emitted": count,
        "executions_skipped": count,
        "verifications_not_applicable": count,
        "output": str(output),
    }


def _database(command: str, path: Path) -> dict[str, str | int]:
    if command == "init":
        initialize(path)
    connection = open_readonly(path)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        count = int(connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0])
        return {"path": str(path), "schema_version": version, "source_records": count}
    finally:
        connection.close()


def _extended(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "maps" and args.operation == "route":
        from .map_pack import load_map_pack
        from .navigation_graph import WorldMapGraph, load_world_graph

        world = (
            WorldMapGraph.from_map_pack(load_map_pack(args.path))
            if args.map_pack
            else load_world_graph(args.path)
        )
        route = world.shortest_path(args.source, args.target)
        return {
            "found": route is not None,
            "route": None if route is None else route.to_dict(),
            "verification": None if route is None else world.verify_route(route),
        }
    if args.command == "maps" and args.operation == "platform-route":
        from .navigation_graph import (
            MotionState,
            build_platform_graph,
            load_movement_capabilities,
            load_platform_geometry,
        )

        graph = build_platform_graph(
            load_platform_geometry(args.geometry), load_movement_capabilities(args.movement)
        )
        result = graph.find_path(MotionState(*args.start), args.target)
        return {
            "search": result.to_dict(),
            "verification": None if result.path is None else graph.verify_path(result.path),
        }
    if args.command == "knowledge":
        if args.operation == "fetch-plan":
            from .knowledge_sync import HttpsExportSource, plan_from_source

            return plan_from_source(
                args.path,
                HttpsExportSource(args.url, args.permission, args.license, args.sha256),
                args.output,
            )
        if args.operation in {"project-domains", "verify-domains", "query-domain"}:
            from .catalog_domains import (
                build_domain_snapshot,
                query_domain,
                verify_domain_snapshot,
            )

            if args.operation == "project-domains":
                return build_domain_snapshot(args.path, args.output)
            if args.operation == "verify-domains":
                return {"verified": True, **verify_domain_snapshot(args.path)}
            return {
                "records": query_domain(
                    args.path,
                    args.table,
                    args.game_version,
                    scope=args.scope,
                    entity_id=args.entity_id,
                    limit=args.limit,
                )
            }
        if args.operation in {"plan-sync", "apply-sync"}:
            from .knowledge_sync import apply_sync, plan_sync

            if args.operation == "plan-sync":
                return plan_sync(args.path, args.source, args.output)
            return apply_sync(args.path, args.plan)
        from .catalog import (
            catalog_summary,
            create_snapshot,
            import_catalog,
            initialize_catalog,
            query_catalog,
            verify_snapshot,
        )

        if args.operation == "init":
            initialize_catalog(args.path)
        if args.operation in {"init", "check"}:
            return catalog_summary(args.path)
        if args.operation == "import":
            return asdict(import_catalog(args.path, args.source))
        if args.operation == "query":
            return {
                "records": [
                    asdict(record)
                    for record in query_catalog(
                        args.path,
                        args.kind,
                        args.entity_id,
                        args.game_version,
                        allow_unverified=args.allow_unverified,
                    )
                ]
            }
        if args.operation == "snapshot":
            return create_snapshot(args.path, args.output)
        return {"verified": True, **verify_snapshot(args.path)}
    if args.command == "assets":
        if args.operation in {"index", "index-check", "index-query"}:
            from .asset_index import index_assets, query_assets, verify_asset_index

            if args.operation == "index":
                return asdict(index_assets(args.database, args.registry, tuple(args.asset_ids)))
            indexed = (
                verify_asset_index(args.database, args.registry)
                if args.operation == "index-check"
                else query_assets(
                    args.database,
                    args.registry,
                    entity_id=args.entity_id,
                    game_version=args.game_version,
                    after_asset_id=args.after_asset_id,
                    limit=args.limit,
                )
            )
            return {"verified": True, "assets": [asdict(entry) for entry in indexed]}
        from .asset_registry import import_asset, list_assets, verify_asset

        if args.operation == "import":
            return asdict(
                import_asset(
                    args.registry,
                    args.source,
                    entity_id=args.entity_id,
                    game_version=args.game_version,
                    source_uri=args.source_uri,
                    license=args.license,
                    permission=args.permission,
                )
            )
        if args.operation == "verify":
            return asdict(verify_asset(args.registry, args.asset_id))
        return {"assets": [asdict(item) for item in list_assets(args.registry)]}
    if args.command == "animations":
        from .animation_pack import (
            AnimationPlacement,
            compose_animation,
            export_animation_pack,
            load_animation_pack,
            verify_animation_pack,
        )
        from .scene_animation import (
            bind_scene_animation,
            export_scene_animation,
            load_scene_animation,
        )

        if args.operation == "build":
            return export_animation_pack(load_animation_pack(args.path, args.mapping), args.output)
        scene = (
            load_scene_animation(args.path)
            if (args.path / "scene-animation.json").exists()
            else None
        )
        animation_pack = scene.pack if scene is not None else verify_animation_pack(args.path)
        if args.operation == "scene":
            from .catalog import strict_json

            if args.mapping.is_symlink() or not args.mapping.is_file():
                raise ValueError("invalid scene binding file")
            with args.mapping.open("rb") as stream:
                binding_bytes = stream.read(4097)
            if len(binding_bytes) > 4096:
                raise ValueError("scene binding file exceeds 4KiB")
            bindings = strict_json(binding_bytes.decode("utf-8"))
            return export_scene_animation(
                bind_scene_animation(animation_pack, bindings), args.output
            )
        if args.operation == "sample":
            rendered = compose_animation(
                animation_pack,
                (
                    AnimationPlacement(
                        args.clip,
                        args.elapsed_ms,
                        tuple(args.position),
                        flip_x=args.flip_x,
                    ),
                ),
                size=tuple(args.size),
            )
            rendered.save_png(args.output)
            return {
                "output": str(args.output),
                "placements": [asdict(p) for p in rendered.placements],
            }
        return {
            "verified": True,
            "animation_pack": animation_pack.manifest(),
            "scene": None if scene is None else scene.manifest(),
        }
    if args.command == "classes":
        from .class_package import compile_class_package, load_class_package

        class_content = load_class_package(args.path)
        compiled = compile_class_package(class_content)
        return {
            "verified": True,
            "validation_scope": "synthetic_package_structure_and_compilation; mission_not_run",
            "class_package": class_content.manifest(),
            "capabilities": compiled.to_dict(),
        }
    if args.command == "maps":
        from .map_pack import load_map_pack

        checked_pack = load_map_pack(args.path)
        return {
            "verified": True,
            "validation_scope": "bounded_structure_and_identity; task_solvability_not_proven",
            "map_ids": list(checked_pack.map_ids),
            "map_pack": checked_pack.manifest(),
        }
    if args.command == "dataset":
        from .dataset import generate_dataset, validate_dataset

        if args.operation == "create":
            from .map_pack import load_map_pack
            from .scene_animation import load_scene_animation
            from .sprite_pack import load_sprite_pack

            if bool(args.asset_registry) != bool(args.sprite_map):
                raise ValueError("--asset-registry and --sprite-map must be supplied together")
            sprites = (
                load_sprite_pack(args.asset_registry, args.sprite_map)
                if args.asset_registry
                else None
            )
            return generate_dataset(
                args.path,
                args.sessions,
                args.frames
                if args.frames is not None
                else (48 if args.profile == "lifecycle" else 8),
                args.seed,
                sprite_pack=sprites,
                profile=args.profile,
                map_pack=load_map_pack(args.map_pack) if args.map_pack else None,
                animation=load_scene_animation(args.animation_scene)
                if args.animation_scene
                else None,
            )
        return validate_dataset(args.path)
    if args.command == "recording" and args.operation == "measure-strike":
        from .input_bindings import load_input_profile
        from .recording_evaluation import measure_strike

        return measure_strike(
            args.path,
            args.labels,
            args.output,
            start=args.start,
            end=args.end,
            player_index=args.player_index,
            monster_index=args.monster_index,
            level=args.level,
            skill_id=args.skill_id,
            weapon=args.weapon,
            reviewer=args.reviewer,
            facing=args.facing,
            feedback=args.feedback,
            feedback_frames=args.feedback_frame,
            movement_frame=args.movement_frame,
            weapon_speed_grade=args.weapon_speed_grade,
            speed_effects=args.speed_effect,
            input_profile=load_input_profile(args.input_profile) if args.input_profile else None,
            input_key=args.input_key,
        )
    if args.command == "recording" and args.operation == "verify-strike":
        from .recording_evaluation import verify_strike

        return verify_strike(args.path, args.labels, args.report)
    if args.command == "recording" and args.operation == "summarize-strikes":
        from .recording_evaluation import summarize_strikes

        if args.verified_case:
            return summarize_strikes(
                [report for _, _, report in args.verified_case],
                args.output,
                sources=[(recording, labels) for recording, labels, _ in args.verified_case],
            )
        return summarize_strikes(args.report, args.output)
    if args.command == "recording" and args.operation == "review-events":
        from .recording_evaluation import review_events

        return review_events(
            args.path,
            args.output,
            reviewer=args.reviewer,
            events=[(kind, int(start), int(end)) for kind, start, end in args.event],
        )
    if args.command == "recording" and args.operation == "verify-events":
        from .recording_evaluation import verify_events

        return verify_events(args.path, args.report)
    if args.command == "recording" and args.operation == "summarize-windows":
        from .recording_evaluation import summarize_recording_windows

        return summarize_recording_windows(args.session, args.output)
    if args.command == "recording" and args.operation == "verify-windows":
        from .recording_evaluation import verify_recording_windows

        return verify_recording_windows(args.session, args.report)
    if args.command == "recording" and args.operation == "audit-route":
        from .recording_evaluation import audit_route

        return audit_route(args.path, args.labels, args.graph, args.output)
    if args.command == "recording" and args.operation == "label-template":
        from .recording_evaluation import create_label_template

        return create_label_template(args.path, args.output)
    if args.command == "recording" and args.operation not in {"observe", "benchmark"}:
        from .recordings import import_recording, validate_recording

        if args.operation == "import":
            return import_recording(
                args.source,
                args.output,
                source_uri=args.source_uri,
                permission=args.permission,
                license=args.license,
                fps=args.fps,
                max_frames=args.max_frames,
                presentation_time=args.presentation_time,
                start_frame=args.start_frame,
            )
        return validate_recording(args.path)
    if args.command == "model":
        if args.operation == "audit-terrain":
            from .terrain_audit import audit_terrain_architecture

            return audit_terrain_architecture(
                args.dataset,
                args.config,
                args.output,
                split=args.split,
                image_size=args.image_size,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
            )
        from .training import export_yolo, train_yolo

        if args.operation == "train":
            return train_yolo(
                args.dataset,
                args.output,
                args.config,
                epochs=args.epochs,
                image_size=args.image_size,
                batch=args.batch,
                seed=args.seed,
            )
        return export_yolo(args.weights, args.output, image_size=args.image_size, scope=args.scope)
    if args.command == "chat-draft":
        from .lab import chat_draft

        return chat_draft(args.incoming, args.reply)
    if args.command in {"vision", "lab", "recording"}:
        from .onnx_detector import OnnxDetector
        from .vision import validate_terrain_source

        detector = None
        if getattr(args, "model", None) or getattr(args, "manifest", None):
            if not args.model or not args.manifest:
                raise ValueError("--model and --manifest must be supplied together")
            if args.command == "recording" and args.operation == "benchmark":
                detector = OnnxDetector(
                    args.model, args.manifest, confidence_threshold=args.confidence
                )
            else:
                detector = OnnxDetector(args.model, args.manifest)
        requested_terrain_source = getattr(args, "terrain_source", "fixture")
        if requested_terrain_source == "model" and detector is None:
            raise ValueError("--terrain-source model requires --model and --manifest")
        terrain_source = validate_terrain_source(requested_terrain_source, detector)
        if args.command == "lab" and args.operation in {"export-bundle", "verify-bundle"}:
            from .failure_bundle import export_failure_bundle, verify_failure_bundle

            if args.operation == "export-bundle":
                return export_failure_bundle(
                    args.path,
                    args.output,
                    include_video=args.video,
                    reinfer=args.reinfer,
                    detector=detector,
                    terrain_source=terrain_source,
                )
            return verify_failure_bundle(
                args.path, reinfer=args.reinfer, detector=detector, terrain_source=terrain_source
            )
        if args.command == "recording" and args.operation == "benchmark":
            from .recording_evaluation import benchmark_recordings

            return benchmark_recordings(
                args.cases,
                args.output,
                detector,
                confidence=args.confidence,
                development_datasets=args.development_dataset,
            )
        if args.command == "recording":
            from .recordings import observe_recording

            return observe_recording(args.path, args.output, detector=detector)
        if args.command == "vision":
            if args.operation == "ocr":
                from .ocr import TesseractOCR
                from .scene import ImageFrame

                reader = TesseractOCR(
                    binary=args.binary,
                    language=args.language,
                    tessdata_dir=args.tessdata_dir,
                    timeout_seconds=args.timeout,
                )
                return asdict(
                    reader.recognize(
                        ImageFrame.from_file(args.path),
                        crop=None if args.crop is None else tuple(args.crop),
                    )
                )
            if args.operation == "benchmark":
                from .evaluation import benchmark_dataset

                return benchmark_dataset(
                    args.path,
                    args.output,
                    detector=detector,
                    split=args.split,
                    terrain_metrics=args.terrain_metrics,
                )
            from .agent import HybridPerception
            from .map_pack import load_map_pack
            from .scene import ImageFrame
            from .vision import PixelPerception

            pack = load_map_pack(args.map_pack) if args.map_pack else None
            markers = None if pack is None else pack.markers
            perception = (
                PixelPerception(map_markers=markers)
                if detector is None
                else HybridPerception(detector, terrain_source=terrain_source, map_markers=markers)
            )
            return asdict(perception.observe(ImageFrame.from_file(args.path)))
        from .class_package import load_class_package
        from .lab import (
            CapabilityRunConfig,
            RunConfig,
            run_episode,
            run_soak,
            run_suite,
            verify_run,
        )
        from .map_pack import load_map_pack
        from .runtime_pipeline import RuntimeConfig as StructuredRuntimeConfig

        pack = load_map_pack(args.map_pack) if getattr(args, "map_pack", None) else None
        package_paths = getattr(args, "class_pack", None)
        class_packages = (
            None
            if package_paths is None
            else tuple(
                load_class_package(path)
                for path in ([package_paths] if isinstance(package_paths, Path) else package_paths)
            )
        )
        from .mission_program import load_mission_program

        mission_path = getattr(args, "mission_program", None)
        mission = None if mission_path is None else load_mission_program(mission_path)
        selected_runtime = getattr(args, "runtime", None)
        if selected_runtime is None:
            selected_runtime = (
                "structured"
                if class_packages is not None
                or mission is not None
                or (pack and any(len(m.portals) > 1 for m in pack.maps))
                else "legacy"
            )
        if class_packages is not None and selected_runtime != "structured":
            raise ValueError("--class-pack requires --runtime structured")
        if mission is not None and selected_runtime != "structured":
            raise ValueError("--mission-program requires --runtime structured")
        runtime = (
            StructuredRuntimeConfig(mission=mission) if selected_runtime == "structured" else None
        )
        from .scene_animation import load_scene_animation

        animation = (
            load_scene_animation(args.animation_scene)
            if getattr(args, "animation_scene", None)
            else None
        )

        if args.operation == "run":
            from .input_bindings import load_input_profile
            from .profiles import load_class_profile
            from .sprite_pack import load_sprite_pack

            if bool(args.asset_registry) != bool(args.sprite_map):
                raise ValueError("--asset-registry and --sprite-map must be supplied together")
            sprites = (
                load_sprite_pack(args.asset_registry, args.sprite_map)
                if args.asset_registry
                else None
            )
            class_package = None if class_packages is None else class_packages[0]
            job = args.job or ("assassin" if class_package is None else class_package.class_id)
            if class_package is not None and args.knowledge_db is not None:
                raise ValueError("--class-pack and legacy --knowledge-db cannot be combined")
            profile = (
                load_class_profile(args.knowledge_db, job, args.game_version)
                if args.knowledge_db
                else None
            )
            return run_episode(
                args.output,
                (RunConfig if class_package is None else CapabilityRunConfig)(
                    args.seed, job, args.max_steps, args.fault, args.record_stride
                ),
                detector,
                class_profile=profile,
                sprite_pack=sprites,
                terrain_source=terrain_source,
                map_pack=pack,
                runtime=runtime,
                animation=animation,
                class_package=class_package,
                input_profile=load_input_profile(args.input_bindings)
                if args.input_bindings
                else None,
            )
        if args.operation == "suite":
            return run_suite(
                args.output,
                args.seeds,
                detector=detector,
                terrain_source=terrain_source,
                map_pack=pack,
                runtime=runtime,
                animation=animation,
                class_packages=class_packages,
            )
        if args.operation == "soak":
            return run_soak(
                args.output,
                args.seconds,
                detector=detector,
                resource_interval_seconds=args.resource_interval,
                max_resource_samples=args.max_resource_samples,
                terrain_source=terrain_source,
                map_pack=pack,
                runtime=runtime,
                animation=animation,
                class_packages=class_packages,
            )
        if args.operation == "reinfer":
            from .lab import verify_perception

            return verify_perception(args.path, detector=detector, terrain_source=terrain_source)
        return verify_run(args.path)
    raise ValueError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "workbench":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be in [1, 65535]")
            try:
                import uvicorn

                from .workbench_api import create_app
            except ModuleNotFoundError as exc:
                raise ValueError("install the workbench extra: pip install '.[workbench]'") from exc
            uvicorn.run(
                create_app(args.workspace, args.examples),
                host="127.0.0.1",
                port=args.port,
                proxy_headers=False,
            )
            return 0
        if args.command == "demo":
            config = load_config(args.config) if args.config else RuntimeConfig()
            if args.frames is not None:
                config = replace(config, frames=args.frames)
            summary = _demo(config, args.output)
        elif args.command == "db":
            summary = _database(args.db_command, args.path)
        else:
            summary = _extended(args)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    if args.command == "lab" and (
        summary.get("status") in {"failed", "stopped"} or summary.get("passed") is False
    ):
        return 1
    return 0
