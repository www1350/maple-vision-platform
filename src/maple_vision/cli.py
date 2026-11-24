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
    maps = commands.add_parser("maps", help="validate a bounded offline synthetic map pack")
    map_commands = maps.add_subparsers(dest="operation", required=True)
    map_check = map_commands.add_parser("check")
    map_check.add_argument("path", type=Path)
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
    recording_import.add_argument("--fps", type=float)
    recording_import.add_argument("--max-frames", type=int, default=300)
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
    run.add_argument("--job", choices=("assassin", "dragon_knight", "cleric"), default="assassin")
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
    suite = lab_commands.add_parser("suite")
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--seeds", type=int, default=3)
    replay = lab_commands.add_parser("replay")
    replay.add_argument("path", type=Path)
    reinfer = lab_commands.add_parser(
        "reinfer", help="re-run every recorded perception and policy decision with matching inputs"
    )
    reinfer.add_argument("path", type=Path)
    soak = lab_commands.add_parser("soak")
    soak.add_argument("--output", type=Path, required=True)
    soak.add_argument("--seconds", type=int, default=60)
    soak.add_argument("--resource-interval", type=float, default=1.0)
    soak.add_argument("--max-resource-samples", type=int, default=1024)
    for model_command in (suite, soak, reinfer):
        model_command.add_argument("--model", type=Path)
        model_command.add_argument("--manifest", type=Path)
    for model_command in (run, suite, soak, reinfer):
        model_command.add_argument(
            "--terrain-source", choices=("fixture", "model"), default="fixture"
        )
    for map_command in (run, suite, soak):
        map_command.add_argument("--map-pack", type=Path)
    chat = commands.add_parser(
        "chat-draft", help="prepare a local human-reviewed reply; sends nothing"
    )
    chat.add_argument("--incoming", required=True)
    chat.add_argument("--reply", required=True)
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
    if args.command == "knowledge":
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
            )
        return validate_dataset(args.path)
    if args.command == "recording" and args.operation != "observe":
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
            detector = OnnxDetector(args.model, args.manifest)
        requested_terrain_source = getattr(args, "terrain_source", "fixture")
        if requested_terrain_source == "model" and detector is None:
            raise ValueError("--terrain-source model requires --model and --manifest")
        terrain_source = validate_terrain_source(requested_terrain_source, detector)
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
        from .lab import RunConfig, run_episode, run_soak, run_suite, verify_run
        from .map_pack import load_map_pack

        pack = load_map_pack(args.map_pack) if getattr(args, "map_pack", None) else None

        if args.operation == "run":
            from .profiles import load_class_profile
            from .sprite_pack import load_sprite_pack

            if bool(args.asset_registry) != bool(args.sprite_map):
                raise ValueError("--asset-registry and --sprite-map must be supplied together")
            sprites = (
                load_sprite_pack(args.asset_registry, args.sprite_map)
                if args.asset_registry
                else None
            )

            profile = (
                load_class_profile(args.knowledge_db, args.job, args.game_version)
                if args.knowledge_db
                else None
            )
            return run_episode(
                args.output,
                RunConfig(args.seed, args.job, args.max_steps, args.fault, args.record_stride),
                detector,
                class_profile=profile,
                sprite_pack=sprites,
                terrain_source=terrain_source,
                map_pack=pack,
            )
        if args.operation == "suite":
            return run_suite(
                args.output,
                args.seeds,
                detector=detector,
                terrain_source=terrain_source,
                map_pack=pack,
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
            )
        if args.operation == "reinfer":
            from .lab import verify_perception

            return verify_perception(args.path, detector=detector, terrain_source=terrain_source)
        return verify_run(args.path)
    raise ValueError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
