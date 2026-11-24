"""Runnable P0 demonstration and SQLite schema administration."""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            config = load_config(args.config) if args.config else RuntimeConfig()
            if args.frames is not None:
                config = replace(config, frames=args.frames)
            summary = _demo(config, args.output)
        else:
            summary = _database(args.db_command, args.path)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    return 0
