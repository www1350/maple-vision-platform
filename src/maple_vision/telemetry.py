"""Versioned JSONL traces exclude raw pixels and external credentials."""

import json
from dataclasses import asdict
from typing import TextIO

from .contracts import SCHEMA_VERSION, TickResult, nonempty


def write_tick(stream: TextIO, run_id: str, tick: TickResult) -> None:
    nonempty(run_id, "run_id")
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "tick": asdict(tick)}
    stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
