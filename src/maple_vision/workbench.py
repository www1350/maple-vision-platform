"""Persistent local simulator sessions and a single-worker validation queue."""

import base64
import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.util import find_spec
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from .class_package import ClassPackage, load_class_package
from .failure_bundle import export_failure_bundle, verify_failure_bundle
from .lab import CapabilityRunConfig, RunConfig, run_episode, verify_perception, verify_run
from .map_pack import MapPack, load_map_pack
from .mission_program import MissionProgram, load_mission_program
from .onnx_detector import OnnxDetector
from .runtime_pipeline import RuntimeConfig
from .scene import SceneRenderer
from .sim_types import CLASS_SPECS, SimAction
from .simulation import Simulator
from .vision import PixelPerception


class Conflict(ValueError):
    """The requested transition no longer applies."""


class Workbench:
    def __init__(self, root: Path, examples: Path | None = None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "sessions.sqlite"
        self.classes: dict[str, ClassPackage | None] = dict.fromkeys(CLASS_SPECS)
        self.maps: dict[str, MapPack | None] = {"builtin": None}
        self.models: dict[str, Path | None] = {"pixel": None}
        self.missions: dict[str, MissionProgram | None] = {"standard": None}
        if examples is not None:
            if not examples.is_dir():
                raise ValueError("examples directory does not exist")
            for file in sorted((examples / "classes").glob("*.json")):
                package = load_class_package(file)
                if package.class_id in self.classes:
                    raise ValueError("duplicate class ID")
                self.classes[package.class_id] = package
            for file in sorted((examples / "maps").glob("*.json")):
                if file.stem in self.maps:
                    raise ValueError("duplicate map selection")
                self.maps[file.stem] = load_map_pack(file)
            for file in sorted((examples / "missions").glob("*.json")):
                if file.stem in self.missions:
                    raise ValueError("duplicate mission selection")
                self.missions[file.stem] = load_mission_program(file)
            model = examples / "lifecycle-model"
            if (model / "model.onnx").is_file() and (model / "manifest.json").is_file():
                if find_spec("onnxruntime") is not None:
                    self.models["lifecycle-model"] = model.resolve()

        self.guard = sqlite3.connect(self.root / "owner.sqlite", timeout=0, check_same_thread=False)
        try:
            self.guard.execute("BEGIN EXCLUSIVE")
            with self.db() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                        created TEXT NOT NULL, config TEXT NOT NULL, outcome TEXT, error TEXT);
                    CREATE TABLE IF NOT EXISTS sims (
                        id TEXT PRIMARY KEY, version INTEGER NOT NULL, initial TEXT NOT NULL,
                        current TEXT NOT NULL, saved TEXT);
                """)
                db.execute(
                    "UPDATE jobs SET state='interrupted' WHERE state IN ('queued','running')"
                )
        except BaseException:
            self.guard.close()
            raise
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="maple-lab")
        self.lock = threading.Lock()

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.database, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.guard.close()

    def check_selection(self, job: str, map_id: str, model: str) -> None:
        if job not in self.classes or map_id not in self.maps or model not in self.models:
            raise ValueError("unknown class, map or perception selection")

    def compatible_missions(self, pack: MapPack | None) -> list[str]:
        names = []
        for name, mission in self.missions.items():
            if mission is not None:
                try:
                    mission.bind_map_pack(pack)
                except ValueError:
                    continue
            names.append(name)
        return names

    def entries(self) -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.execute(
                "SELECT id,kind,state,created,outcome,error FROM jobs ORDER BY rowid DESC LIMIT 200"
            )
            return [dict(row) for row in rows]

    def job(self, identity: str) -> dict[str, Any]:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return {**dict(row), "config": json.loads(row["config"])}

    def submit(self, kind: str, config: dict[str, Any]) -> str:
        config = dict(config)
        if kind == "run":
            self.check_selection(config["job"], config["map_id"], config["model"])
            mission_id = config.get("mission", "standard")
            if mission_id not in self.missions:
                raise ValueError("unknown mission selection")
            mission = self.missions[mission_id]
            if mission is not None:
                pack = self.maps[config["map_id"]]
                mission.bind_map_pack(pack)
            config["mission_program"] = None if mission is None else mission.to_dict()
            config_type = RunConfig if self.classes[config["job"]] is None else CapabilityRunConfig
            config_type(
                seed=config["seed"],
                job=config["job"],
                fault=config["fault"],
                max_steps=config["max_steps"],
            )
            if config["max_steps"] > 2500:
                raise ValueError("workbench accepts up to 2500 steps per episode")
        elif kind in {"replay", "reinfer", "export_bundle", "verify_bundle"}:
            source = self.job(config["source"])
            expected = "export_bundle" if kind == "verify_bundle" else "run"
            if source["kind"] != expected or source["state"] != "finished":
                raise Conflict("source evidence is unavailable for this operation")
        else:
            raise ValueError("unknown operation")
        identity = uuid4().hex
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')"
            ).fetchone()[0]
            if count >= 4:
                raise Conflict("the workbench queue is full")
            db.execute(
                "INSERT INTO jobs VALUES (?,?, 'queued',?,?,NULL,NULL)",
                (identity, kind, datetime.now(UTC).isoformat(), json.dumps(config)),
            )
            db.commit()
            self.executor.submit(self.execute, identity)
        return identity

    def cancel(self, identity: str) -> None:
        with self.db() as db:
            if not db.execute(
                "UPDATE jobs SET state='cancelled' WHERE id=? AND state='queued'", (identity,)
            ).rowcount:
                self.job(identity)
                raise Conflict("only queued jobs can be cancelled")

    def detector(self, name: str) -> OnnxDetector | None:
        path = self.models.get(name)
        if name not in self.models:
            raise ValueError("the recorded model is not registered")
        return None if path is None else OnnxDetector(path / "model.onnx", path / "manifest.json")

    def execute(self, identity: str) -> None:
        with self.db() as db:
            if not db.execute(
                "UPDATE jobs SET state='running' WHERE id=? AND state='queued'", (identity,)
            ).rowcount:
                return
        try:
            row = self.job(identity)
            config = row["config"]
            location = self.root / identity
            location.mkdir()
            kind = row["kind"]
            if kind == "run":
                package = self.classes[config["job"]]
                factory = RunConfig if package is None else CapabilityRunConfig
                result = run_episode(
                    location / "run",
                    factory(config["seed"], config["job"], config["max_steps"], config["fault"], 8),
                    self.detector(config["model"]),
                    map_pack=self.maps[config["map_id"]],
                    class_package=package,
                    runtime=RuntimeConfig(
                        mission=MissionProgram.from_dict(config["mission_program"])
                        if config.get("mission_program") is not None
                        else None
                    ),
                )
            else:
                source = self.job(config["source"])
                base = self.root / source["id"]
                if kind == "replay":
                    result = verify_run(base / "run")
                elif kind == "reinfer":
                    result = verify_perception(
                        base / "run", detector=self.detector(source["config"]["model"])
                    )
                elif kind == "export_bundle":
                    result = export_failure_bundle(base / "run", location / "bundle")
                else:
                    result = verify_failure_bundle(base / "bundle")
            (location / "result.json").write_text(
                json.dumps(result, allow_nan=False), encoding="utf-8"
            )
            outcome = str(result["status"]) if kind == "run" else "verified"
            with self.db() as db:
                db.execute(
                    "UPDATE jobs SET state='finished',outcome=? WHERE id=?", (outcome, identity)
                )
        except Exception as exc:
            with self.db() as db:
                db.execute(
                    "UPDATE jobs SET state='error',error=? WHERE id=?",
                    (f"{type(exc).__name__}: {exc}"[:1000], identity),
                )

    def evidence_dir(self, identity: str) -> Path | None:
        job = self.job(identity)
        if job["state"] != "finished":
            return None
        if job["kind"] == "run":
            return self.root / identity / "run"
        if job["kind"] == "export_bundle":
            return self.root / identity / "bundle"
        return self.evidence_dir(job["config"]["source"])

    def detail(self, identity: str) -> dict[str, Any]:
        row = self.job(identity)
        path = self.evidence_dir(identity)
        frames = (
            []
            if path is None
            else sorted(
                f.stem for f in (path / "frames").glob("*.png") if re.fullmatch(r"[0-9]{6}", f.stem)
            )
        )
        if path is not None and (path / "final.png").is_file():
            frames.append("final")
        result = None
        if row["state"] == "finished":
            result = json.loads((self.root / identity / "result.json").read_text(encoding="utf-8"))
        return {**row, "frames": frames, "result": result}

    def frame(self, identity: str, name: str) -> Path:
        path = self.evidence_dir(identity)
        if path is None or (name != "final" and not re.fullmatch(r"[0-9]{6}", name)):
            raise KeyError(name)
        png = path / "final.png" if name == "final" else path / "frames" / (name + ".png")
        if not png.is_file():
            raise KeyError(name)
        return png

    def trace(self, identity: str, offset: int) -> list[dict[str, Any]]:
        path = self.evidence_dir(identity)
        if path is None:
            return []
        trace = path / "trace.jsonl"
        if not trace.is_file():
            trace = path / "source-run" / "trace.jsonl"
        with trace.open(encoding="utf-8") as f:
            return [json.loads(line) for line in islice(f, offset, offset + 4)]

    def create_sim(self, job: str, map_id: str, seed: int) -> dict[str, Any]:
        self.check_selection(job, map_id, "pixel")
        pack = self.maps[map_id]
        sim = Simulator(
            seed,
            job,
            map_id="meadow" if pack is None else pack.entry_map,
            map_pack=pack,
            class_package=self.classes[job],
        )
        identity = uuid4().hex
        state = json.dumps(sim.save())
        with self.db() as db:
            db.execute("INSERT INTO sims VALUES (?,0,?,?,NULL)", (identity, state, state))
        return self.sim_view(sim, identity, 0)

    def session(
        self, identity: str, mode: str, version: int | None = None, action: SimAction | None = None
    ) -> dict[str, Any]:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sims WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise KeyError(identity)
            if mode != "read" and version != row["version"]:
                raise Conflict("session changed; refresh its current revision")
            current = row["current"]
            if mode == "reset":
                current = row["initial"]
            elif mode == "restore":
                if row["saved"] is None:
                    raise Conflict("no saved state")
                current = row["saved"]
            elif mode not in {"read", "step", "save"}:
                raise ValueError("unknown session operation")
            sim = Simulator.restore(json.loads(current))
            receipt = None
            if mode == "step":
                if action is None:
                    raise ValueError("an action is required")
                receipt = asdict(sim.step(action))
                current = json.dumps(sim.save())
            revision = int(row["version"])
            if mode != "read":
                revision += 1
                db.execute(
                    "UPDATE sims SET version=?,current=?,saved=? WHERE id=?",
                    (revision, current, current if mode == "save" else row["saved"], identity),
                )
        return {**self.sim_view(sim, identity, revision), "receipt": receipt}

    @staticmethod
    def sim_view(sim: Simulator, identity: str, revision: int) -> dict[str, Any]:
        snapshot = sim.snapshot()
        frame = SceneRenderer(map_pack=sim.map_pack).render(snapshot)
        markers = None if sim.map_pack is None else sim.map_pack.markers
        obs = PixelPerception(map_markers=markers).observe(frame)
        return {
            "id": identity,
            "revision": revision,
            "sequence": frame.sequence,
            "timestamp_ms": frame.timestamp_ms,
            "ground_truth": asdict(snapshot),
            "observation": asdict(obs),
            "image": base64.b64encode(frame.png_bytes()).decode(),
        }
