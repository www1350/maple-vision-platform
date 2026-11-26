"""Self-contained, bounded projections of real offline run evidence.

Only applied actions reconstruct simulator truth. Recorded observations, beliefs
and task decisions remain recorded evidence until ``verify_perception`` actually
runs. Optional MP4 files are lossy presentations, never the pixel authority.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import math
import os
import re
import shutil
import stat
from collections.abc import Generator, Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, BinaryIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from .catalog import strict_json
from .contracts import Detection, Status
from .lab import (
    _Frames,
    _load_run_animation,
    canonical,
    digest,
    verify_perception,
    verify_run,
)
from .recordings import _VIDEO_LOCK
from .scene import ImageFrame, SceneRenderer
from .sim_types import SimAction
from .simulation import Simulator
from .sprite_pack import verify_dataset_sprite_pack
from .vision import Detector, TerrainSource, VisualObservation

MAX_FRAMES = 10_001
MAX_SOURCE_FILES = 12_000
MAX_BUNDLE_FILES = 25_000
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 512 * 1024
MAX_PNG_BYTES = 512 * 1024
MAX_VIDEO_BYTES = 128 * 1024 * 1024

_FRAME_LAYERS = (
    "ground_truth.jsonl",
    "observations.jsonl",
    "world_states.jsonl",
    "task_trace.jsonl",
    "supervisor.jsonl",
    "frames.jsonl",
)
_ACTION_LAYERS = ("action_trace.jsonl", "verification.jsonl")
_STATIC_LAYERS = ("versions.lock", "config.yaml", "report.md")
_VIDEO_FILES = {"replay.mp4", "video-info.json", "video_frames.jsonl"}
_HASH = re.compile(r"[0-9a-f]{64}")
_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
_SEMANTICS = {
    "ground_truth": "independent_simulator_replay_of_actual_applied_actions",
    "rgb": "exact_reconstructed_capture_pixels_including_recorded_faults",
    "observations_and_tasks": "recorded_evidence; not_reinferred_by_default",
    "video": "lossy_display_only; not_authoritative_rgb_or_semantic_reproduction",
    "scope": "offline_synthetic_fixture; not_official_client_or_real_domain_acceptance",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 240
        or _PATH.fullmatch(value) is None
        or any(part in {".", ".."} for part in value.split("/"))
        or len(value.split("/")) > 5
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("invalid failure bundle relative path")
    return value


def _hash_value(value: object) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError("invalid failure bundle SHA-256")
    return value


def _object(value: object, keys: set[str] | None, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or (keys is not None and set(value) != keys):
        raise ValueError(f"invalid {name} fields")
    return value


def _integer(value: object, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _read(path: Path, maximum: int = MAX_FILE_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        if path.is_symlink():
            raise ValueError("evidence must not be a symbolic link")
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError("evidence exceeds file byte budget or is not a regular file")
            data = stream.read(maximum + 1)
            if len(data) != info.st_size or len(data) > maximum:
                raise ValueError("evidence changed size or exceeds file byte budget")
            return data
    except OSError as exc:
        raise ValueError(f"cannot read local evidence: {path.name}") from exc


def _json(path: Path, maximum: int = MAX_JSON_BYTES) -> Any:
    try:
        return strict_json(_read(path, maximum).decode("utf-8"))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("evidence must be bounded UTF-8 JSON") from exc


def _tree(root: Path, *, source: bool) -> dict[str, int]:
    """Check every path and every size before hashing or decoding any payload."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be a real directory")
    count_limit = MAX_SOURCE_FILES if source else MAX_BUNDLE_FILES
    byte_limit = MAX_SOURCE_BYTES if source else MAX_BUNDLE_BYTES
    result: dict[str, int] = {}
    entries = total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > count_limit + MAX_FRAMES + 64:
                    raise ValueError("evidence exceeds directory entry budget")
                relative = _relative(Path(child.path).relative_to(root).as_posix())
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_size > MAX_FILE_BYTES:
                        raise ValueError("evidence exceeds file byte budget")
                    result[relative] = info.st_size
                    total += info.st_size
                    if len(result) > count_limit or total > byte_limit:
                        raise ValueError("evidence exceeds file count or total byte budget")
                else:
                    raise ValueError("symlinks and special files are not evidence")
    return result


def _copy_tree(source: Path, target: Path, *, source_run: bool) -> dict[str, int]:
    files = _tree(source, source=source_run)
    target.mkdir()
    total = 0
    limit = MAX_SOURCE_BYTES if source_run else MAX_BUNDLE_BYTES
    for relative, size in sorted(files.items()):
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("evidence path escapes its source")
        data = _read(path)
        if len(data) != size:
            raise ValueError("evidence changed during private capture")
        total += len(data)
        if total > limit:
            raise ValueError("captured evidence exceeds byte budget")
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        with copied.open("xb") as stream:
            stream.write(data)
    return files


@contextmanager
def _output(output: Path, source: Path) -> Iterator[Path]:
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("failure bundle output must be outside its source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield stage
        # Reserve the directory exclusively; the authoritative manifest is last.
        # Readers reject the incomplete publication, never accept a partial bundle.
        destination.mkdir()
        published: list[Path] = []
        try:
            for item in sorted(stage.iterdir(), key=lambda p: (p.name == "manifest.json", p.name)):
                target = destination / item.name
                if target.exists() or target.is_symlink():
                    raise FileExistsError("another writer changed the reserved output")
                item.rename(target)
                published.append(target)
        except BaseException:
            for item in reversed(published):
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _lines(path: Path) -> Generator[dict[str, Any]]:
    with path.open("rb") as stream:
        count = 0
        while line := stream.readline(MAX_LINE_BYTES + 1):
            count += 1
            if count > MAX_FRAMES or len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError("JSONL exceeds frame/line budget or lacks a complete line")
            try:
                yield _object(strict_json(line.decode("utf-8")), None, "JSONL row")
            except (UnicodeError, RecursionError) as exc:
                raise ValueError("JSONL row must be bounded UTF-8 JSON") from exc


@dataclass(frozen=True)
class _Source:
    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    terminal: dict[str, Any]
    manifest_sha256: str
    steps: int
    completion_overridden: bool


def _source(root: Path) -> _Source:
    inventory = _tree(root, source=True)
    manifest = _object(_json(root / "manifest.json", MAX_MANIFEST_BYTES), None, "run manifest")
    schema = _integer(manifest.get("schema_version"), 1, 2, "source run schema")
    if manifest.get("kind") != "offline-run-bundle":
        raise ValueError("source must be an offline run bundle")
    files = _object(manifest.get("files"), None, "run file map")
    if set(files) | {"manifest.json"} != set(inventory) or "manifest.json" in files:
        raise ValueError("source run contains unlisted or missing files")
    for name, expected in files.items():
        _relative(name)
        _hash_value(expected)
        if name == "final.png" or (name.startswith("frames/") and name.endswith(".png")):
            _png_pixels(root / name)
    summary = _object(_json(root / "summary.json"), None, "run summary")
    if type(summary.get("schema_version")) is not int or summary["schema_version"] != schema:
        raise ValueError("source run schema disagrees with summary")
    steps = _integer(summary.get("steps"), 0, MAX_FRAMES - 1, "run steps")
    count = sum(1 for _ in _lines(root / "trace.jsonl"))
    if count != steps:
        raise ValueError("trace count disagrees with source steps")
    evidence = _object(
        summary.get("perception_evidence"),
        {"schema_version", "settings", "terminal"},
        "terminal perception evidence (regenerate older runs before layered export)",
    )
    _integer(evidence["schema_version"], 1, 1, "perception evidence schema")
    terminal = _object(
        evidence["terminal"], {"cause", "guard", "decision", "observation"}, "terminal evidence"
    )
    cause = terminal["cause"]
    completion_overridden = False
    if cause == "step_budget_exhausted":
        if terminal["guard"] is not None or terminal["decision"] is not None:
            raise ValueError("budget termination must not fabricate a terminal policy evaluation")
        if summary.get("reason") != cause or summary.get("status") != "failed":
            raise ValueError("budget terminal evidence disagrees with run outcome")
        config = _object(summary.get("config"), None, "run configuration")
        if type(config.get("max_steps")) is not int or config["max_steps"] != steps:
            raise ValueError("budget termination does not occur at the actual budget")
    elif cause == "supervisor":
        guard = _object(terminal["guard"], {"state", "reason"}, "terminal supervisor")
        if (
            guard["state"] != "stopped"
            or terminal["decision"] is not None
            or summary.get("status") != "stopped"
            or summary.get("reason") != guard["reason"]
        ):
            raise ValueError("supervisor terminal evidence disagrees with run outcome")
    elif cause == "policy":
        decision = _object(
            terminal["decision"], {"action", "task", "state", "reason"}, "terminal decision"
        )
        guard = _object(terminal["guard"], {"state", "reason"}, "terminal supervisor")
        if guard["state"] != "running":
            raise ValueError("terminal policy was not permitted by the supervisor")
        if not isinstance(decision["state"], str) or decision["state"] not in {
            "completed",
            "failed",
            "stopped",
        }:
            raise ValueError("terminal policy decision is not terminal")
        if (summary.get("status"), summary.get("reason")) != (
            decision["state"],
            decision["reason"],
        ):
            branch_check = summary.get("conditional_branch_check")
            branch_mismatch = (
                isinstance(branch_check, dict) and branch_check.get("matched") is False
            )
            expected_reason = (
                "conditional_branch_check_failed"
                if branch_mismatch
                else "independent_goal_check_failed"
            )
            completion_overridden = (
                decision["state"] == "completed"
                and summary.get("status") == "failed"
                and summary.get("reason") == expected_reason
            )
            if not completion_overridden:
                raise ValueError("terminal policy evidence disagrees with run outcome")
        SimAction(**_object(decision["action"], None, "terminal action"))
    else:
        raise ValueError("unsupported run termination cause")
    return _Source(
        root,
        manifest,
        summary,
        terminal,
        _sha(_read(root / "manifest.json")),
        steps,
        completion_overridden,
    )


def _verify_source(source: _Source) -> dict[str, Any]:
    replay = verify_run(source.root)
    if source.completion_overridden and replay["independent_goal_check"] is not False:
        raise ValueError(
            "claimed independent goal override disagrees with the actual replay oracle"
        )
    return replay


def _observation(value: object, frame: ImageFrame) -> dict[str, Any]:
    fields = {
        "sequence",
        "timestamp_ms",
        "status",
        "confidence",
        "detections",
        "hp_ratio",
        "mp_ratio",
        "map_id",
        "dialogue_visible",
        "source_id",
    }
    data = _object(value, fields, "recorded observation")
    detections = data["detections"]
    if not isinstance(detections, list) or len(detections) > 512:
        raise ValueError("recorded observation exceeds detection budget")
    parsed = []
    for item in detections:
        item = _object(item, {"class_id", "bbox", "confidence"}, "detection")
        if not isinstance(item["bbox"], list) or len(item["bbox"]) != 4:
            raise ValueError("invalid recorded detection box")
        parsed.append(Detection(item["class_id"], tuple(item["bbox"]), item["confidence"]))
    options = {key: item for key, item in data.items() if key not in {"status", "detections"}}
    observation = VisualObservation(
        **options, status=Status(data["status"]), detections=tuple(parsed)
    )
    if (observation.sequence, observation.timestamp_ms) != (frame.sequence, frame.timestamp_ms):
        raise ValueError("recorded observation disagrees with actual capture identity")
    return data


class _Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.total = sum(_tree(root, source=False).values())

    def write(self, name: str, data: bytes) -> None:
        if len(data) > MAX_FILE_BYTES or self.total + len(data) > MAX_BUNDLE_BYTES:
            raise ValueError("derived evidence exceeds output byte budget")
        path = self.root / _relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        self.total += len(data)

    def line(self, stream: BinaryIO, row: dict[str, Any]) -> None:
        data = canonical(row) + b"\n"
        if (
            len(data) > MAX_LINE_BYTES
            or stream.tell() + len(data) > MAX_FILE_BYTES
            or self.total + len(data) > MAX_BUNDLE_BYTES
        ):
            raise ValueError("derived evidence exceeds line/file/output byte budget")
        stream.write(data)
        self.total += len(data)


def _metadata(source: _Source) -> dict[str, bytes]:
    summary = source.summary
    versions = {
        "schema_version": 1,
        "source_manifest_sha256": source.manifest_sha256,
        "source_run_schema": source.manifest["schema_version"],
        "recorded_runtime_identity": summary.get("runtime_identity"),
        "ruleset": summary.get("ruleset"),
        "model_manifest": summary.get("model_manifest"),
        "content": {
            key: summary.get(key)
            for key in ("class_profile", "map_pack", "sprite_pack", "animation", "class_package")
        },
    }
    config = {
        "schema_version": 1,
        "source_manifest_sha256": source.manifest_sha256,
        "run": summary["config"],
        "perception": summary["perception_evidence"]["settings"],
        "runtime": source.manifest.get("runtime"),
    }
    report = (
        "# Offline FailureBundle\n\n"
        f"Recorded outcome: **{summary['status']}**. Reason: `{summary['reason']}`.\n\n"
        f"This bundle contains {source.steps} actually applied actions and "
        f"{source.steps + 1} frames, including the real terminal observation. "
        f"Termination cause: `{source.terminal['cause']}`. Successful runs are also supported.\n\n"
        "## Verification boundary\n\n"
        "The default verifier checks source-run integrity, replays actual applied actions, "
        "reconstructs deterministic simulator states and exact capture RGB pixels, and "
        "reprojects every derived layer. Requested actions do not advance ground truth.\n\n"
        "Observations, temporal beliefs, tasks, visual-effect verdicts and supervisor decisions "
        "are faithfully copied recorded evidence. Their perception and policy have **not been "
        "re-executed by default**. Use `verify_failure_bundle(..., reinfer=True)` to actually "
        "rerun all N+1 frames with the matching local detector/settings. The returned report "
        "describes that invocation; no archived success flag substitutes for execution.\n\n"
        "The optional MP4 is lossy display material. Its container, sample timeline and hashes "
        "are checked separately; `video.semantic_reproduction` is false. Authoritative pixels "
        "are the reconstructed RGB hashes and lossless PNG frames. Matching a display hash "
        "is not proof of exact source pixels or a proof against a completely re-authored "
        "video plus manifest. No permissive pixel threshold is called exact agreement.\n\n"
        "## Scope and provenance\n\n"
        "Everything here concerns the independent offline synthetic simulator. It does not "
        "establish official-client behavior, real-domain model accuracy, account safety, "
        "or a 24-hour wall-clock soak. Asset provenance remains in source-run sidecars.\n\n"
        "`source-run/` is an exact private copy. Original source paths are unnecessary for "
        "action/RGB replay. Optional model reinference still needs the exact local model. "
        "Checksums bind content and detect inconsistency; they are not a digital signature "
        "of the original author. `config.yaml` is JSON, a YAML 1.2 subset.\n"
    )
    return {
        "versions.lock": canonical(versions) + b"\n",
        "config.yaml": canonical(config) + b"\n",
        "report.md": report.encode("utf-8"),
    }


def _project(source: _Source, output: Path) -> list[dict[str, Any]]:
    simulator = Simulator.restore(_object(_json(source.root / "initial-state.json"), None, "state"))
    if simulator.step_ms != 50 or simulator.snapshot().timestamp_ms != 0:
        raise ValueError(
            "FailureBundle requires a real uniform 50 ms source timeline starting at 0"
        )
    sprites = (
        None
        if source.summary.get("sprite_pack") is None
        else verify_dataset_sprite_pack(source.root, source.summary["sprite_pack"])
    )
    animation = _load_run_animation(source.root, source.summary, source.manifest["files"])
    renderer = SceneRenderer(sprite_pack=sprites, map_pack=simulator.map_pack, animation=animation)
    frames = _Frames(source.summary["config"]["fault"])
    writer = _Writer(output)
    captures: list[dict[str, Any]] = []
    with ExitStack() as stack:
        rows = stack.enter_context(closing(_lines(source.root / "trace.jsonl")))
        streams = {
            name: stack.enter_context((output / name).open("xb"))
            for name in _FRAME_LAYERS + _ACTION_LAYERS
        }
        for index in range(source.steps + 1):
            terminal = index == source.steps
            row = None if terminal else next(rows)
            snapshot = simulator.snapshot()
            if snapshot.sequence != index or snapshot.timestamp_ms != index * 50:
                raise ValueError("source replay does not have uniform real simulation timestamps")
            sample = renderer.render_sample(snapshot)
            frame = frames.apply(sample.frame)
            if (frame.width, frame.height) != (320, 192):
                raise ValueError("FailureBundle requires actual 320 x 192 source frames")
            reference = {
                "kind": "terminal" if terminal else "trace",
                "sha256": digest(source.terminal if terminal else row),
            }
            common = {
                "schema_version": 1,
                "frame_index": index,
                "sequence": snapshot.sequence,
                "timestamp_ms": snapshot.timestamp_ms,
                "terminal": terminal,
                "source": reference,
            }
            capture: dict[str, Any] = {
                **common,
                "capture_sequence": frame.sequence,
                "capture_timestamp_ms": frame.timestamp_ms,
                "width": frame.width,
                "height": frame.height,
                "rgb_sha256": _sha(frame.pixels),
                "png_file": f"frames/{index:06d}.png",
            }
            captures.append(capture)
            writer.write(capture["png_file"], frame.png_bytes())
            writer.line(streams["frames.jsonl"], capture)
            state = simulator.save()
            writer.line(
                streams["ground_truth.jsonl"],
                {
                    **common,
                    "simulator_state": state,
                    "simulator_state_sha256": digest(state),
                    "rendered_rgb_sha256": _sha(sample.frame.pixels),
                    "capture_rgb_sha256": capture["rgb_sha256"],
                    "annotations": [asdict(item) for item in sample.annotations],
                    "annotation_scope": "clean_simulator_render; not_faulted_capture_perception",
                },
            )
            if terminal:
                evidence = source.terminal
                runtime = source.summary.get("runtime_evidence", {}).get("terminal")
            else:
                assert row is not None
                required = {
                    "schema_version",
                    "sequence",
                    "frame_sha256",
                    "observation",
                    "guard",
                    "decision",
                    "execution",
                    "visual_effect",
                    "after_state_sha256",
                } | ({"runtime"} if source.manifest["schema_version"] == 2 else set())
                _object(row, required, "source trace")
                evidence = row
                runtime = row.get("runtime")
            observation = _observation(evidence["observation"], frame)
            writer.line(streams["observations.jsonl"], {**common, "observation": observation})
            writer.line(
                streams["world_states.jsonl"],
                {
                    **common,
                    "representation": "recorded_temporal_belief" if runtime else "unavailable",
                    "belief": None if runtime is None else runtime["belief"],
                },
            )
            policy_evaluated = (
                runtime["policy_evaluated"]
                if runtime is not None
                else evidence["decision"] is not None and evidence["guard"]["state"] == "running"
            )
            writer.line(
                streams["task_trace.jsonl"],
                {
                    **common,
                    "decision": evidence["decision"],
                    "policy_evaluated": policy_evaluated,
                    "runtime_decision": None if runtime is None else runtime["decision"],
                    "runtime_state": None if runtime is None else runtime["state"],
                    "interruption": None if runtime is None else runtime.get("interruption"),
                    "termination_cause": source.terminal["cause"] if terminal else None,
                },
            )
            writer.line(streams["supervisor.jsonl"], {**common, "guard": evidence["guard"]})
            if terminal:
                continue
            assert row is not None
            execution_fields = {"applied_action", "accepted", "reason", "events"}
            if source.summary.get("input_profile") is not None:
                execution_fields.add("input_dispatch")
            execution = _object(row["execution"], execution_fields, "execution")
            if type(execution["accepted"]) is not bool:
                raise ValueError("execution accepted must be a boolean")
            applied = SimAction(**execution["applied_action"])
            outcome = simulator.step(applied)
            actual = {
                "applied_action": asdict(applied),
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "events": list(outcome.events),
            }
            recorded_actual = {key: execution[key] for key in actual}
            if canonical(actual) != canonical(recorded_actual):
                raise ValueError("actual action replay disagrees with recorded execution")
            projected_execution = dict(actual)
            if "input_dispatch" in execution:
                projected_execution["input_dispatch"] = execution["input_dispatch"]
            if digest(simulator.save()) != row["after_state_sha256"]:
                raise ValueError("actual action replay disagrees with recorded state")
            writer.line(
                streams["action_trace.jsonl"],
                {
                    **common,
                    "requested_action": row["decision"]["action"],
                    "execution": projected_execution,
                    "dispatch": None if runtime is None else runtime["dispatch"],
                    "receipt": None if runtime is None else runtime["receipt"],
                    "after_state_sha256": row["after_state_sha256"],
                },
            )
            writer.line(
                streams["verification.jsonl"],
                {
                    **common,
                    "action_replay_verified": True,
                    "actual_execution": projected_execution,
                    "resulting_state_sha256": digest(simulator.save()),
                    "recorded_visual_effect": row["visual_effect"],
                    "perception_reproduced": False,
                    "policy_reproduced": False,
                },
            )
    for name, data in _metadata(source).items():
        writer.write(name, data)
    return captures


def _png_pixels(path: Path) -> bytes:
    data = _read(path, MAX_PNG_BYTES)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format != "PNG"
                or image.size != (320, 192)
                or image.mode != "RGB"
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("replay frame must be a single 320 x 192 RGB PNG")
            image.load()
            return image.tobytes()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("replay PNG cannot be decoded") from exc


def _boxes(data: memoryview) -> list[tuple[bytes, memoryview]]:
    result: list[tuple[bytes, memoryview]] = []
    position = 0
    while position < len(data):
        if len(data) - position < 8:
            raise ValueError("truncated MP4 box")
        size = int.from_bytes(data[position : position + 4], "big")
        kind = bytes(data[position + 4 : position + 8])
        header = 8
        if size == 1:
            if len(data) - position < 16:
                raise ValueError("truncated extended MP4 box")
            size = int.from_bytes(data[position + 8 : position + 16], "big")
            header = 16
        if size < header or size > len(data) - position or len(result) >= MAX_FRAMES + 64:
            raise ValueError("invalid or over-budget MP4 boxes")
        result.append((kind, data[position + header : position + size]))
        position += size
    return result


def _box(items: list[tuple[bytes, memoryview]], kind: bytes) -> memoryview:
    matches = [data for name, data in items if name == kind]
    if len(matches) != 1:
        raise ValueError(f"MP4 requires one {kind!r} box")
    return matches[0]


def _clock(payload: memoryview) -> tuple[int, int]:
    if len(payload) < 20:
        raise ValueError("truncated MP4 clock")
    if payload[0] == 0:
        offset, width = 12, 4
    elif payload[0] == 1 and len(payload) >= 32:
        offset, width = 20, 8
    else:
        raise ValueError("unsupported MP4 clock version")
    timescale = int.from_bytes(payload[offset : offset + 4], "big")
    duration = int.from_bytes(payload[offset + 4 : offset + 4 + width], "big")
    if not 1 <= timescale <= 1_000_000_000:
        raise ValueError("invalid MP4 timescale")
    return timescale, duration


def _video_packets(
    data: bytes,
    top: list[tuple[bytes, memoryview]],
    table: list[tuple[bytes, memoryview]],
    sizes: memoryview,
    count: int,
) -> None:
    """Bound every declared sample to actual local media bytes, even without cv2."""
    media_ranges = []
    position = 0
    for name, payload in top:
        header = 16 if int.from_bytes(data[position : position + 4], "big") == 1 else 8
        end = position + header + len(payload)
        if name == b"mdat":
            media_ranges.append((position + header, end))
        position = end
    constant = int.from_bytes(sizes[4:8], "big")
    lengths = (
        [constant] * count
        if constant
        else [int.from_bytes(sizes[index : index + 4], "big") for index in range(12, len(sizes), 4)]
    )
    if any(length < 1 or length > len(data) for length in lengths):
        raise ValueError("invalid MP4 encoded sample size")
    offset_kinds = [name for name, _ in table if name in {b"stco", b"co64"}]
    if len(offset_kinds) != 1:
        raise ValueError("MP4 requires exactly one local chunk-offset table")
    offsets = _box(table, offset_kinds[0])
    width = 4 if offset_kinds[0] == b"stco" else 8
    if len(offsets) < 8 or bytes(offsets[:4]) != b"\0\0\0\0":
        raise ValueError("invalid MP4 chunk offsets")
    chunks = int.from_bytes(offsets[4:8], "big")
    if not 1 <= chunks <= count or len(offsets) != 8 + chunks * width:
        raise ValueError("invalid MP4 chunk count")
    locations = [
        int.from_bytes(offsets[index : index + width], "big")
        for index in range(8, len(offsets), width)
    ]
    mapping = _box(table, b"stsc")
    if len(mapping) < 8 or bytes(mapping[:4]) != b"\0\0\0\0":
        raise ValueError("invalid MP4 chunk sample table")
    entries = int.from_bytes(mapping[4:8], "big")
    if not 1 <= entries <= chunks or len(mapping) != 8 + entries * 12:
        raise ValueError("invalid MP4 chunk sample count")
    runs: list[tuple[int, int]] = []
    for index in range(8, len(mapping), 12):
        first = int.from_bytes(mapping[index : index + 4], "big")
        repetitions = int.from_bytes(mapping[index + 4 : index + 8], "big")
        reference = int.from_bytes(mapping[index + 8 : index + 12], "big")
        if (
            not 1 <= first <= chunks
            or not 1 <= repetitions <= count
            or reference != 1
            or (not runs and first != 1)
            or (runs and first <= runs[-1][0])
        ):
            raise ValueError("invalid MP4 chunk-to-sample mapping")
        runs.append((first, repetitions))
    consumed = run = 0
    occupied = []
    for chunk, offset in enumerate(locations, 1):
        while run + 1 < len(runs) and runs[run + 1][0] <= chunk:
            run += 1
        following = consumed + runs[run][1]
        if following > count:
            raise ValueError("MP4 chunks claim more samples than actual frames")
        end = offset + sum(lengths[consumed:following])
        if not any(start <= offset < end <= stop for start, stop in media_ranges):
            raise ValueError("MP4 sample chunk lies outside local media bytes")
        occupied.append((offset, end))
        consumed = following
    ordered = sorted(occupied)
    if consumed != count or any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("MP4 sample chunks overlap or omit frames")


def _video_container(data: bytes, count: int, step_ms: int) -> dict[str, Any]:
    """Read bounded ISO-BMFF sample times, independent of decoder FPS hints."""
    top = _boxes(memoryview(data))
    ftyp = _box(top, b"ftyp")
    if len(ftyp) < 8 or (len(ftyp) - 8) % 4 or not any(name == b"mdat" for name, _ in top):
        raise ValueError("video is not a complete MP4 container")
    if any(name in {b"moof", b"sidx"} for name, _ in top):
        raise ValueError("fragmented video is outside the fixed local replay format")
    moov = _boxes(_box(top, b"moov"))
    movie_scale, movie_duration = _clock(_box(moov, b"mvhd"))
    track = _boxes(_box(moov, b"trak"))
    media = _boxes(_box(track, b"mdia"))
    handler = _box(media, b"hdlr")
    if len(handler) < 12 or bytes(handler[8:12]) != b"vide":
        raise ValueError("MP4 must have exactly one video track")
    timescale, duration = _clock(_box(media, b"mdhd"))
    if duration * 1000 != count * step_ms * timescale:
        raise ValueError("MP4 duration disagrees with actual simulation timing")
    if movie_duration * 1000 != count * step_ms * movie_scale:
        raise ValueError("MP4 movie timeline disagrees with actual simulation timing")
    for name, payload in track:
        if name == b"edts":
            edits = _box(_boxes(payload), b"elst")
            if (
                len(edits) != 20
                or bytes(edits[:8]) != b"\0\0\0\0\0\0\0\1"
                or int.from_bytes(edits[8:12], "big") != movie_duration
                or bytes(edits[12:20]) != b"\0\0\0\0\0\1\0\0"
            ):
                raise ValueError("MP4 edit list offsets or distorts the source timeline")
    media_info = _boxes(_box(media, b"minf"))
    references = _box(_boxes(_box(media_info, b"dinf")), b"dref")
    if len(references) < 8 or bytes(references[:8]) != b"\0\0\0\0\0\0\0\1":
        raise ValueError("MP4 requires one self-contained data reference")
    reference = _box(_boxes(references[8:]), b"url ")
    if bytes(reference) != b"\0\0\0\1":
        raise ValueError("MP4 must not reference external media")
    table = _boxes(_box(media_info, b"stbl"))
    if any(name == b"ctts" for name, _ in table):
        raise ValueError("composition-offset MP4 is outside the fixed local replay format")
    descriptions = _box(table, b"stsd")
    if len(descriptions) < 8 or bytes(descriptions[:8]) != b"\0\0\0\0\0\0\0\1":
        raise ValueError("MP4 requires one video sample description")
    description = _box(_boxes(descriptions[8:]), b"mp4v")
    if (
        len(description) < 78
        or int.from_bytes(description[6:8], "big") != 1
        or int.from_bytes(description[24:26], "big") != 320
        or int.from_bytes(description[26:28], "big") != 192
    ):
        raise ValueError("MP4 sample dimensions/codec disagree with replay frames")
    timing = _box(table, b"stts")
    if len(timing) < 8 or bytes(timing[:4]) != b"\0\0\0\0":
        raise ValueError("invalid MP4 timing table")
    entries = int.from_bytes(timing[4:8], "big")
    if not 1 <= entries <= MAX_FRAMES or len(timing) != 8 + entries * 8:
        raise ValueError("invalid MP4 timing entry count")
    samples = 0
    for position in range(8, len(timing), 8):
        repetitions = int.from_bytes(timing[position : position + 4], "big")
        delta = int.from_bytes(timing[position + 4 : position + 8], "big")
        if not 1 <= repetitions <= count or delta * 1000 != step_ms * timescale:
            raise ValueError("MP4 sample timing disagrees with the actual 50 ms timeline")
        samples += repetitions
    sizes = _box(table, b"stsz")
    if len(sizes) < 12 or bytes(sizes[:4]) != b"\0\0\0\0":
        raise ValueError("invalid MP4 sample-size table")
    constant_size = int.from_bytes(sizes[4:8], "big")
    sample_count = int.from_bytes(sizes[8:12], "big")
    if samples != count or sample_count != count:
        raise ValueError("MP4 frame count disagrees with N+1 source frames")
    if len(sizes) != (12 if constant_size else 12 + count * 4):
        raise ValueError("invalid MP4 sample-size entries")
    _video_packets(data, top, table, sizes, count)
    # Decode is also bounded by this independently checked single video track.
    return {
        "container": "mp4",
        "codec": "mp4v",
        "width": 320,
        "height": 192,
        "frames": count,
        "step_ms": step_ms,
        "first_pts_ms": 0,
        "last_pts_ms": (count - 1) * step_ms,
        "duration_ms": count * step_ms,
        "media_timescale": timescale,
        "media_duration": duration,
        "composition_offsets": False,
        "external_media": False,
    }


def _cv2(*, required: bool) -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        if required:
            raise ValueError("local MP4 creation requires the video extra (opencv-python)") from exc
        return None


def _codec_identity(cv2: Any) -> dict[str, str]:
    return {
        "opencv": str(cv2.__version__),
        "build_sha256": _sha(cv2.getBuildInformation().encode()),
    }


def _decode_video(path: Path, captures: list[dict[str, Any]], cv2: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with (
        _VIDEO_LOCK,
        patch.dict(os.environ, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "protocol_whitelist;file,pipe"}),
    ):
        reader = cv2.VideoCapture(str(path.resolve()), cv2.CAP_FFMPEG)
        try:
            if not reader.isOpened():
                raise ValueError("local MP4 decoder cannot open the generated replay")
            for index in range(len(captures) + 1):
                ok, bgr = reader.read()
                if not ok:
                    break
                if index >= len(captures):
                    raise ValueError("decoded MP4 contains extra frames")
                if bgr is None or bgr.shape != (192, 320, 3):
                    raise ValueError("decoded MP4 dimensions disagree with source frames")
                pts = float(reader.get(cv2.CAP_PROP_POS_MSEC))
                if not math.isfinite(pts) or abs(pts - index * 50) > 0.001:
                    raise ValueError("decoded MP4 PTS disagrees with source playback timeline")
                result.append(
                    {
                        "schema_version": 1,
                        "frame_index": index,
                        "pts_ms": index * 50,
                        "source_rgb_sha256": captures[index]["rgb_sha256"],
                        "decoded_rgb_sha256": _sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()),
                    }
                )
        finally:
            reader.release()
    if len(result) != len(captures):
        raise ValueError("MP4 decoder did not produce all N+1 frames")
    return result


def _make_video(root: Path, captures: list[dict[str, Any]]) -> dict[str, Any]:
    cv2 = _cv2(required=True)
    path = root / "replay.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (320, 192)
    )
    try:
        if not writer.isOpened():
            raise ValueError("local MP4 encoder is unavailable; no placeholder video is produced")
        for capture in captures:
            rgb = np.frombuffer(_png_pixels(root / capture["png_file"]), dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb.reshape(192, 320, 3), cv2.COLOR_RGB2BGR))
            if path.exists() and path.stat().st_size > MAX_VIDEO_BYTES:
                raise ValueError("MP4 exceeds video byte budget")
    finally:
        writer.release()
    data = _read(path, MAX_VIDEO_BYTES)
    container = _video_container(data, len(captures), 50)
    decoded = _decode_video(path, captures, cv2)
    info = {
        "schema_version": 1,
        "role": "lossy_display_only",
        "semantic_reproduction": False,
        "file": "replay.mp4",
        "sha256": _sha(data),
        "codec_identity": _codec_identity(cv2),
        "container": container,
        "source_rgb_timeline_sha256": digest(captures),
        "decoded_timeline_sha256": digest(decoded),
    }
    output = _Writer(root)
    output.write("video-info.json", canonical(info) + b"\n")
    with (root / "video_frames.jsonl").open("xb") as stream:
        for row in decoded:
            output.line(stream, row)
    return {
        "present": True,
        "container_verified": True,
        "decode_check": "matched",
        "semantic_reproduction": False,
    }


def _check_video(root: Path, captures: list[dict[str, Any]]) -> dict[str, Any]:
    info = _object(
        _json(root / "video-info.json"),
        {
            "schema_version",
            "role",
            "semantic_reproduction",
            "file",
            "sha256",
            "codec_identity",
            "container",
            "source_rgb_timeline_sha256",
            "decoded_timeline_sha256",
        },
        "video information",
    )
    _integer(info["schema_version"], 1, 1, "video schema")
    if (
        info["role"] != "lossy_display_only"
        or info["semantic_reproduction"] is not False
        or info["file"] != "replay.mp4"
        or info["source_rgb_timeline_sha256"] != digest(captures)
    ):
        raise ValueError("video information is not bound to the source RGB timeline")
    identity = _object(info["codec_identity"], {"opencv", "build_sha256"}, "video codec identity")
    if not isinstance(identity["opencv"], str) or not 1 <= len(identity["opencv"]) <= 100:
        raise ValueError("invalid recorded OpenCV version")
    _hash_value(identity["build_sha256"])
    data = _read(root / "replay.mp4", MAX_VIDEO_BYTES)
    if _sha(data) != _hash_value(info["sha256"]):
        raise ValueError("video checksum mismatch")
    container = _video_container(data, len(captures), 50)
    if canonical(container) != canonical(info["container"]):
        raise ValueError("video container metadata disagrees with actual sample tables")
    recorded = list(_lines(root / "video_frames.jsonl"))
    if len(recorded) != len(captures) or digest(recorded) != info["decoded_timeline_sha256"]:
        raise ValueError("video decoded-frame timeline count/hash mismatch")
    for index, row in enumerate(recorded):
        _object(
            row,
            {"schema_version", "frame_index", "pts_ms", "source_rgb_sha256", "decoded_rgb_sha256"},
            "video frame",
        )
        expected = {
            "schema_version": 1,
            "frame_index": index,
            "pts_ms": index * 50,
            "source_rgb_sha256": captures[index]["rgb_sha256"],
            "decoded_rgb_sha256": _hash_value(row["decoded_rgb_sha256"]),
        }
        if canonical(row) != canonical(expected):
            raise ValueError("video frame disagrees with source RGB timeline")
    cv2 = _cv2(required=False)
    if cv2 is None:
        check = "not_run_decoder_unavailable"
    elif _codec_identity(cv2) != identity:
        check = "not_run_decoder_mismatch"
    else:
        if canonical(_decode_video(root / "replay.mp4", captures, cv2)) != canonical(recorded):
            raise ValueError("video decoded pixels disagree with recorded display hashes")
        check = "matched"
    return {
        "present": True,
        "container_verified": True,
        "decode_check": check,
        "semantic_reproduction": False,
    }


def _manifest(source: _Source, hashes: dict[str, str], include_video: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "offline-failure-bundle",
        "source": {
            "directory": "source-run",
            "schema_version": source.manifest["schema_version"],
            "manifest_sha256": source.manifest_sha256,
        },
        "counts": {"steps": source.steps, "frames": source.steps + 1},
        "timeline": {
            "step_ms": 50,
            "first_timestamp_ms": 0,
            "last_timestamp_ms": source.steps * 50,
        },
        "layers": {
            **{name: source.steps + 1 for name in _FRAME_LAYERS},
            **{name: source.steps for name in _ACTION_LAYERS},
        },
        "semantics": _SEMANTICS,
        "include_video": include_video,
        "files": dict(sorted(hashes.items())),
    }


def _options(reinfer: bool, detector: Detector | None, terrain_source: TerrainSource) -> None:
    if type(reinfer) is not bool:
        raise ValueError("reinfer must be boolean")
    if not reinfer and (detector is not None or terrain_source != "fixture"):
        raise ValueError("a supplied detector/terrain mode requires reinfer=True")
    if not isinstance(terrain_source, str) or terrain_source not in {"fixture", "model"}:
        raise ValueError("unsupported terrain source")


def _report(
    source: _Source,
    replay: dict[str, Any],
    files: int,
    video: dict[str, Any],
    reinference: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "verified": True,
        "scope": "offline_failure_bundle",
        "source_schema": source.manifest["schema_version"],
        "source_manifest_sha256": source.manifest_sha256,
        "recorded_status": source.summary["status"],
        "recorded_reason": source.summary["reason"],
        "steps_replayed": source.steps,
        "frames_reconstructed": source.steps + 1,
        "action_replay_verified": True,
        "deterministic_state_reproduced": True,
        "rgb_reproduced": True,
        "layer_projection_verified": True,
        "perception_reproduced": reinference is not None,
        "policy_reproduced": reinference is not None,
        "action_replay": replay,
        "reinference": reinference,
        "video": video,
        "files_checked": files,
    }


def export_failure_bundle(
    run: Path,
    output: Path,
    *,
    include_video: bool = False,
    reinfer: bool = False,
    detector: Detector | None = None,
    terrain_source: TerrainSource = "fixture",
) -> dict[str, Any]:
    """Export a real successful or unsuccessful run, without overwriting any output.

    ``reinfer=True`` actually invokes the original perception/policy verifier.
    That optional result belongs to this call; the archive makes no reusable
    assertion that a future default verification has performed reinference.
    """
    _options(reinfer, detector, terrain_source)
    if type(include_video) is not bool:
        raise ValueError("include_video must be boolean")
    if include_video:
        _cv2(required=True)
    with _output(Path(output), Path(run)) as stage:
        _copy_tree(Path(run), stage / "source-run", source_run=True)
        source = _source(stage / "source-run")
        replay = _verify_source(source)
        reinference = (
            verify_perception(source.root, detector, terrain_source=terrain_source)
            if reinfer
            else None
        )
        captures = _project(source, stage)
        video = (
            _make_video(stage, captures)
            if include_video
            else {
                "present": False,
                "semantic_reproduction": False,
                "decode_check": "not_applicable",
            }
        )
        inventory = _tree(stage, source=False)
        hashes = {name: _sha(_read(stage / name)) for name in inventory}
        manifest = canonical(_manifest(source, hashes, include_video)) + b"\n"
        if (
            len(manifest) > MAX_MANIFEST_BYTES
            or sum(inventory.values()) + len(manifest) > MAX_BUNDLE_BYTES
        ):
            raise ValueError("failure bundle manifest/output exceeds byte budget")
        (stage / "manifest.json").write_bytes(manifest)
        report = _report(source, replay, len(hashes), video, reinference)
    report["output"] = str(Path(output))
    return report


def verify_failure_bundle(
    path: Path,
    *,
    reinfer: bool = False,
    detector: Detector | None = None,
    terrain_source: TerrainSource = "fixture",
) -> dict[str, Any]:
    """Read-only verification from bounded private bytes, not original source paths."""
    _options(reinfer, detector, terrain_source)
    with TemporaryDirectory(prefix="maple-failure-verify-") as temporary:
        base = Path(temporary)
        root = base / "bundle"
        inventory = _copy_tree(Path(path), root, source_run=False)
        manifest = _object(
            _json(root / "manifest.json", MAX_MANIFEST_BYTES),
            {
                "schema_version",
                "kind",
                "source",
                "counts",
                "timeline",
                "layers",
                "semantics",
                "include_video",
                "files",
            },
            "failure bundle manifest",
        )
        _integer(manifest["schema_version"], 1, 1, "failure bundle schema")
        if (
            manifest["kind"] != "offline-failure-bundle"
            or type(manifest["include_video"]) is not bool
        ):
            raise ValueError("unsupported failure bundle kind/video marker")
        hashes = _object(manifest["files"], None, "failure file map")
        if set(hashes) | {"manifest.json"} != set(inventory) or "manifest.json" in hashes:
            raise ValueError("failure bundle contains unlisted or missing evidence")
        for name, expected in hashes.items():
            if _sha(_read(root / _relative(name))) != _hash_value(expected):
                raise ValueError(f"failure bundle checksum mismatch: {name}")
        source = _source(root / "source-run")
        if canonical(manifest) != canonical(_manifest(source, hashes, manifest["include_video"])):
            raise ValueError(
                "failure manifest disagrees with source schema/timeline/layer semantics"
            )
        replay = _verify_source(source)
        projection = base / "projection"
        projection.mkdir()
        captures = _project(source, projection)
        projected = _tree(projection, source=False)
        expected_files = {
            "source-run/manifest.json",
            *("source-run/" + name for name in source.manifest["files"]),
            *projected,
            *(_VIDEO_FILES if manifest["include_video"] else set()),
        }
        if set(hashes) != expected_files:
            raise ValueError("failure bundle is missing required layers or has unexpected files")
        for name in projected:
            if name.startswith("frames/"):
                if _png_pixels(root / name) != _png_pixels(projection / name):
                    raise ValueError(f"RGB projection mismatch: {name}")
            elif _read(root / name) != _read(projection / name):
                raise ValueError(f"layer projection mismatch: {name}")
        video = (
            _check_video(root, captures)
            if manifest["include_video"]
            else {
                "present": False,
                "semantic_reproduction": False,
                "decode_check": "not_applicable",
            }
        )
        reinference = (
            verify_perception(source.root, detector, terrain_source=terrain_source)
            if reinfer
            else None
        )
        return _report(source, replay, len(hashes), video, reinference)
