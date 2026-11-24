"""Session-split synthetic YOLO dataset creation and independent integrity checks."""

import hashlib
import json
import math
import shutil
from pathlib import Path
from random import Random
from tempfile import mkdtemp
from typing import Any

from PIL import Image

from .scene import CLASS_NAMES, SceneRenderer
from .sim_types import SimAction
from .simulation import Simulator
from .sprite_pack import SpritePack, verify_dataset_sprite_pack

DATASET_VERSION = 1


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", "utf-8")


def generate_dataset(
    output: str | Path,
    sessions: int = 9,
    frames_per_session: int = 8,
    seed: int = 0,
    *,
    sprite_pack: SpritePack | None = None,
) -> dict[str, object]:
    """Generate portable PNG/YOLO labels without sharing sessions between splits.

    Geometry and annotations are synthetic fixtures, never scraped game assets.
    A fresh output directory is required. Construction is staged and validated
    before the final rename so failed generation does not publish a half dataset.
    """
    sessions = _integer(sessions, "sessions", 3, 1000)
    frames_per_session = _integer(frames_per_session, "frames_per_session", 1, 1000)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    if sprite_pack is not None and not isinstance(sprite_pack, SpritePack):
        raise ValueError("sprite_pack must be a verified SpritePack")
    if sessions * frames_per_session > 100_000:
        raise ValueError("dataset exceeds 100000 frames")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(f"dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        manifest = _generate(staging, sessions, frames_per_session, seed, sprite_pack)
        _json(staging / "manifest.json", manifest)
        summary = validate_dataset(staging)
        if destination.exists():
            raise FileExistsError(f"dataset already exists: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **summary,
        "output": str(destination),
        "manifest_sha256": _hash(destination / "manifest.json"),
    }


def _data_config() -> dict[str, object]:
    # JSON is a YAML subset; omitting `path` makes Ultralytics use the YAML parent.
    return {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(CLASS_NAMES),
    }


def _generate(
    staging: Path,
    sessions: int,
    frames_per_session: int,
    seed: int,
    sprite_pack: SpritePack | None,
) -> dict[str, object]:
    rng = Random(seed)
    ids = list(range(sessions))
    rng.shuffle(ids)
    train_count = max(1, int(sessions * 0.6))
    val_count = max(1, round(sessions * 0.2))
    groups = [[index for index in ids if (index // 3) % 2 == map_index] for map_index in (0, 1)]
    eligible_groups = [group for group in groups if len(group) >= 3]
    # Where enough sessions exist, reserve each fixture map in all three splits.
    val_count = max(val_count, len(eligible_groups))
    test_count = max(sessions - train_count - val_count, len(eligible_groups), 1)
    capacities = {"train": sessions - val_count - test_count, "val": val_count, "test": test_count}
    split_for: dict[int, str] = {}
    for group in eligible_groups:
        for index, split in zip(group[:3], capacities, strict=True):
            split_for[index] = split
            capacities[split] -= 1
    for index in ids:
        if index in split_for:
            continue
        choices = [split for split, remaining in capacities.items() for _ in range(remaining)]
        split = rng.choice(choices)
        split_for[index] = split
        capacities[split] -= 1
    for split in ("train", "val", "test"):
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    session_records: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for index in range(sessions):
        session_id = f"s{index:04d}"
        session_seed = (seed + 1_000_003 * (index + 1)) % 2**32
        style_seed = session_seed ^ 0xA571
        job = ("assassin", "dragon_knight", "cleric")[index % 3]
        map_id = ("meadow", "cavern")[(index // 3) % 2]
        split = split_for[index]
        session_records.append(
            {
                "id": session_id,
                "split": split,
                "seed": session_seed,
                "style_seed": style_seed,
                "job": job,
                "map_id": map_id,
            }
        )
        simulator = Simulator(seed=session_seed, job=job, map_id=map_id)
        renderer = SceneRenderer(style_seed=style_seed, sprite_pack=sprite_pack)
        action_rng = Random(session_seed)
        # A reproducible action-only setup creates visible loot while leaving two
        # living monsters. It neither mutates hidden state nor injects labels into pixels.
        for _ in range(21):
            simulator.step(SimAction("move", direction=1))
        for _ in range(16):
            simulator.step(SimAction("attack", direction=1))
        for _ in range(8):
            simulator.step(SimAction("move", direction=-1))
        for frame_index in range(frames_per_session):
            for _ in range(4 if frame_index else action_rng.randrange(1, 6)):
                choice = action_rng.randrange(8)
                direction = 1 if action_rng.randrange(4) else -1
                if choice < 3:
                    action = SimAction("move", direction=direction)
                elif choice == 3:
                    action = SimAction("jump", direction=direction)
                elif choice == 4:
                    action = SimAction("jump_attack", direction=direction)
                elif choice == 5:
                    action = SimAction("attack", direction=direction)
                elif choice == 6:
                    action = SimAction("loot")
                else:
                    action = SimAction("idle")
                simulator.step(action)
            sample = renderer.render_sample(simulator.snapshot())
            stem = f"{session_id}_{frame_index:05d}"
            image_name = f"images/{split}/{stem}.png"
            label_name = f"labels/{split}/{stem}.txt"
            image_path = staging / image_name
            label_path = staging / label_name
            image_path.write_bytes(sample.frame.png_bytes())
            labels: list[str] = []
            for detection in sample.annotations:
                x1, y1, x2, y2 = detection.bbox
                width, height = sample.frame.width, sample.frame.height
                labels.append(
                    f"{CLASS_NAMES.index(detection.class_id)} "
                    f"{(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
                    f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
                )
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), "utf-8")
            samples.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "sequence": sample.frame.sequence,
                    "timestamp_ms": sample.frame.timestamp_ms,
                    "width": sample.frame.width,
                    "height": sample.frame.height,
                    "image": image_name,
                    "label": label_name,
                    "image_sha256": _hash(image_path),
                    "label_sha256": _hash(label_path),
                    "objects": len(labels),
                }
            )
    _json(staging / "data.yaml", _data_config())
    manifest: dict[str, object] = {
        "schema_version": DATASET_VERSION,
        "origin": "synthetic-fixture-v1",
        "seed": seed,
        "classes": list(CLASS_NAMES),
        "split_unit": "session",
        "sessions": session_records,
        "samples": samples,
        "data_yaml_sha256": _hash(staging / "data.yaml"),
        "notes": (
            "Synthetic pixel fixture only; real recordings require a separate held-out evaluation."
        ),
    }
    if sprite_pack is not None:
        sprite_pack.write_sources(staging)
        manifest["visual_origin"] = "synthetic_geometry_with_authorized_assets"
        manifest["sprite_pack"] = sprite_pack.manifest()
    return manifest


def _contained(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("invalid dataset path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("dataset path must remain inside its directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("dataset symlink escapes its directory")
    if not resolved.is_file():
        raise ValueError(f"dataset file is missing: {relative}")
    return resolved


def validate_dataset(output: str | Path) -> dict[str, object]:
    """Verify hashes, YOLO bounds, grouping, sequence order and image dimensions."""
    root = Path(output)
    manifest: Any = json.loads((root / "manifest.json").read_text("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != DATASET_VERSION:
        raise ValueError("unsupported dataset manifest")
    if manifest.get("origin") != "synthetic-fixture-v1" or manifest.get("split_unit") != "session":
        raise ValueError("unsupported dataset provenance or split unit")
    if manifest.get("classes") != list(CLASS_NAMES):
        raise ValueError("dataset class order does not match this schema")
    if _hash(root / "data.yaml") != manifest.get("data_yaml_sha256"):
        raise ValueError("dataset training configuration checksum mismatch")
    if json.loads((root / "data.yaml").read_text("utf-8")) != _data_config():
        raise ValueError("dataset training configuration disagrees with manifest")
    sprite_metadata: dict[str, object] = {}
    if "sprite_pack" in manifest or "visual_origin" in manifest:
        if manifest.get("visual_origin") != "synthetic_geometry_with_authorized_assets":
            raise ValueError("dataset sprite visual origin is missing or unsupported")
        sprite_pack = verify_dataset_sprite_pack(root, manifest.get("sprite_pack"))
        sprite_metadata = {
            "visual_origin": manifest["visual_origin"],
            "sprite_pack": sprite_pack.manifest(),
        }
    elif (root / "sprite_objects").exists():
        raise ValueError("dataset has unmanifested sprite sources")
    sessions = manifest.get("sessions")
    samples = manifest.get("samples")
    if (
        not isinstance(sessions, list)
        or not sessions
        or not isinstance(samples, list)
        or not samples
    ):
        raise ValueError("dataset must have nonempty sessions and samples")
    split_for: dict[str, str] = {}
    map_for: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {split: {name: 0 for name in CLASS_NAMES} for split in split_counts}
    map_counts = {split: {"meadow": 0, "cavern": 0} for split in split_counts}
    for session in sessions:
        if not isinstance(session, dict) or not isinstance(session.get("id"), str):
            raise ValueError("invalid session")
        session_id, split = session["id"], session.get("split")
        if (
            not session_id
            or session_id in split_for
            or not isinstance(split, str)
            or split not in split_counts
        ):
            raise ValueError("invalid or duplicate session split")
        split_for[session_id] = split
        map_id = session.get("map_id")
        if not isinstance(map_id, str) or map_id not in {"meadow", "cavern"}:
            raise ValueError("unknown session map")
        if session.get("job") not in ("assassin", "dragon_knight", "cleric"):
            raise ValueError("unknown session class")
        _integer(session.get("seed"), "session seed", 0, 2**32 - 1)
        _integer(session.get("style_seed"), "style seed", 0, 2**32 - 1)
        map_for[session_id] = map_id
    if set(split_for.values()) != set(split_counts):
        raise ValueError("train/val/test must each contain a session")
    used_paths: set[str] = set()
    hash_splits: dict[str, str] = {}
    last_sequence: dict[str, int] = {}
    last_time: dict[str, int] = {}
    total_objects = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("invalid sample")
        session_id, split = sample.get("session_id"), sample.get("split")
        if (
            not isinstance(session_id, str)
            or not isinstance(split, str)
            or split_for.get(session_id) != split
        ):
            raise ValueError("session leakage or unknown session")
        width = _integer(sample.get("width"), "width", 1, 8192)
        height = _integer(sample.get("height"), "height", 1, 8192)
        sequence = _integer(sample.get("sequence"), "sequence", 0, 2**63 - 1)
        timestamp = _integer(sample.get("timestamp_ms"), "timestamp_ms", 0, 2**63 - 1)
        if sequence <= last_sequence.get(session_id, -1) or timestamp < last_time.get(
            session_id, -1
        ):
            raise ValueError("session frame order is not monotonic")
        last_sequence[session_id], last_time[session_id] = sequence, timestamp
        paths: dict[str, Path] = {}
        for kind, suffix in (("image", ".png"), ("label", ".txt")):
            relative = sample.get(kind)
            path = _contained(root, relative)
            if path.suffix.lower() != suffix or relative in used_paths:
                raise ValueError("duplicate or invalid dataset file")
            if not isinstance(relative, str) or not relative.startswith(f"{kind}s/{split}/"):
                raise ValueError("sample path does not match its split")
            used_paths.add(relative)
            if _hash(path) != sample.get(f"{kind}_sha256"):
                raise ValueError(f"{kind} checksum mismatch")
            paths[kind] = path
        image_hash = sample["image_sha256"]
        if image_hash in hash_splits and hash_splits[image_hash] != split:
            raise ValueError("identical image leaked across dataset splits")
        hash_splits[image_hash] = split
        with Image.open(paths["image"]) as image:
            image.load()
            if image.size != (width, height) or image.mode != "RGB":
                raise ValueError("image dimensions or mode disagree with manifest")
        rows = paths["label"].read_text("utf-8").splitlines()
        object_count = _integer(sample.get("objects"), "objects", 0, 1_000_000)
        if len(rows) != object_count:
            raise ValueError("label count disagrees with manifest")
        for row in rows:
            fields = row.split()
            if len(fields) != 5:
                raise ValueError("a YOLO annotation must have five fields")
            try:
                class_id = int(fields[0])
                cx, cy, w, h = map(float, fields[1:])
            except ValueError as exc:
                raise ValueError("invalid YOLO annotation number") from exc
            if str(class_id) != fields[0] or not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError("unknown YOLO class")
            if not all(math.isfinite(v) for v in (cx, cy, w, h)) or not (0 < w <= 1 and 0 < h <= 1):
                raise ValueError("invalid YOLO normalized dimensions")
            tolerance = 1e-7
            if not (
                -tolerance <= cx - w / 2 < cx + w / 2 <= 1 + tolerance
                and -tolerance <= cy - h / 2 < cy + h / 2 <= 1 + tolerance
            ):
                raise ValueError("YOLO annotation extends outside image")
            class_counts[split][CLASS_NAMES[class_id]] += 1
        total_objects += object_count
        split_counts[split] += 1
        map_counts[split][map_for[session_id]] += 1
    if set(last_sequence) != set(split_for) or any(count == 0 for count in split_counts.values()):
        raise ValueError("empty session or split")
    actual_paths = {
        path.relative_to(root).as_posix()
        for directory in (root / "images", root / "labels")
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_paths != used_paths:
        raise ValueError("dataset contains unlisted image or label files")
    return {
        **sprite_metadata,
        "schema_version": DATASET_VERSION,
        "origin": "synthetic-fixture-v1",
        "sessions": len(sessions),
        "frames": len(samples),
        "objects": total_objects,
        "split_frames": split_counts,
        "split_sessions": {split: list(split_for.values()).count(split) for split in split_counts},
        "class_counts": class_counts,
        "missing_classes": {
            split: [name for name, count in counts.items() if not count]
            for split, counts in class_counts.items()
        },
        "split_maps": map_counts,
        "missing_maps": {
            split: [map_id for map_id, count in counts.items() if not count]
            for split, counts in map_counts.items()
        },
        "validated": True,
    }
