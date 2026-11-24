"""Session-split synthetic YOLO dataset creation and independent integrity checks."""

import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from random import Random
from tempfile import mkdtemp
from typing import Any

from PIL import Image

from .catalog import strict_json
from .scene import CLASS_NAMES, RenderedSample, SceneRenderer
from .sim_types import SimAction, SimSnapshot
from .simulation import GROUND, MAP_IDS, Simulator
from .sprite_pack import SpritePack, verify_dataset_sprite_pack

DATASET_VERSION = 1
LIFECYCLE_DATASET_VERSION = 2
LIFECYCLE_PHASES = ("spawn", "combat", "loot", "cleared", "dialogue", "transition")
LIFECYCLE_TAGS = (
    "left_edge",
    "right_edge",
    "high_platform",
    "not_grounded",
    "monsters_present",
    "loot_present",
    "cleared",
    "dialogue_visible",
    "map_transition",
    "attack_hit",
    "loot_collected",
)
POSITION_BINS = (
    ("left_edge", 0, 40),
    ("left", 40, 120),
    ("middle", 120, 200),
    ("right", 200, 280),
    ("right_edge", 280, 307),
)
LIFECYCLE_MAX_STEPS = 4096
LIFECYCLE_MAX_TOTAL_STEPS = 1_000_000


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
    profile: str = "legacy",
) -> dict[str, object]:
    """Generate portable PNG/YOLO labels without sharing sessions between splits.

    Geometry and annotations are synthetic fixtures, never scraped game assets.
    A fresh output directory is required. Construction is staged and validated
    before the final rename so failed generation does not publish a half dataset.
    """
    sessions = _integer(sessions, "sessions", 3, 1000)
    frames_per_session = _integer(frames_per_session, "frames_per_session", 1, 1000)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    if not isinstance(profile, str) or profile not in {"legacy", "lifecycle"}:
        raise ValueError("profile must be legacy or lifecycle")
    if profile == "lifecycle" and frames_per_session < 12:
        raise ValueError("lifecycle requires at least 12 frames per session")
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
        generator = _generate if profile == "legacy" else _generate_lifecycle
        manifest = generator(staging, sessions, frames_per_session, seed, sprite_pack)
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


def _label_text(sample: RenderedSample) -> str:
    rows = []
    for detection in sample.annotations:
        x1, y1, x2, y2 = detection.bbox
        width, height = sample.frame.width, sample.frame.height
        rows.append(
            f"{CLASS_NAMES.index(detection.class_id)} "
            f"{(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
            f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def _lifecycle_trajectory(
    seed: int, job: str, minimum_frames: int
) -> tuple[list[SimAction], list[SimSnapshot]]:
    """Training-only scenario oracle: generate real actions, never policy observations.

    No simulator state is edited. A full round trip provides actual entry events
    on both maps. Navigation and scenario selection may inspect training truth.
    This is not a learned-agent task-success measurement.
    """
    simulator = Simulator(seed=seed, job=job, map_id="meadow")
    actions: list[SimAction] = []
    snapshots = [simulator.snapshot()]

    def step(action: SimAction) -> SimSnapshot:
        if len(actions) >= LIFECYCLE_MAX_STEPS:
            raise ValueError("lifecycle trajectory exceeded its bounded step budget")
        outcome = simulator.step(action)
        if not outcome.snapshot.player.alive:
            raise ValueError("lifecycle scenario player died before coverage was produced")
        actions.append(action)
        snapshots.append(outcome.snapshot)
        return outcome.snapshot

    def move_to(x: float) -> None:
        while True:
            player = snapshots[-1].player
            distance = x - player.x
            if abs(distance) <= 2 and player.grounded:
                return
            step(
                SimAction("move", direction=1 if distance > 0 else -1)
                if abs(distance) > 2
                else SimAction()
            )

    for map_id in MAP_IDS:
        if snapshots[-1].map_id != map_id:
            raise ValueError("lifecycle trajectory did not reach the expected map")
        # The initial frame and actual first post-entry step are spawn examples.
        step(SimAction())
        move_to(0)
        while snapshots[-1].monsters:
            state = snapshots[-1]
            player = state.player
            spec = simulator.class_spec
            target = min(state.monsters, key=lambda monster: abs(monster.x - player.x))
            distance = target.x + target.w / 2 - (player.x + player.w / 2)
            if player.hp < spec.max_hp / 2:
                step(SimAction("potion_hp"))
            elif player.mp < max(spec.attack_mp_cost, 8):
                step(SimAction("potion_mp"))
            elif abs(distance) > spec.attack_range * 0.85:
                step(SimAction("move", direction=1 if distance > 0 else -1))
            else:
                step(SimAction("attack", direction=1 if distance >= 0 else -1))
        while snapshots[-1].drops:
            state = snapshots[-1]
            player = state.player
            target_drop = min(state.drops, key=lambda drop: abs(drop.x - player.x))
            distance = target_drop.x + target_drop.w / 2 - (player.x + player.w / 2)
            step(
                SimAction("loot")
                if abs(distance) <= 20
                else SimAction("move", direction=1 if distance > 0 else -1)
            )
        # These coverage scenarios occur only after genuine combat and collection.
        move_to(306)
        ladder = snapshots[-1].ladders[0]
        move_to(ladder.x + ladder.w / 2 - snapshots[-1].player.w / 2)
        while snapshots[-1].player.y + snapshots[-1].player.h > ladder.y:
            step(SimAction("climb", direction=-1))
        step(SimAction())
        move_to(snapshots[-1].npcs[0].x)
        dialogue = step(SimAction("interact"))
        if dialogue.dialogue is None:
            raise ValueError("lifecycle scenario failed to open the NPC dialogue")
        step(SimAction())
        portal = snapshots[-1].portals[0]
        move_to(portal.x + portal.w / 2 - snapshots[-1].player.w / 2)
        transition = step(SimAction("portal", target=portal.id))
        if "map_changed" not in transition.events:
            raise ValueError("lifecycle scenario failed to produce a map transition")
    while len(snapshots) < minimum_frames:
        # Extra requested frames remain truthful cleared-world observations.
        step(SimAction())
    return actions, snapshots


def _lifecycle_facts(snapshot: SimSnapshot, first_entries: dict[str, int]) -> dict[str, Any]:
    first_entries.setdefault(snapshot.map_id, snapshot.sequence)
    player = snapshot.player
    transition = "map_changed" in snapshot.events
    if transition:
        phase = "transition"
    elif snapshot.sequence in (0, first_entries[snapshot.map_id] + 1):
        phase = "spawn"
    elif snapshot.dialogue is not None:
        phase = "dialogue"
    elif snapshot.monsters:
        phase = "combat"
    elif snapshot.drops:
        phase = "loot"
    else:
        phase = "cleared"
    position_bin = next(name for name, low, high in POSITION_BINS if low <= player.x < high)
    conditions = {
        "left_edge": player.x < 40,
        "right_edge": player.x >= 280,
        "high_platform": player.grounded and player.y + player.h < GROUND,
        "not_grounded": not player.grounded,
        "monsters_present": bool(snapshot.monsters),
        "loot_present": bool(snapshot.drops),
        "cleared": not snapshot.monsters and not snapshot.drops,
        "dialogue_visible": snapshot.dialogue is not None,
        "map_transition": transition,
        "attack_hit": "attack_hit" in snapshot.events,
        "loot_collected": "loot" in snapshot.events,
    }
    return {
        "map_id": snapshot.map_id,
        "phase": phase,
        "coverage": {
            "player_x": player.x,
            "player_y": player.y,
            "position_bin": position_bin,
            "grounded": player.grounded,
            "monsters_alive": len(snapshot.monsters),
            "drops_remaining": len(snapshot.drops),
            "dialogue_visible": snapshot.dialogue is not None,
            "map_transition": transition,
            "tags": [tag for tag in LIFECYCLE_TAGS if conditions[tag]],
        },
    }


def _selected_lifecycle_frames(
    snapshots: list[SimSnapshot], count: int, seed: int
) -> tuple[list[int], list[dict[str, Any]]]:
    first_entries: dict[str, int] = {}
    facts = [_lifecycle_facts(snapshot, first_entries) for snapshot in snapshots]
    selected: set[int] = set()
    for map_id in MAP_IDS:
        for phase in LIFECYCLE_PHASES:
            candidates = [
                i
                for i, value in enumerate(facts)
                if (value["map_id"], value["phase"]) == (map_id, phase)
            ]
            if phase == "combat":
                candidates = [i for i in candidates if "attack_hit" in facts[i]["coverage"]["tags"]]
            if not candidates:
                raise ValueError(f"lifecycle scenario lacks required phase: {map_id}/{phase}")
            selected.add(candidates[0])
    # With at least 18 frames these genuine high/edge/climbing examples are reserved.
    for tag in ("high_platform", "right_edge", "not_grounded"):
        for map_id in MAP_IDS:
            if len(selected) >= count:
                break
            candidates = [
                i
                for i, value in enumerate(facts)
                if value["map_id"] == map_id and tag in value["coverage"]["tags"]
            ]
            if candidates:
                selected.add(candidates[0])
    rng = Random(seed)
    buckets = [
        [
            i
            for i, value in enumerate(facts)
            if i not in selected and (value["map_id"], value["phase"]) == (map_id, phase)
        ]
        for map_id in MAP_IDS
        for phase in LIFECYCLE_PHASES
    ]
    for bucket in buckets:
        rng.shuffle(bucket)
    while len(selected) < count:
        progress = False
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.add(bucket.pop())
                progress = True
        if not progress:
            raise ValueError(
                "lifecycle trajectory cannot supply requested distinct frame identities"
            )
    return sorted(selected), facts


def _lifecycle_splits(sessions: int, seed: int) -> dict[int, str]:
    rng = Random(seed)
    ids = list(range(sessions))
    rng.shuffle(ids)
    groups = [[index for index in ids if index % 3 == job] for job in range(3)]
    eligible = [group for group in groups if len(group) >= 3]
    train_count = max(1, int(sessions * 0.6))
    val_count = max(1, round(sessions * 0.2), len(eligible))
    test_count = max(1, sessions - train_count - val_count, len(eligible))
    remaining = {"train": sessions - val_count - test_count, "val": val_count, "test": test_count}
    assignment: dict[int, str] = {}
    for group in eligible:
        for index, split in zip(group[:3], remaining, strict=True):
            assignment[index] = split
            remaining[split] -= 1
    for index in ids:
        if index not in assignment:
            options = [split for split, size in remaining.items() for _ in range(size)]
            split = rng.choice(options)
            assignment[index] = split
            remaining[split] -= 1
    return assignment


def _coverage_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def empty() -> dict[str, Any]:
        return {
            "frames": 0,
            "phase_counts": dict.fromkeys(LIFECYCLE_PHASES, 0),
            "class_counts": dict.fromkeys(CLASS_NAMES, 0),
            "player_position_bins": {name: 0 for name, _, _ in POSITION_BINS},
            "tag_counts": dict.fromkeys(LIFECYCLE_TAGS, 0),
        }

    by_split = {
        split: {**empty(), "maps": {map_id: empty() for map_id in MAP_IDS}}
        for split in ("train", "val", "test")
    }
    for sample in samples:
        coverage = sample["coverage"]
        split = by_split[sample["split"]]
        for group in (split, split["maps"][sample["map_id"]]):
            group["frames"] += 1
            group["phase_counts"][sample["phase"]] += 1
            group["player_position_bins"][coverage["position_bin"]] += 1
            for class_id, count in coverage["visible_class_counts"].items():
                group["class_counts"][class_id] += count
            for tag in coverage["tags"]:
                group["tag_counts"][tag] += 1
    return {
        "schema_version": 1,
        "basis": "replayed_training_scenario_truth_and_visible_rendered_labels",
        "required_phases_per_map_per_session": list(LIFECYCLE_PHASES),
        "position_axis": "player sprite upper-left x in the 320-pixel fixture world",
        "position_bins": [
            {"name": name, "minimum_inclusive": low, "maximum_exclusive": high}
            for name, low, high in POSITION_BINS
        ],
        "tags_overlap": True,
        "by_split": by_split,
    }


def _generate_lifecycle(
    staging: Path,
    sessions: int,
    frames_per_session: int,
    seed: int,
    sprite_pack: SpritePack | None,
) -> dict[str, object]:
    assignment = _lifecycle_splits(sessions, seed)
    for split in ("train", "val", "test"):
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    session_records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    total_steps = 0
    for index in range(sessions):
        session_id = f"s{index:04d}"
        session_seed = (seed + 1_000_003 * (index + 1)) % 2**32
        style_seed = session_seed ^ 0xA571
        job = ("assassin", "dragon_knight", "cleric")[index % 3]
        split = assignment[index]
        actions, snapshots = _lifecycle_trajectory(session_seed, job, frames_per_session)
        total_steps += len(actions)
        if total_steps > LIFECYCLE_MAX_TOTAL_STEPS:
            raise ValueError("lifecycle dataset exceeds the total trajectory step limit")
        selected, facts = _selected_lifecycle_frames(snapshots, frames_per_session, session_seed)
        session_records.append(
            {
                "id": session_id,
                "split": split,
                "seed": session_seed,
                "style_seed": style_seed,
                "job": job,
                "map_id": "meadow",
                "trajectory": {"step_ms": 50, "actions": [asdict(action) for action in actions]},
            }
        )
        renderer = SceneRenderer(style_seed=style_seed, sprite_pack=sprite_pack)
        for frame_index, sequence in enumerate(selected):
            sample = renderer.render_sample(snapshots[sequence])
            stem = f"{session_id}_{frame_index:05d}"
            image_name, label_name = f"images/{split}/{stem}.png", f"labels/{split}/{stem}.txt"
            (staging / image_name).write_bytes(sample.frame.png_bytes())
            (staging / label_name).write_text(_label_text(sample), "utf-8")
            metadata = facts[sequence]
            metadata["coverage"]["visible_class_counts"] = {
                name: sum(item.class_id == name for item in sample.annotations)
                for name in CLASS_NAMES
            }
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
                    "image_sha256": _hash(staging / image_name),
                    "label_sha256": _hash(staging / label_name),
                    "objects": len(sample.annotations),
                    **metadata,
                }
            )
    _json(staging / "data.yaml", _data_config())
    manifest: dict[str, object] = {
        "schema_version": LIFECYCLE_DATASET_VERSION,
        "origin": "synthetic-fixture-v1",
        "profile": "lifecycle",
        "generator_version": "lifecycle-v1",
        "seed": seed,
        "frames_per_session": frames_per_session,
        "classes": list(CLASS_NAMES),
        "split_unit": "session",
        "sessions": session_records,
        "samples": samples,
        "coverage": _coverage_summary(samples),
        "data_yaml_sha256": _hash(staging / "data.yaml"),
        "notes": (
            "Training-only truth-guided action scenarios; not a visual-agent success evaluation. "
            "Synthetic fixtures require separate real-recording validation."
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


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_lifecycle(
    root: Path, manifest: dict[str, Any], sprite_pack: SpritePack | None
) -> dict[str, Any]:
    """Replay each bounded session once, independently of the scenario planner."""
    fields = {
        "schema_version",
        "origin",
        "profile",
        "generator_version",
        "seed",
        "frames_per_session",
        "classes",
        "split_unit",
        "sessions",
        "samples",
        "coverage",
        "data_yaml_sha256",
        "notes",
    }
    if sprite_pack is not None:
        fields.update(("sprite_pack", "visual_origin"))
    if set(manifest) != fields or (manifest.get("profile"), manifest.get("generator_version")) != (
        "lifecycle",
        "lifecycle-v1",
    ):
        raise ValueError("unsupported lifecycle manifest fields or generator version")
    seed = _integer(manifest["seed"], "lifecycle seed", 0, 2**32 - 1)
    frames_per_session = _integer(manifest["frames_per_session"], "lifecycle frames", 12, 1000)
    sessions = manifest["sessions"]
    _integer(len(sessions), "lifecycle sessions", 3, 1000)
    if len(sessions) * frames_per_session > 100_000:
        raise ValueError("lifecycle dataset exceeds 100000 frames")
    if not isinstance(manifest["notes"], str) or not manifest["notes"]:
        raise ValueError("lifecycle notes must be nonempty text")
    expected_splits = _lifecycle_splits(len(sessions), seed)
    sample_fields = {
        "session_id",
        "split",
        "sequence",
        "timestamp_ms",
        "width",
        "height",
        "image",
        "label",
        "image_sha256",
        "label_sha256",
        "objects",
        "map_id",
        "phase",
        "coverage",
    }
    grouped: dict[str, list[dict[str, Any]]] = {session["id"]: [] for session in sessions}
    for sample in manifest["samples"]:
        if set(sample) != sample_fields:
            raise ValueError("invalid lifecycle sample metadata fields")
        grouped[sample["session_id"]].append(sample)
    total_steps = 0
    verified_samples: list[dict[str, Any]] = []
    for index, session in enumerate(sessions):
        expected_seed = (seed + 1_000_003 * (index + 1)) % 2**32
        if set(session) != {"id", "split", "seed", "style_seed", "job", "map_id", "trajectory"}:
            raise ValueError("invalid lifecycle session fields")
        if (
            session["id"] != f"s{index:04d}"
            or session["seed"] != expected_seed
            or session["style_seed"] != expected_seed ^ 0xA571
            or session["job"] != ("assassin", "dragon_knight", "cleric")[index % 3]
            or session["map_id"] != "meadow"
            or session["split"] != expected_splits[index]
        ):
            raise ValueError("lifecycle session identity disagrees with seeded generation")
        trajectory = session["trajectory"]
        if (
            not isinstance(trajectory, dict)
            or set(trajectory) != {"step_ms", "actions"}
            or type(trajectory["step_ms"]) is not int
            or trajectory["step_ms"] != 50
            or not isinstance(trajectory["actions"], list)
            or not 1 <= len(trajectory["actions"]) <= LIFECYCLE_MAX_STEPS
        ):
            raise ValueError("invalid or over-budget lifecycle trajectory")
        total_steps += len(trajectory["actions"])
        if total_steps > LIFECYCLE_MAX_TOTAL_STEPS:
            raise ValueError("lifecycle dataset exceeds the total trajectory step limit")
        records = grouped[session["id"]]
        if len(records) != frames_per_session:
            raise ValueError("lifecycle session frame count disagrees with manifest")
        by_sequence = {sample["sequence"]: sample for sample in records}
        if max(by_sequence) > len(trajectory["actions"]):
            raise ValueError("lifecycle frame is outside its recorded trajectory")
        simulator = Simulator(session["seed"], session["job"], session["map_id"], step_ms=50)
        renderer = SceneRenderer(style_seed=session["style_seed"], sprite_pack=sprite_pack)
        first_entries: dict[str, int] = {}
        phases: dict[str, set[str]] = {map_id: set() for map_id in MAP_IDS}
        selected_hit_maps: set[str] = set()
        observed_dialogues: set[str] = set()
        snapshot = simulator.snapshot()
        for sequence in range(len(trajectory["actions"]) + 1):
            facts = _lifecycle_facts(snapshot, first_entries)
            if snapshot.dialogue is not None:
                observed_dialogues.add(snapshot.map_id)
            if sequence in by_sequence:
                record = by_sequence[sequence]
                rendered = renderer.render_sample(snapshot)
                facts["coverage"]["visible_class_counts"] = {
                    name: sum(item.class_id == name for item in rendered.annotations)
                    for name in CLASS_NAMES
                }
                if (
                    record["timestamp_ms"] != snapshot.timestamp_ms
                    or record["map_id"] != facts["map_id"]
                    or record["phase"] != facts["phase"]
                    or _canonical(record["coverage"]) != _canonical(facts["coverage"])
                ):
                    raise ValueError("lifecycle metadata disagrees with replayed scenario facts")
                with Image.open(_contained(root, record["image"])) as image:
                    if image.size != (rendered.frame.width, rendered.frame.height) or (
                        image.tobytes() != rendered.frame.pixels
                    ):
                        raise ValueError("lifecycle image disagrees with trajectory rendering")
                if _contained(root, record["label"]).read_text("utf-8") != _label_text(rendered):
                    raise ValueError("lifecycle labels disagree with visible rendered instances")
                phases[snapshot.map_id].add(record["phase"])
                if record["phase"] == "combat" and "attack_hit" in snapshot.events:
                    selected_hit_maps.add(snapshot.map_id)
                verified_samples.append(record)
            if sequence < len(trajectory["actions"]):
                raw = trajectory["actions"][sequence]
                if not isinstance(raw, dict) or set(raw) != {
                    "kind",
                    "direction",
                    "skill",
                    "target",
                }:
                    raise ValueError("invalid lifecycle action fields")
                snapshot = simulator.step(SimAction(**raw)).snapshot
                if not snapshot.player.alive:
                    raise ValueError("lifecycle trajectory contains a dead player")
        if any(found != set(LIFECYCLE_PHASES) for found in phases.values()):
            raise ValueError("lifecycle session is missing required map/phase coverage")
        if selected_hit_maps != set(MAP_IDS):
            raise ValueError("lifecycle session lacks a selected combat attack_hit on each map")
        state = simulator.save()
        if (
            set(observed_dialogues) != set(MAP_IDS)
            or snapshot.map_id != "meadow"
            or any(
                world["drops"] or any(monster["hp"] for monster in world["monsters"])
                for world in state["maps"].values()
            )
        ):
            raise ValueError("lifecycle source trajectory did not finish its two-map round trip")
    coverage = _coverage_summary(verified_samples)
    if _canonical(manifest["coverage"]) != _canonical(coverage):
        raise ValueError("lifecycle coverage summary disagrees with verified frames")
    return {
        "profile": "lifecycle",
        "generator_version": "lifecycle-v1",
        "coverage": coverage,
        "trajectory_steps_verified": total_steps,
        "trajectory_verification": "actions_pixels_labels_and_phase_facts",
        "visual_agent_success_evaluated": False,
    }


def validate_dataset(output: str | Path) -> dict[str, object]:
    """Verify hashes, YOLO bounds, grouping, sequence order and image dimensions."""
    root = Path(output)
    if (root / "manifest.json").stat().st_size > 128 * 1024 * 1024:
        raise ValueError("dataset manifest exceeds 128 MiB")
    manifest: Any = strict_json((root / "manifest.json").read_text("utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in (DATASET_VERSION, LIFECYCLE_DATASET_VERSION)
    ):
        raise ValueError("unsupported dataset manifest")
    version = manifest["schema_version"]
    if manifest.get("origin") != "synthetic-fixture-v1" or manifest.get("split_unit") != "session":
        raise ValueError("unsupported dataset provenance or split unit")
    if manifest.get("classes") != list(CLASS_NAMES):
        raise ValueError("dataset class order does not match this schema")
    if _hash(root / "data.yaml") != manifest.get("data_yaml_sha256"):
        raise ValueError("dataset training configuration checksum mismatch")
    if json.loads((root / "data.yaml").read_text("utf-8")) != _data_config():
        raise ValueError("dataset training configuration disagrees with manifest")
    sprite_metadata: dict[str, object] = {}
    sprite_pack = None
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
        if version == LIFECYCLE_DATASET_VERSION and (width, height) != (320, 192):
            raise ValueError("lifecycle image dimensions must match the 320x192 fixture")
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
        sample_map = (
            sample.get("map_id") if version == LIFECYCLE_DATASET_VERSION else map_for[session_id]
        )
        if not isinstance(sample_map, str) or sample_map not in MAP_IDS:
            raise ValueError("unknown sample map")
        map_counts[split][sample_map] += 1
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
    lifecycle_metadata = (
        _validate_lifecycle(root, manifest, sprite_pack)
        if version == LIFECYCLE_DATASET_VERSION
        else {}
    )
    return {
        **sprite_metadata,
        **lifecycle_metadata,
        "schema_version": version,
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
