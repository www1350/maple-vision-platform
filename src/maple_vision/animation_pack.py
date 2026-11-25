"""Bounded animation assets, integer timelines and self-contained source packs.

This is an independent content API. It does not alter the existing renderer or
static SpritePack contract. Alpha bounds describe a pasted frame after canvas
clipping, not its visibility after other scene objects occlude it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any

from PIL import Image

from .asset_registry import AssetEntry
from .catalog import canonical_json, strict_json, validate_provenance
from .sim_types import ACTION_KINDS
from .sprite_pack import PACK_SCOPES, SPRITE_CLASSES

ANIMATION_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 512 * 1024
MAX_CLIPS = 32
MAX_FRAMES_PER_CLIP = 128
MAX_FRAME_REFERENCES = 512
MAX_SOURCES = 256
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PACK_SOURCE_BYTES = 32 * 1024 * 1024
MAX_FRAME_DIMENSION = 1024
MAX_FRAME_PIXELS = 1024 * 1024
MAX_PACK_PIXELS = 8 * 1024 * 1024
MAX_CLIP_DURATION_MS = 60_000
MAX_CANVAS_PIXELS = 4 * 1024 * 1024
MAX_PLACEMENTS = 128
MAX_COMPOSITION_WORK_PIXELS = 16 * 1024 * 1024
MAX_TIME_MS = 2**53 - 1
ACTION_STATES = ACTION_KINDS | {"hurt", "death", "spawn", "despawn"}
Box = tuple[int, int, int, int]
_FORMATS = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
}
_CLIP_KEYS = {
    "clip_id",
    "entity_id",
    "game_version",
    "class_id",
    "action_state",
    "playback",
    "frames",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value: object, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"invalid animation {label}")
    return value


def _token(value: object, label: str) -> str:
    text = _text(value, label, 120)
    if not text[0].isalnum() or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for char in text
    ):
        raise ValueError(f"animation {label} must be an ASCII identifier")
    return text


def _hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"animation {label} must be a lowercase SHA256")
    return value


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid animation {label} fields")
    return value


def _version(value: object) -> None:
    if type(value) is not int or value != ANIMATION_SCHEMA_VERSION:
        raise ValueError("unsupported animation schema version")


def _read(path: Path, budget: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("animation source must be a regular nonsymlink file")
    with path.open("rb") as stream:
        data = stream.read(budget + 1)
    if not data or len(data) > budget:
        raise ValueError("animation source is empty or exceeds its byte budget")
    return data


def _json(path: Path, budget: int = MAX_MANIFEST_BYTES) -> object:
    try:
        return strict_json(_read(path, budget).decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("animation metadata must be UTF-8") from exc


def _directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("animation source directory must be a nonsymlink directory")


def _entry(value: object) -> AssetEntry:
    metadata = _object(value, set(AssetEntry.__dataclass_fields__), "asset metadata")
    entry = AssetEntry(**metadata)
    _hash(entry.asset_id, "asset ID")
    _hash(entry.sha256, "source checksum")
    _text(entry.entity_id, "entity ID")
    _text(entry.game_version, "game version")
    validate_provenance(entry.source_uri, entry.permission, entry.license)
    width = _integer(entry.width, "frame width", 1, MAX_FRAME_DIMENSION)
    height = _integer(entry.height, "frame height", 1, MAX_FRAME_DIMENSION)
    if width * height > MAX_FRAME_PIXELS:
        raise ValueError("animation frame exceeds pixel budget")
    extensions = {media_type: extension for extension, media_type in _FORMATS.values()}
    if (
        not isinstance(entry.media_type, str)
        or entry.media_type not in extensions
        or entry.relative_path != f"objects/{entry.sha256}.{extensions[entry.media_type]}"
    ):
        raise ValueError("animation asset path must be content-addressed")
    payload = {key: item for key, item in metadata.items() if key != "asset_id"}
    if _sha(canonical_json(payload).encode("utf-8")) != entry.asset_id:
        raise ValueError("animation asset metadata SHA256 mismatch")
    return entry


def _decode(encoded: bytes, entry: AssetEntry) -> tuple[bytes, Box]:
    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError("animation source exceeds byte budget or is not immutable bytes")
    if _sha(encoded) != entry.sha256:
        raise ValueError("animation source content SHA256 mismatch")
    try:
        with Image.open(BytesIO(encoded)) as image:
            if image.format not in _FORMATS or getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    "animation sources must be separately imported single-frame images"
                )
            extension, media_type = _FORMATS[image.format]
            if (
                image.size != (entry.width, entry.height)
                or image.width * image.height > MAX_FRAME_PIXELS
                or entry.media_type != media_type
                or entry.relative_path != f"objects/{entry.sha256}.{extension}"
            ):
                raise ValueError("animation decoded dimensions or source format disagree")
            image.verify()
        with Image.open(BytesIO(encoded)) as image:
            image.load()
            rgba = image.convert("RGBA")
            bbox = rgba.getchannel("A").getbbox()
            if bbox is None:
                raise ValueError("animation frames cannot be fully transparent")
            return rgba.tobytes(), bbox
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("animation source cannot be decoded") from exc


@dataclass(frozen=True)
class AnimationSource:
    asset: AssetEntry
    encoded: bytes
    rgba: bytes = field(init=False, repr=False)
    visible_bbox: Box = field(init=False)

    def __post_init__(self) -> None:
        if type(self.asset) is not AssetEntry:
            raise ValueError("animation source requires AssetEntry metadata")
        _entry(asdict(self.asset))
        rgba, bbox = _decode(self.encoded, self.asset)
        object.__setattr__(self, "rgba", rgba)
        object.__setattr__(self, "visible_bbox", bbox)

    def image(self) -> Image.Image:
        return Image.frombytes("RGBA", (self.asset.width, self.asset.height), self.rgba)

    @property
    def source_file(self) -> str:
        return "animation_objects/" + self.asset.relative_path.removeprefix("objects/")

    def metadata(self) -> dict[str, object]:
        return {
            "asset": asdict(self.asset),
            "source_file": self.source_file,
            "rgba_sha256": _sha(self.rgba),
            "visible_bbox": list(self.visible_bbox),
        }


@dataclass(frozen=True)
class AnimationFrame:
    source: AnimationSource
    duration_ms: int
    anchor: tuple[int, int]

    def __post_init__(self) -> None:
        if type(self.source) is not AnimationSource:
            raise ValueError("animation frame needs a verified source")
        _integer(self.duration_ms, "frame duration", 1, MAX_CLIP_DURATION_MS)
        if not isinstance(self.anchor, tuple) or len(self.anchor) != 2:
            raise ValueError("animation anchor must be an immutable pixel-edge pair")
        _integer(self.anchor[0], "anchor x", 0, self.source.asset.width)
        _integer(self.anchor[1], "anchor y", 0, self.source.asset.height)

    def metadata(self) -> dict[str, object]:
        return {
            "asset_id": self.source.asset.asset_id,
            "duration_ms": self.duration_ms,
            "anchor": list(self.anchor),
        }


def _clip_identity(value: dict[str, Any]) -> None:
    _token(value["clip_id"], "clip ID")
    _text(value["entity_id"], "entity ID")
    _text(value["game_version"], "game version")
    if not isinstance(value["class_id"], str) or value["class_id"] not in SPRITE_CLASSES:
        raise ValueError("unsupported animation class")
    if not isinstance(value["action_state"], str) or value["action_state"] not in ACTION_STATES:
        raise ValueError("unsupported animation action state")
    if not isinstance(value["playback"], str) or value["playback"] not in {"loop", "once"}:
        raise ValueError("animation playback must be loop or once")


@dataclass(frozen=True)
class AnimationClip:
    clip_id: str
    entity_id: str
    game_version: str
    class_id: str
    action_state: str
    playback: str
    frames: tuple[AnimationFrame, ...]

    def __post_init__(self) -> None:
        _clip_identity({key: getattr(self, key) for key in _CLIP_KEYS - {"frames"}})
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= MAX_FRAMES_PER_CLIP
            or any(type(frame) is not AnimationFrame for frame in self.frames)
        ):
            raise ValueError("animation clips need 1..128 immutable verified frames")
        if self.duration_ms > MAX_CLIP_DURATION_MS:
            raise ValueError("animation clip exceeds its total duration budget")
        if any(
            (frame.source.asset.entity_id, frame.source.asset.game_version)
            != (self.entity_id, self.game_version)
            for frame in self.frames
        ):
            raise ValueError("animation source entity or version disagrees with clip")

    @property
    def duration_ms(self) -> int:
        return sum(frame.duration_ms for frame in self.frames)

    def sample(self, elapsed_ms: int) -> AnimationSample:
        elapsed = _integer(elapsed_ms, "animation elapsed time", 0, MAX_TIME_MS)
        duration = self.duration_ms
        finished = self.playback == "once" and elapsed >= duration
        cycle = elapsed // duration if self.playback == "loop" else 0
        local = elapsed % duration if self.playback == "loop" else min(elapsed, duration - 1)
        start = 0
        for index, frame in enumerate(self.frames):
            if local < start + frame.duration_ms:
                return AnimationSample(self.clip_id, index, cycle, local - start, finished, frame)
            start += frame.duration_ms
        raise AssertionError("validated animation duration did not contain its sample")

    def metadata(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "entity_id": self.entity_id,
            "game_version": self.game_version,
            "class_id": self.class_id,
            "action_state": self.action_state,
            "playback": self.playback,
            "frames": [frame.metadata() for frame in self.frames],
        }


@dataclass(frozen=True)
class AnimationSample:
    clip_id: str
    frame_index: int
    cycle_index: int
    frame_elapsed_ms: int
    finished: bool
    frame: AnimationFrame

    def __post_init__(self) -> None:
        _token(self.clip_id, "sample clip ID")
        _integer(self.frame_index, "sample frame index", 0, MAX_FRAMES_PER_CLIP - 1)
        _integer(self.cycle_index, "sample cycle index", 0, MAX_TIME_MS)
        if type(self.frame) is not AnimationFrame or type(self.finished) is not bool:
            raise ValueError("invalid animation sample frame or completion flag")
        _integer(self.frame_elapsed_ms, "sample frame time", 0, self.frame.duration_ms - 1)


@dataclass(frozen=True)
class AnimationPack:
    scope: str
    clips: tuple[AnimationClip, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or self.scope not in PACK_SCOPES:
            raise ValueError("invalid animation pack scope")
        if (
            not isinstance(self.clips, tuple)
            or not 1 <= len(self.clips) <= MAX_CLIPS
            or any(type(clip) is not AnimationClip for clip in self.clips)
            or sum(len(clip.frames) for clip in self.clips) > MAX_FRAME_REFERENCES
        ):
            raise ValueError("animation pack exceeds clip or frame-reference budget")
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("animation clip IDs must be unique")
        keys = {(clip.entity_id, clip.game_version, clip.action_state) for clip in self.clips}
        if len(keys) != len(self.clips):
            raise ValueError("animation entity/version/action binding is ambiguous")
        classes: dict[tuple[str, str], str] = {}
        for clip in self.clips:
            key = clip.entity_id, clip.game_version
            if key in classes and classes[key] != clip.class_id:
                raise ValueError("one versioned animation entity cannot have multiple classes")
            classes[key] = clip.class_id
        sources = self.sources
        if (
            len(sources) > MAX_SOURCES
            or sum(len(source.encoded) for source in sources) > MAX_PACK_SOURCE_BYTES
            or sum(source.asset.width * source.asset.height for source in sources) > MAX_PACK_PIXELS
        ):
            raise ValueError("animation pack exceeds source bytes or decoded-pixel budget")
        object.__setattr__(self, "clips", tuple(sorted(self.clips, key=lambda clip: clip.clip_id)))

    @property
    def sources(self) -> tuple[AnimationSource, ...]:
        result: dict[str, AnimationSource] = {}
        for clip in self.clips:
            for frame in clip.frames:
                identifier = frame.source.asset.asset_id
                previous = result.get(identifier)
                if previous is not None and previous != frame.source:
                    raise ValueError("animation source ID has contradictory content")
                result[identifier] = frame.source
        return tuple(result[key] for key in sorted(result))

    def get(self, clip_id: str) -> AnimationClip | None:
        _token(clip_id, "clip ID")
        return next((clip for clip in self.clips if clip.clip_id == clip_id), None)

    def resolve(self, entity_id: str, game_version: str, action_state: str) -> AnimationClip | None:
        _text(entity_id, "entity ID")
        _text(game_version, "game version")
        if not isinstance(action_state, str) or action_state not in ACTION_STATES:
            raise ValueError("unsupported animation action state")
        return next(
            (
                clip
                for clip in self.clips
                if (clip.entity_id, clip.game_version, clip.action_state)
                == (entity_id, game_version, action_state)
            ),
            None,
        )

    def manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ANIMATION_SCHEMA_VERSION,
            "kind": "animation-pack",
            "scope": self.scope,
            "clips": [clip.metadata() for clip in self.clips],
            "sources": {source.asset.asset_id: source.metadata() for source in self.sources},
        }
        return {**payload, "pack_sha256": _sha(canonical_json(payload).encode("utf-8"))}

    def write_sources(self, output: Path) -> None:
        """Copy captured source bytes into a new subtree of an existing bundle."""
        output = Path(output)
        _directory(output)
        directory = output / "animation_objects"
        if directory.exists() or directory.is_symlink():
            raise ValueError("animation source output already exists")
        directory.mkdir()
        written: set[str] = set()
        created: list[Path] = []
        try:
            for source in self.sources:
                if source.source_file in written:
                    continue
                target = output / source.source_file
                with target.open("xb") as stream:
                    created.append(target)
                    stream.write(source.encoded)
                written.add(source.source_file)
        except BaseException:
            _remove_created(created, directory)
            raise


def _remove_created(files: list[Path], directory: Path) -> None:
    """Best-effort rollback of our own files; never recursively remove strangers."""
    for path in reversed(files):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass


def _descriptors(value: object) -> tuple[list[dict[str, Any]], set[str]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CLIPS:
        raise ValueError("animation clips exceed their count budget")
    clips: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    references = 0
    for item in value:
        clip = _object(item, _CLIP_KEYS, "clip")
        _clip_identity(clip)
        frames = clip["frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES_PER_CLIP:
            raise ValueError("animation frames exceed their count budget")
        references += len(frames)
        if references > MAX_FRAME_REFERENCES:
            raise ValueError("animation frame-reference budget exceeded")
        duration = 0
        for frame in frames:
            descriptor = _object(frame, {"asset_id", "duration_ms", "anchor"}, "frame")
            identifiers.add(_hash(descriptor["asset_id"], "asset ID"))
            duration += _integer(
                descriptor["duration_ms"], "frame duration", 1, MAX_CLIP_DURATION_MS
            )
            anchor = descriptor["anchor"]
            if not isinstance(anchor, list) or len(anchor) != 2:
                raise ValueError("animation anchor must be a two-element JSON list")
            for coordinate in anchor:
                _integer(coordinate, "anchor coordinate", 0, MAX_FRAME_DIMENSION)
        if duration > MAX_CLIP_DURATION_MS or len(identifiers) > MAX_SOURCES:
            raise ValueError("animation duration or source-count budget exceeded")
        clips.append(clip)
    return clips, identifiers


def _pack(
    scope: str, descriptors: list[dict[str, Any]], sources: dict[str, AnimationSource]
) -> AnimationPack:
    clips: list[AnimationClip] = []
    for descriptor in descriptors:
        frames = tuple(
            AnimationFrame(sources[frame["asset_id"]], frame["duration_ms"], tuple(frame["anchor"]))
            for frame in descriptor["frames"]
        )
        identity = {key: value for key, value in descriptor.items() if key != "frames"}
        clips.append(AnimationClip(**identity, frames=frames))
    return AnimationPack(scope, tuple(clips))


def _load_sources(
    entries: dict[str, AssetEntry], paths: dict[str, Path]
) -> dict[str, AnimationSource]:
    if sum(entry.width * entry.height for entry in entries.values()) > MAX_PACK_PIXELS:
        raise ValueError("animation pack exceeds decoded-pixel budget")
    declared_bytes = 0
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError("animation source must be a regular nonsymlink file")
        size = path.stat().st_size
        if not 1 <= size <= MAX_SOURCE_BYTES:
            raise ValueError("animation source exceeds byte budget or is empty")
        declared_bytes += size
    if declared_bytes > MAX_PACK_SOURCE_BYTES:
        raise ValueError("animation pack exceeds source-byte budget")
    sources: dict[str, AnimationSource] = {}
    actual_bytes = 0
    for identifier in sorted(entries):
        encoded = _read(paths[identifier], MAX_SOURCE_BYTES)
        actual_bytes += len(encoded)
        if actual_bytes > MAX_PACK_SOURCE_BYTES:
            raise ValueError("animation pack exceeds source-byte budget")
        sources[identifier] = AnimationSource(entries[identifier], encoded)
    return sources


def load_animation_pack(registry: Path, mapping: Path) -> AnimationPack:
    """Load explicit frame IDs from a strict mapping and verified registry bytes."""
    data = _object(_json(Path(mapping)), {"schema_version", "scope", "clips"}, "mapping")
    _version(data["schema_version"])
    if not isinstance(data["scope"], str) or data["scope"] not in PACK_SCOPES:
        raise ValueError("invalid animation mapping scope")
    descriptors, identifiers = _descriptors(data["clips"])
    registry = Path(registry)
    for directory in (registry, registry / "records", registry / "objects"):
        _directory(directory)
    entries: dict[str, AssetEntry] = {}
    for identifier in sorted(identifiers):
        entry = _entry(_json(registry / "records" / f"{identifier}.json", 64 * 1024))
        if entry.asset_id != identifier:
            raise ValueError("animation source record ID mismatch")
        entries[identifier] = entry
    paths = {identifier: registry / entry.relative_path for identifier, entry in entries.items()}
    return _pack(data["scope"], descriptors, _load_sources(entries, paths))


def export_animation_pack(pack: AnimationPack, output: Path) -> dict[str, object]:
    """Export immutable captured sources and manifest into a new directory."""
    if type(pack) is not AnimationPack:
        raise ValueError("animation export requires a verified AnimationPack")
    manifest = pack.manifest()
    encoded = (canonical_json(manifest) + "\n").encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("animation manifest exceeds byte budget")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("animation export output already exists")
    output.mkdir(parents=True)
    sources_written = False
    manifest_created = False
    manifest_path = output / "animation-pack.json"
    try:
        pack.write_sources(output)
        sources_written = True
        with manifest_path.open("xb") as stream:
            manifest_created = True
            stream.write(encoded)
    except BaseException:
        if manifest_created:
            _remove_created([manifest_path], output)
        if sources_written:
            _remove_created(
                list({output / source.source_file for source in pack.sources}),
                output / "animation_objects",
            )
        _remove_created([], output)
        raise
    return manifest


def verify_animation_pack(bundle: Path, manifest: object | None = None) -> AnimationPack:
    """Rebuild a pack exclusively from its embedded metadata and copied sources."""
    bundle = Path(bundle)
    _directory(bundle)
    value = _json(bundle / "animation-pack.json") if manifest is None else manifest
    data = _object(
        value,
        {"schema_version", "kind", "scope", "clips", "sources", "pack_sha256"},
        "pack manifest",
    )
    _version(data["schema_version"])
    if (
        data["kind"] != "animation-pack"
        or not isinstance(data["scope"], str)
        or data["scope"] not in PACK_SCOPES
    ):
        raise ValueError("invalid animation pack kind or scope")
    _hash(data["pack_sha256"], "pack checksum")
    descriptors, identifiers = _descriptors(data["clips"])
    metadata = data["sources"]
    if (
        not isinstance(metadata, dict)
        or not 1 <= len(metadata) <= MAX_SOURCES
        or set(metadata) != identifiers
    ):
        raise ValueError("animation manifest source IDs disagree with referenced frames")
    directory = bundle / "animation_objects"
    _directory(directory)
    entries: dict[str, AssetEntry] = {}
    paths: dict[str, Path] = {}
    for identifier in sorted(identifiers):
        item = _object(
            metadata[identifier],
            {"asset", "source_file", "rgba_sha256", "visible_bbox"},
            "source metadata",
        )
        entry = _entry(item["asset"])
        expected_path = "animation_objects/" + entry.relative_path.removeprefix("objects/")
        if entry.asset_id != identifier or item["source_file"] != expected_path:
            raise ValueError("animation source ID or content-addressed path mismatch")
        _hash(item["rgba_sha256"], "decoded checksum")
        bbox = item["visible_bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise ValueError("animation visible bbox must contain four integers")
        if not (0 <= bbox[0] < bbox[2] <= entry.width and 0 <= bbox[1] < bbox[3] <= entry.height):
            raise ValueError("animation visible bbox lies outside source pixels")
        entries[identifier], paths[identifier] = entry, bundle / expected_path
    if set(islice(directory.iterdir(), MAX_SOURCES + 1)) != set(paths.values()):
        raise ValueError("animation source subtree contains unlisted files")
    payload = {key: item for key, item in data.items() if key != "pack_sha256"}
    canonical = canonical_json(payload).encode("utf-8")
    if len(canonical) > MAX_MANIFEST_BYTES or _sha(canonical) != data["pack_sha256"]:
        raise ValueError("animation pack metadata checksum mismatch or byte budget exceeded")
    result = _pack(data["scope"], descriptors, _load_sources(entries, paths))
    if canonical_json(result.manifest()) != canonical_json(data):
        raise ValueError("animation decoded pixels or alpha bounds disagree with manifest")
    return result


@dataclass(frozen=True)
class AnimationPlacement:
    clip_id: str
    elapsed_ms: int
    position: tuple[int, int]
    layer: int = 0
    flip_x: bool = False

    def __post_init__(self) -> None:
        _token(self.clip_id, "clip ID")
        _integer(self.elapsed_ms, "animation elapsed time", 0, MAX_TIME_MS)
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError("animation position must be an immutable coordinate pair")
        for coordinate in self.position:
            _integer(coordinate, "animation position", -1_000_000, 1_000_000)
        _integer(self.layer, "animation layer", -4096, 4096)
        if type(self.flip_x) is not bool:
            raise ValueError("animation flip_x must be boolean")


@dataclass(frozen=True)
class PlacedAnimation:
    placement_index: int
    clip_id: str
    frame_index: int
    finished: bool
    canvas_bounds: Box
    visible_bbox: Box | None

    def __post_init__(self) -> None:
        _integer(self.placement_index, "placement index", 0, MAX_PLACEMENTS - 1)
        _token(self.clip_id, "placement clip ID")
        _integer(self.frame_index, "placement frame index", 0, MAX_FRAMES_PER_CLIP - 1)
        if type(self.finished) is not bool:
            raise ValueError("placement completion flag must be boolean")
        for label, box in (
            ("canvas bounds", self.canvas_bounds),
            ("visible bbox", self.visible_bbox),
        ):
            if box is None and label == "visible bbox":
                continue
            if not isinstance(box, tuple) or len(box) != 4:
                raise ValueError(f"animation {label} must be an immutable four-coordinate box")
            for coordinate in box:
                _integer(coordinate, label, -1_002_048, 1_002_048)
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"invalid animation {label} extent")
        if self.visible_bbox is not None and (
            min(self.visible_bbox) < 0
            or max(self.visible_bbox) > 4096
            or self.visible_bbox[0] < self.canvas_bounds[0]
            or self.visible_bbox[1] < self.canvas_bounds[1]
            or self.visible_bbox[2] > self.canvas_bounds[2]
            or self.visible_bbox[3] > self.canvas_bounds[3]
        ):
            raise ValueError("animation visible bbox must lie inside canvas and source bounds")


@dataclass(frozen=True)
class AnimationComposition:
    width: int
    height: int
    rgba: bytes = field(repr=False)
    placements: tuple[PlacedAnimation, ...]
    visible_bbox: Box | None

    def __post_init__(self) -> None:
        _integer(self.width, "composition width", 1, 4096)
        _integer(self.height, "composition height", 1, 4096)
        if self.width * self.height > MAX_CANVAS_PIXELS:
            raise ValueError("animation composition exceeds pixel budget")
        if type(self.rgba) is not bytes or len(self.rgba) != self.width * self.height * 4:
            raise ValueError("animation composition RGBA buffer has invalid size or type")
        if (
            not isinstance(self.placements, tuple)
            or not 1 <= len(self.placements) <= MAX_PLACEMENTS
            or any(type(placement) is not PlacedAnimation for placement in self.placements)
            or tuple(placement.placement_index for placement in self.placements)
            != tuple(range(len(self.placements)))
        ):
            raise ValueError("animation composition placements must have canonical indices")
        actual = (
            Image.frombytes("RGBA", (self.width, self.height), self.rgba).getchannel("A").getbbox()
        )
        if self.visible_bbox != actual:
            raise ValueError("animation composition alpha bbox disagrees with pixels")

    def image(self) -> Image.Image:
        return Image.frombytes("RGBA", (self.width, self.height), self.rgba)

    def png_bytes(self) -> bytes:
        output = BytesIO()
        self.image().save(output, format="PNG", optimize=False, compress_level=9)
        return output.getvalue()

    def save_png(self, path: Path) -> None:
        if Path(path).is_symlink():
            raise ValueError("animation composition output cannot be a symlink")
        with Path(path).open("xb") as stream:
            stream.write(self.png_bytes())


def compose_animation(
    pack: AnimationPack,
    placements: tuple[AnimationPlacement, ...],
    *,
    size: tuple[int, int] = (320, 192),
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> AnimationComposition:
    """Paste samples at anchor positions, ordered by layer then input position.

    Each placement's bbox is its nonzero alpha after canvas clipping. Other
    placements may cover it later; these are not full-scene occlusion labels.
    Mirroring also mirrors the pixel-edge anchor. No scaling is performed.
    """
    if type(pack) is not AnimationPack:
        raise ValueError("animation composition requires a verified pack")
    if (
        not isinstance(placements, tuple)
        or not 1 <= len(placements) <= MAX_PLACEMENTS
        or any(type(placement) is not AnimationPlacement for placement in placements)
    ):
        raise ValueError("animation composition exceeds placement budget")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("animation canvas size must be an immutable pair")
    width = _integer(size[0], "canvas width", 1, 4096)
    height = _integer(size[1], "canvas height", 1, 4096)
    if width * height > MAX_CANVAS_PIXELS:
        raise ValueError("animation canvas exceeds pixel budget")
    if not isinstance(background, tuple) or len(background) != 4:
        raise ValueError("animation background must be an RGBA tuple")
    for channel in background:
        _integer(channel, "background channel", 0, 255)
    samples: list[AnimationSample] = []
    work = width * height
    for placement in placements:
        clip = pack.get(placement.clip_id)
        if clip is None:
            raise ValueError("animation placement references an unknown clip")
        sample = clip.sample(placement.elapsed_ms)
        work += sample.frame.source.asset.width * sample.frame.source.asset.height
        if work > MAX_COMPOSITION_WORK_PIXELS:
            raise ValueError("animation composition exceeds pixel-work budget")
        samples.append(sample)
    canvas = Image.new("RGBA", size, background)
    rendered: list[PlacedAnimation] = []
    for index in sorted(range(len(placements)), key=lambda item: (placements[item].layer, item)):
        placement, sample = placements[index], samples[index]
        source = sample.frame.source
        sw, sh = source.asset.width, source.asset.height
        ax, ay = sample.frame.anchor
        if placement.flip_x:
            ax = sw - ax
        left, top = placement.position[0] - ax, placement.position[1] - ay
        bounds = left, top, left + sw, top + sh
        visible: Box | None = None
        x1, y1, x2, y2 = max(0, left), max(0, top), min(width, left + sw), min(height, top + sh)
        if x1 < x2 and y1 < y2:
            image = source.image()
            if placement.flip_x:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            cropped = image.crop((x1 - left, y1 - top, x2 - left, y2 - top))
            alpha = cropped.getchannel("A").getbbox()
            if alpha is not None:
                visible = alpha[0] + x1, alpha[1] + y1, alpha[2] + x1, alpha[3] + y1
                canvas.alpha_composite(cropped, dest=(x1, y1))
        rendered.append(
            PlacedAnimation(
                index, sample.clip_id, sample.frame_index, sample.finished, bounds, visible
            )
        )
    return AnimationComposition(
        width,
        height,
        canvas.tobytes(),
        tuple(sorted(rendered, key=lambda item: item.placement_index)),
        canvas.getchannel("A").getbbox(),
    )
