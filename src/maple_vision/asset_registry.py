"""Local, content-addressed registry for explicitly authorized image assets."""

from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .catalog import canonical_json, strict_json, validate_provenance

MAX_ASSET_BYTES = 32 * 1024 * 1024
MAX_ASSET_PIXELS = 16 * 1024 * 1024
_HASH = re.compile(r"^[0-9a-f]{64}$")
_FORMATS = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
}


@dataclass(frozen=True)
class AssetEntry:
    asset_id: str
    entity_id: str
    game_version: str
    source_uri: str
    license: str
    permission: str
    sha256: str
    relative_path: str
    width: int
    height: int
    media_type: str


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be a nonempty string of at most 4096 characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _decode(data: bytes) -> tuple[int, int, str, str]:
    if not data or len(data) > MAX_ASSET_BYTES:
        raise ValueError("asset must be between 1 byte and 32 MiB")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("asset decoding requires Pillow; install the vision extra") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in _FORMATS:
                raise ValueError("asset format must be PNG, JPEG, or WEBP")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_ASSET_PIXELS:
                raise ValueError("asset dimensions exceed the 16-megapixel limit")
            extension, media_type = _FORMATS[image.format]
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated images must be imported as separately labeled frames")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"asset image cannot be decoded: {exc}") from exc
    return width, height, extension, media_type


def _directories(registry: Path, *, create: bool) -> tuple[Path, Path]:
    if registry.is_symlink():
        raise ValueError("registry root must not be a symlink")
    if create:
        registry.mkdir(parents=True, exist_ok=True)
    objects, records = registry / "objects", registry / "records"
    for directory in (objects, records):
        if directory.is_symlink():
            raise ValueError("registry subdirectory must not be a symlink")
        if create:
            directory.mkdir(exist_ok=True)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    return objects, records


def _publish(path: Path, data: bytes) -> None:
    """Publish complete immutable bytes without replacing an existing artifact."""
    if path.is_symlink():
        raise ValueError("registry files must not be symlinks")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable registry content mismatch: {path.name}")
        return
    staging = path.parent / (".pending-" + uuid4().hex)
    try:
        with staging.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and fails if another writer wins.
            os.link(staging, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"immutable registry content mismatch: {path.name}") from None
    finally:
        staging.unlink(missing_ok=True)


def import_asset(
    registry_dir: str | Path,
    source_path: str | Path,
    *,
    entity_id: str,
    game_version: str,
    source_uri: str,
    license: str,
    permission: str,
) -> AssetEntry:
    """Decode an authorized local image and preserve both bytes and provenance.

    Metadata is an assertion from the importer, not a proof of legal ownership.
    The registry makes it inspectable; it does not download or extract game files.
    """
    _text(entity_id, "entity_id")
    _text(game_version, "game_version")
    validate_provenance(source_uri, permission, license)
    source = Path(source_path)
    if source.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError("asset exceeds 32 MiB")
    data = source.read_bytes()
    width, height, extension, media_type = _decode(data)
    sha256 = hashlib.sha256(data).hexdigest()
    relative_path = f"objects/{sha256}.{extension}"
    metadata: dict[str, Any] = {
        "entity_id": entity_id,
        "game_version": game_version,
        "source_uri": source_uri,
        "license": license,
        "permission": permission,
        "sha256": sha256,
        "relative_path": relative_path,
        "width": width,
        "height": height,
        "media_type": media_type,
    }
    asset_id = hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()
    entry = AssetEntry(asset_id=asset_id, **metadata)
    registry = Path(registry_dir)
    _, records = _directories(registry, create=True)
    _publish(registry / relative_path, data)
    _publish(records / f"{asset_id}.json", (canonical_json(asdict(entry)) + "\n").encode("utf-8"))
    return entry


def _read_entry(path: Path) -> AssetEntry:
    if path.is_symlink():
        raise ValueError("asset metadata must not be a symlink")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("asset metadata exceeds 64 KiB")
    metadata = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or set(metadata) != set(AssetEntry.__dataclass_fields__):
        raise ValueError("invalid asset metadata fields")
    entry = AssetEntry(**metadata)
    for field in ("asset_id", "sha256"):
        value = getattr(entry, field)
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"invalid {field}")
    for field in ("entity_id", "game_version", "relative_path", "media_type"):
        _text(getattr(entry, field), field)
    validate_provenance(entry.source_uri, entry.permission, entry.license)
    for dimension in (entry.width, entry.height):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("invalid asset dimensions")
    metadata.pop("asset_id")
    calculated = hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()
    if entry.asset_id != calculated or path.name != f"{entry.asset_id}.json":
        raise ValueError("asset metadata SHA256 mismatch")
    valid_paths = {f"objects/{entry.sha256}.{ext}" for ext, _ in _FORMATS.values()}
    if entry.relative_path not in valid_paths:
        raise ValueError("asset path is not content-addressed")
    return entry


def verify_asset(registry_dir: str | Path, asset_id: str) -> AssetEntry:
    if not isinstance(asset_id, str) or not _HASH.fullmatch(asset_id):
        raise ValueError("asset_id must be a lowercase SHA256")
    registry = Path(registry_dir)
    _, records = _directories(registry, create=False)
    entry = _read_entry(records / f"{asset_id}.json")
    image_path = registry / entry.relative_path
    if image_path.is_symlink():
        raise ValueError("asset object must not be a symlink")
    if image_path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError("stored asset exceeds 32 MiB")
    data = image_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ValueError("asset content SHA256 mismatch")
    width, height, extension, media_type = _decode(data)
    if (width, height, media_type, image_path.suffix) != (
        entry.width,
        entry.height,
        entry.media_type,
        f".{extension}",
    ):
        raise ValueError("decoded asset metadata mismatch")
    return entry


def list_assets(registry_dir: str | Path) -> tuple[AssetEntry, ...]:
    registry = Path(registry_dir)
    _, records = _directories(registry, create=False)
    return tuple(verify_asset(registry, path.stem) for path in sorted(records.glob("*.json")))
