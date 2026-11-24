"""Verified local registry images mapped to optional renderer classes.

This module never downloads assets. License and permission fields remain the
importer's assertions; hashes establish integrity, not ownership or authenticity.
"""

import hashlib
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .asset_registry import MAX_ASSET_BYTES, AssetEntry, verify_asset
from .catalog import canonical_json, strict_json, validate_provenance

SPRITE_CLASSES = frozenset({"player", "monster", "loot", "portal", "npc", "platform", "ladder"})
PACK_SCOPES = frozenset({"synthetic_fixture", "authorized_assets"})
_FORMATS = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_image(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("sprite object cannot be a symlink")
    with path.open("rb") as stream:
        data = stream.read(MAX_ASSET_BYTES + 1)
    if len(data) > MAX_ASSET_BYTES:
        raise ValueError("sprite object exceeds size limit")
    return data


def _rgba(data: bytes, entry: AssetEntry) -> bytes:
    if not data or len(data) > MAX_ASSET_BYTES or _hash(data) != entry.sha256:
        raise ValueError("sprite source checksum mismatch")
    with Image.open(BytesIO(data)) as image:
        if image.format not in _FORMATS or getattr(image, "n_frames", 1) != 1:
            raise ValueError("sprites must be single-frame PNG/JPEG/WebP images")
        extension, media_type = _FORMATS[image.format]
        if (
            image.size != (entry.width, entry.height)
            or image.width * image.height > 16_000_000
            or entry.media_type != media_type
            or entry.relative_path != f"objects/{entry.sha256}.{extension}"
        ):
            raise ValueError("sprite image metadata mismatch")
        image.load()
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getbbox() is None:
            raise ValueError("sprite must contain visible pixels")
        return rgba.tobytes()


@dataclass(frozen=True)
class SpriteAsset:
    class_id: str
    asset: AssetEntry
    encoded: bytes
    rgba: bytes

    def __post_init__(self) -> None:
        if (
            not isinstance(self.class_id, str)
            or self.class_id not in SPRITE_CLASSES
            or not isinstance(self.asset, AssetEntry)
        ):
            raise ValueError("unknown sprite class or metadata")
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > 4096
            for value in (
                self.asset.entity_id,
                self.asset.game_version,
                self.asset.relative_path,
                self.asset.media_type,
            )
        ):
            raise ValueError("invalid sprite identity metadata")
        if any(
            type(value) is not int or value <= 0 for value in (self.asset.width, self.asset.height)
        ):
            raise ValueError("invalid sprite dimensions")
        validate_provenance(self.asset.source_uri, self.asset.permission, self.asset.license)
        metadata = asdict(self.asset)
        asset_id = metadata.pop("asset_id")
        if _hash(canonical_json(metadata).encode()) != asset_id:
            raise ValueError("sprite metadata checksum mismatch")
        if not isinstance(self.encoded, bytes) or not isinstance(self.rgba, bytes):
            raise ValueError("sprite buffers must be immutable bytes")
        if _rgba(self.encoded, self.asset) != self.rgba:
            raise ValueError("sprite RGBA pixels disagree with source image")

    def image(self) -> Image.Image:
        return Image.frombytes("RGBA", (self.asset.width, self.asset.height), self.rgba)

    @property
    def source_file(self) -> str:
        return "sprite_objects/" + Path(self.asset.relative_path).name

    def metadata(self) -> dict[str, object]:
        return {
            "asset": asdict(self.asset),
            "source_file": self.source_file,
            "rgba_sha256": _hash(self.rgba),
        }


@dataclass(frozen=True)
class SpritePack:
    scope: str
    entries: tuple[SpriteAsset, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or self.scope not in PACK_SCOPES:
            raise ValueError("sprite pack scope must be synthetic_fixture or authorized_assets")
        if (
            not isinstance(self.entries, tuple)
            or not self.entries
            or any(not isinstance(entry, SpriteAsset) for entry in self.entries)
        ):
            raise ValueError("sprite pack needs immutable, nonempty SpriteAsset entries")
        if len({entry.class_id for entry in self.entries}) != len(self.entries):
            raise ValueError("sprite pack contains duplicate classes")

    def get(self, class_id: str) -> SpriteAsset | None:
        return next((entry for entry in self.entries if entry.class_id == class_id), None)

    def manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "scope": self.scope,
            "classes": {
                entry.class_id: entry.metadata()
                for entry in sorted(self.entries, key=lambda e: e.class_id)
            },
        }
        return {**payload, "pack_sha256": _hash(canonical_json(payload).encode())}

    def write_sources(self, output: Path) -> None:
        directory = output / "sprite_objects"
        directory.mkdir()
        seen: set[str] = set()
        for entry in self.entries:
            if entry.source_file in seen:
                continue
            with (output / entry.source_file).open("xb") as stream:
                stream.write(entry.encoded)
            seen.add(entry.source_file)


def load_sprite_pack(registry: Path, mapping: Path) -> SpritePack:
    """Load a strict class→asset-ID mapping and independently verified local bytes."""
    mapping = Path(mapping)
    if mapping.stat().st_size > 64 * 1024:
        raise ValueError("sprite mapping exceeds 64 KiB")
    value: Any = strict_json(mapping.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "scope", "classes"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["classes"], dict)
        or not value["classes"]
        or not isinstance(value["scope"], str)
        or value["scope"] not in PACK_SCOPES
    ):
        raise ValueError("invalid sprite mapping schema")
    if not set(value["classes"]) <= SPRITE_CLASSES:
        raise ValueError("unknown sprite class")
    entries: list[SpriteAsset] = []
    for class_id, asset_id in sorted(value["classes"].items()):
        asset = verify_asset(registry, asset_id)
        encoded = _read_image(Path(registry) / asset.relative_path)
        entries.append(SpriteAsset(class_id, asset, encoded, _rgba(encoded, asset)))
    return SpritePack(value["scope"], tuple(entries))


def verify_dataset_sprite_pack(dataset: Path, manifest: object) -> SpritePack:
    """Check copied source bytes, provenance metadata, decoded RGBA and pack hash."""
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "scope", "classes", "pack_sha256"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["classes"], dict)
        or not manifest["classes"]
        or not set(manifest["classes"]) <= SPRITE_CLASSES
    ):
        raise ValueError("invalid dataset sprite pack manifest")
    directory = dataset / "sprite_objects"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("invalid sprite source directory")
    entries: list[SpriteAsset] = []
    expected: set[Path] = set()
    for class_id, metadata in sorted(manifest["classes"].items()):
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"asset", "source_file", "rgba_sha256"}
            or not isinstance(metadata["asset"], dict)
            or set(metadata["asset"]) != set(AssetEntry.__dataclass_fields__)
        ):
            raise ValueError("invalid dataset sprite metadata")
        try:
            asset = AssetEntry(**metadata["asset"])
            source_file = "sprite_objects/" + Path(asset.relative_path).name
            if metadata["source_file"] != source_file:
                raise ValueError("sprite source path mismatch")
            source = dataset / source_file
            if not source.resolve().is_relative_to(dataset.resolve()):
                raise ValueError("sprite source path escapes dataset")
            encoded = _read_image(source)
            entry = SpriteAsset(class_id, asset, encoded, _rgba(encoded, asset))
        except (TypeError, AttributeError) as exc:
            raise ValueError("invalid dataset sprite metadata types") from exc
        entries.append(entry)
        expected.add(source)
    if set(directory.iterdir()) != expected:
        raise ValueError("unlisted dataset sprite source files")
    result = SpritePack(manifest["scope"], tuple(entries))
    if result.manifest() != manifest:
        raise ValueError("dataset sprite pack checksum mismatch")
    return result
