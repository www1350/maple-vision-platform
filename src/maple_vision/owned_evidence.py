"""Seal and independently verify the project-owned window replay bundle."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 1_000_000
_MAX_MEDIA_BYTES = 25_000_000


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_media(root: Path, name: object, suffix: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(suffix):
        raise ValueError(f"invalid owned-window media name: {name!r}")
    path = root / name
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing owned-window media: {name}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"owned-window media must be a regular file: {name}")
    if path.stat().st_size > _MAX_MEDIA_BYTES:
        raise ValueError(f"owned-window media is too large: {name}")
    return path


def _png_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"expected one PNG frame: {path.name}")
        rgb = image.convert("RGB")
        width, height = rgb.size
        rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
    return {
        "file_sha256": _sha256_file(path),
        "rgb_sha256": rgb_sha256,
        "width": width,
        "height": height,
    }


def _gif_metadata(path: Path) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    with Image.open(path) as image:
        if image.format != "GIF":
            raise ValueError(f"expected GIF replay: {path.name}")
        width, height = image.size
        for index in range(int(getattr(image, "n_frames", 1))):
            image.seek(index)
            rgb = image.convert("RGB")
            frames.append(
                {
                    "index": index,
                    "duration_ms": int(image.info.get("duration", 0)),
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
            )
    return {
        "file_sha256": _sha256_file(path),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "frames": frames,
    }


def seal_owned_window_evidence(report: dict[str, Any], root: Path) -> dict[str, Any]:
    """Add decoded and encoded media identities, then seal the JSON manifest."""
    sealed = deepcopy(report)
    sealed["schema_version"] = 2
    captures = sealed.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("owned-window evidence needs capture frames")
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("invalid owned-window capture")
        path = _regular_media(root, capture.get("file"), ".png")
        capture.update(_png_metadata(path))

    key_file = sealed.pop("confirmed_key_file", "confirmed-key.png")
    key_path = _regular_media(root, key_file, ".png")
    sealed["confirmed_key"] = {"file": key_file, **_png_metadata(key_path)}

    replay_value = sealed.get("replay", "replay.gif")
    replay_file = replay_value if isinstance(replay_value, str) else replay_value.get("file")
    replay_path = _regular_media(root, replay_file, ".gif")
    sealed["replay"] = {"file": replay_file, **_gif_metadata(replay_path)}

    payload = {key: value for key, value in sealed.items() if key != "integrity"}
    sealed["integrity"] = {
        "algorithm": "sha256",
        "manifest_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }
    return sealed


def _same_metadata(declared: object, actual: dict[str, Any], label: str) -> None:
    if not isinstance(declared, dict):
        raise ValueError(f"invalid {label} metadata")
    for key, value in actual.items():
        if declared.get(key) != value:
            raise ValueError(f"{label} {key} does not match the media")


def verify_owned_window_evidence(root: Path) -> dict[str, Any]:
    """Reject an incomplete, substituted or path-escaping owned-window bundle."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("owned-window evidence root must be a real directory")
    evidence_path = root / "evidence.json"
    try:
        mode = evidence_path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError("missing owned-window evidence.json") from exc
    if evidence_path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError("owned-window evidence.json must be a regular file")
    raw = evidence_path.read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("owned-window evidence.json is too large")
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid owned-window evidence.json") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 2:
        raise ValueError("unsupported owned-window evidence schema")
    if (
        report.get("verified") is not True
        or report.get("scope") != "project_owned_native_tk_window"
    ):
        raise ValueError("bundle is not verified project-owned-window evidence")

    integrity = report.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        raise ValueError("missing owned-window manifest integrity")
    expected_manifest = integrity.get("manifest_sha256")
    if not isinstance(expected_manifest, str) or not _SHA256.fullmatch(expected_manifest):
        raise ValueError("invalid owned-window manifest digest")
    payload = {key: value for key, value in report.items() if key != "integrity"}
    actual_manifest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if actual_manifest != expected_manifest:
        raise ValueError("owned-window manifest was modified")

    declared_files = {"evidence.json"}
    captures = report.get("captures")
    if not isinstance(captures, list) or len(captures) != 5:
        raise ValueError("owned-window first-use evidence needs five captures")
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict) or not isinstance(capture.get("label"), str):
            raise ValueError("invalid owned-window capture entry")
        name = capture.get("file")
        path = _regular_media(root, name, ".png")
        if not isinstance(name, str):  # narrowed by _regular_media at runtime
            raise AssertionError("unreachable invalid capture name")
        if name in declared_files:
            raise ValueError("duplicate owned-window media name")
        declared_files.add(name)
        _same_metadata(capture, _png_metadata(path), f"capture {index}")

    confirmed_key = report.get("confirmed_key")
    if not isinstance(confirmed_key, dict):
        raise ValueError("missing confirmed-key media")
    key_name = confirmed_key.get("file")
    key_path = _regular_media(root, key_name, ".png")
    if not isinstance(key_name, str):  # narrowed by _regular_media at runtime
        raise AssertionError("unreachable invalid confirmed-key name")
    if key_name in declared_files:
        raise ValueError("duplicate confirmed-key media name")
    declared_files.add(key_name)
    _same_metadata(confirmed_key, _png_metadata(key_path), "confirmed key")

    replay = report.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("missing replay media")
    replay_name = replay.get("file")
    replay_path = _regular_media(root, replay_name, ".gif")
    if not isinstance(replay_name, str):  # narrowed by _regular_media at runtime
        raise AssertionError("unreachable invalid replay name")
    if replay_name in declared_files:
        raise ValueError("duplicate replay media name")
    declared_files.add(replay_name)
    replay_metadata = _gif_metadata(replay_path)
    _same_metadata(replay, replay_metadata, "replay")
    if replay_metadata["frame_count"] != len(captures):
        raise ValueError("replay frame count does not match the captures")

    actual_files = {path.name for path in root.iterdir()}
    if actual_files != declared_files:
        raise ValueError("owned-window bundle contains missing or undeclared files")
    return {
        "verified": True,
        "scope": report["scope"],
        "manifest_sha256": actual_manifest,
        "media_files": len(declared_files) - 1,
        "replay_frames": replay_metadata["frame_count"],
    }
