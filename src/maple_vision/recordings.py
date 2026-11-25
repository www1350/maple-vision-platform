"""Local recording ingestion and perception-only replay, without control backends."""

from __future__ import annotations

import hashlib
import importlib
import io
import math
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from typing import Any
from unittest.mock import patch

from PIL import Image

from .catalog import canonical_json, strict_json, validate_provenance
from .contracts import Detection, Status
from .onnx_detector import ModelManifest
from .scene import ImageFrame
from .vision import Detector, PixelPerception, VisualObservation

RECORDING_SCHEMA_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_FRAME_PIXELS = 16 * 1024 * 1024
MAX_TOTAL_PIXELS = 1_000_000_000
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
_VIDEO_LOCK = Lock()
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "scope",
    "session_id",
    "source_kind",
    "source_uri",
    "permission",
    "license",
    "source_name",
    "source_sha256",
    "source_hash_basis",
    "fps",
    "timebase",
    "frame_count",
    "width",
    "height",
    "max_frames",
    "total_source_frames",
    "truncated",
    "completion_reason",
    "frames",
}
_FRAME_KEYS = {
    "sequence",
    "timestamp_ms",
    "file",
    "sha256",
    "rgb_sha256",
    "source_name",
    "source_sha256",
    "width",
    "height",
}


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _fps(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.1 <= value <= 240
    ):
        raise ValueError("fps must be finite in [0.1, 240]")
    return float(value)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or any(ord(char) < 32 for char in value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("source_name must be a plain file name")
    return value


def _timestamp(sequence: int, fps: float) -> int:
    return round(sequence * 1000 / fps)


def _png(data: bytes) -> tuple[int, int, bytes]:
    if not data or len(data) > MAX_FRAME_BYTES:
        raise ValueError("PNG frame exceeds the 64 MiB limit or is empty")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("recording frame must be a PNG image")
            width, height = image.size
            if not (
                1 <= width <= 8192 and 1 <= height <= 8192 and width * height <= MAX_FRAME_PIXELS
            ):
                raise ValueError("recording frame exceeds dimension/pixel limits")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated PNG must be decoded into separate frames first")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if "A" in image.getbands() and image.getchannel("A").getextrema() != (255, 255):
                raise ValueError("recording PNG must be opaque; flatten transparency explicitly")
            if "transparency" in image.info:
                rgba = image.convert("RGBA")
                if rgba.getchannel("A").getextrema() != (255, 255):
                    raise ValueError("recording PNG must be opaque")
            rgb = image.convert("RGB")
            return width, height, rgb.tobytes()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"recording PNG cannot be decoded: {exc}") from exc


@contextmanager
def _new_output(output: Path, source: Path) -> Iterator[tuple[Path, Path]]:
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the source recording")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield stage, destination
        # Reserve the destination so a racing writer cannot be overwritten.
        destination.mkdir()
        try:
            for child in sorted(
                stage.iterdir(), key=lambda p: p.name in {"manifest.json", "report.json"}
            ):
                child.rename(destination / child.name)
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_frame(
    stage: Path,
    sequence: int,
    timestamp_ms: int,
    width: int,
    height: int,
    pixels: bytes,
    source_name: str,
    source_sha256: str | None,
) -> dict[str, object]:
    frame = ImageFrame(sequence, timestamp_ms, width, height, pixels)
    file = f"frames/{sequence:06d}.png"
    data = frame.png_bytes()
    (stage / file).write_bytes(data)
    return {
        "sequence": sequence,
        "timestamp_ms": frame.timestamp_ms,
        "file": file,
        "sha256": _sha(data),
        "rgb_sha256": _sha(pixels),
        "source_name": source_name,
        "source_sha256": source_sha256,
        "width": width,
        "height": height,
    }


def _check_volume(frame_count: int, width: int, height: int, total_bytes: int) -> None:
    if width * height * frame_count > MAX_TOTAL_PIXELS or total_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("recording import exceeds its total pixel/encoded-byte budget")


def _directory_frames(source: Path, stage: Path, fps: float, max_frames: int) -> dict[str, object]:
    paths = sorted(
        (p for p in source.iterdir() if p.suffix.lower() == ".png"), key=lambda p: p.name
    )
    if not paths:
        raise ValueError("PNG directory contains no frames")
    frames: list[dict[str, object]] = []
    dimensions: tuple[int, int] | None = None
    total_bytes = 0
    for sequence, path in enumerate(paths[:max_frames]):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FRAME_BYTES:
            raise ValueError("PNG source must be a regular local file within the size limit")
        _name(path.name)
        data = path.read_bytes()
        width, height, pixels = _png(data)
        if dimensions is not None and dimensions != (width, height):
            raise ValueError("recording frame dimensions changed; split into separate sessions")
        dimensions = (width, height)
        _check_volume(sequence + 1, width, height, total_bytes)
        frame = _write_frame(
            stage, sequence, _timestamp(sequence, fps), width, height, pixels, path.name, _sha(data)
        )
        total_bytes += (stage / str(frame["file"])).stat().st_size
        _check_volume(sequence + 1, width, height, total_bytes)
        frames.append(frame)
    selected = [
        {"name": frame["source_name"], "sha256": frame["source_sha256"]} for frame in frames
    ]
    return {
        "frames": frames,
        "fps": fps,
        "timebase": "nominal_fps_user",
        "total_source_frames": len(paths),
        "truncated": len(paths) > len(frames),
        "completion_reason": "frame_limit" if len(paths) > len(frames) else "directory_complete",
        "source_sha256": _sha(canonical_json(selected).encode("utf-8")),
        "source_hash_basis": "selected_png_inputs",
    }


def _check_video(source: Path) -> None:
    if source.suffix.lower() != ".mp4" or not source.is_file() or source.is_symlink():
        raise ValueError("video source must be a regular local MP4 file")
    if source.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("MP4 exceeds the 2 GiB input limit")
    with source.open("rb") as stream:
        header = stream.read(12)
    if len(header) < 12 or header[4:8] != b"ftyp":
        raise ValueError("video must have an MP4 file-type header, not a playlist or URL")


def _video_frames(
    source: Path, stage: Path, fps: float | None, max_frames: int
) -> dict[str, object]:
    _check_video(source)
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise ValueError(
            "install the video extra (opencv-python-headless) for local MP4 decoding"
        ) from exc
    source_hash = _file_hash(source)
    frames: list[dict[str, object]] = []
    dimensions: tuple[int, int] | None = None
    total_bytes = 0
    # Explicit FFmpeg backend plus file/pipe-only transport disables decoder network protocols.
    # The lock limits this process-wide environment setting to one import at a time.
    with (
        _VIDEO_LOCK,
        patch.dict(os.environ, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "protocol_whitelist;file,pipe"}),
    ):
        capture = cv2.VideoCapture(str(source.resolve()), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise ValueError("local MP4 could not be opened by the FFmpeg backend")
            rate = _fps(fps if fps is not None else float(capture.get(cv2.CAP_PROP_FPS)))
            hinted = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_hint = (
                int(hinted)
                if math.isfinite(hinted) and hinted.is_integer() and hinted > 0
                else None
            )
            more = False
            for sequence in range(max_frames + 1):
                ok, bgr = capture.read()
                if not ok:
                    break
                if sequence == max_frames:
                    more = True
                    break
                if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                    raise ValueError("decoder did not produce a BGR frame")
                height, width = int(bgr.shape[0]), int(bgr.shape[1])
                if not (
                    1 <= width <= 8192
                    and 1 <= height <= 8192
                    and width * height <= MAX_FRAME_PIXELS
                ):
                    raise ValueError("video frame exceeds dimension/pixel limits")
                if dimensions is not None and dimensions != (width, height):
                    raise ValueError("video dimensions changed; split into separate sessions")
                dimensions = (width, height)
                _check_volume(sequence + 1, width, height, total_bytes)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = _write_frame(
                    stage,
                    sequence,
                    _timestamp(sequence, rate),
                    width,
                    height,
                    rgb.tobytes(),
                    f"video-frame-{sequence:06d}",
                    None,
                )
                total_bytes += (stage / str(frame["file"])).stat().st_size
                _check_volume(sequence + 1, width, height, total_bytes)
                frames.append(frame)
        finally:
            capture.release()
    if not frames:
        raise ValueError("MP4 produced no decodable frames")
    if _file_hash(source) != source_hash:
        raise ValueError("MP4 source changed during decoding")
    return {
        "frames": frames,
        "fps": rate,
        "timebase": "nominal_fps_user" if fps is not None else "nominal_fps_video_metadata",
        "total_source_frames": total_hint,
        "truncated": more,
        "completion_reason": "frame_limit" if more else "decoder_stopped",
        "source_sha256": source_hash,
        "source_hash_basis": "mp4_file",
    }


def _presentation_frames(source: Path, stage: Path, max_frames: int) -> dict[str, object]:
    _check_video(source)
    try:
        av = importlib.import_module("av")
    except ImportError as exc:
        raise ValueError("install the video-pts extra for MP4 presentation timestamps") from exc
    source_hash = _file_hash(source)
    frames: list[dict[str, object]] = []
    total_bytes = 0
    first_pts = previous = 0
    more = False
    dimensions = None
    with (
        source.open("rb") as data,
        av.open(
            data,
            mode="r",
            format="mov",
            options={"protocol_whitelist": "file,pipe", "enable_drefs": "0"},
        ) as container,
    ):
        if not container.streams.video:
            raise ValueError("MP4 has no video stream")
        stream = container.streams.video[0]
        clock = stream.time_base
        if not isinstance(clock, Fraction) or clock <= 0:
            raise ValueError("missing video stream time base")
        for i, decoded in enumerate(container.decode(stream)):
            if i == max_frames:
                more = True
                break
            if decoded.time_base != clock or type(decoded.pts) is not int:
                raise ValueError("video frame has missing or inconsistent presentation time")
            pts = _integer(decoded.pts, "PTS", -(2**63), 2**63 - 1)
            if i and pts <= previous:
                raise ValueError("presentation timestamps must increase strictly")
            if i == 0:
                first_pts = pts
                dimensions = (decoded.width, decoded.height)
            previous = pts
            width = _integer(decoded.width, "width", 1, 8192)
            height = _integer(decoded.height, "height", 1, 8192)
            if (width, height) != dimensions or width * height > MAX_FRAME_PIXELS:
                raise ValueError("video dimensions changed or exceed the pixel limit")
            _check_volume(i + 1, width, height, total_bytes)
            timestamp = round((pts - first_pts) * clock * 1000)
            row = _write_frame(
                stage,
                i,
                timestamp,
                width,
                height,
                decoded.to_image().tobytes(),
                f"video-frame-{i:06d}",
                None,
            )
            row["pts"] = pts
            total_bytes += (stage / str(row["file"])).stat().st_size
            _check_volume(i + 1, width, height, total_bytes)
            frames.append(row)
        clock_data = {
            "numerator": clock.numerator,
            "denominator": clock.denominator,
            "stream_index": stream.index,
        }
    if not frames or _file_hash(source) != source_hash:
        raise ValueError("MP4 has no frames or changed during decoding")
    return {
        "frames": frames,
        "fps": None,
        "timebase": "video_presentation_ticks",
        "video_clock": clock_data,
        "total_source_frames": None,
        "truncated": more,
        "completion_reason": "frame_limit" if more else "decoder_stopped",
        "source_sha256": source_hash,
        "source_hash_basis": "mp4_file",
    }


def import_recording(
    source: Path,
    output: Path,
    *,
    source_uri: str,
    permission: str,
    license: str,
    fps: float | None = None,
    max_frames: int = 300,
    presentation_time: bool = False,
) -> dict[str, object]:
    """Normalize a local PNG directory or MP4 into a hashed frame session.

    FPS determines a nominal constant-rate timeline. For exact MP4 timing,
    presentation_time binds the rational presentation clock to each decoded frame.
    This importer keeps the first max_frames frames and does not invent frames.
    """
    max_frames = _integer(max_frames, "max_frames", 1, 10000)
    rate = _fps(fps) if fps is not None else None
    validate_provenance(source_uri, permission, license)
    source = Path(source)
    if not source.exists() or source.is_symlink():
        raise ValueError("source must be an existing local path, not a URL or symlink")
    _name(source.name)
    directory = source.is_dir()
    if type(presentation_time) is not bool or (
        presentation_time and (directory or rate is not None)
    ):
        raise ValueError("presentation time requires an MP4 without an fps override")
    if directory and rate is None:
        raise ValueError("PNG directories require an explicit fps")
    with _new_output(output, source) as (stage, destination):
        (stage / "frames").mkdir()
        if presentation_time:
            imported = _presentation_frames(source, stage, max_frames)
        elif directory and rate is not None:
            imported = _directory_frames(source, stage, rate, max_frames)
        else:
            imported = _video_frames(source, stage, rate, max_frames)
        frames: Any = imported["frames"]
        manifest: dict[str, object] = {
            "schema_version": 2 if presentation_time else RECORDING_SCHEMA_VERSION,
            "kind": "offline_recording",
            "scope": "user_supplied_recording",
            "source_kind": "png_directory" if directory else "mp4",
            "source_uri": source_uri,
            "permission": permission,
            "license": license,
            "source_name": source.name,
            "frame_count": len(frames),
            "width": frames[0]["width"],
            "height": frames[0]["height"],
            "max_frames": max_frames,
            **imported,
        }
        manifest["session_id"] = _sha(canonical_json(manifest).encode("utf-8"))
        _write_json(stage / "manifest.json", manifest)
        summary = validate_recording(stage)
    return {**summary, "output": str(destination)}


def _validated_manifest(recording: Path) -> dict[str, Any]:
    root = Path(recording)
    manifest_file = root / "manifest.json"
    if (
        root.is_symlink()
        or manifest_file.is_symlink()
        or manifest_file.stat().st_size > 16 * 1024 * 1024
    ):
        raise ValueError("recording manifest must be a bounded regular local file")
    manifest = strict_json(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("recording manifest must be an object")
    exact = manifest.get("schema_version") == 2
    if set(manifest) != (_MANIFEST_KEYS | {"video_clock"} if exact else _MANIFEST_KEYS):
        raise ValueError("recording manifest has missing or unknown fields")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] not in (1, 2)
        or manifest["kind"] != "offline_recording"
        or manifest["scope"] != "user_supplied_recording"
    ):
        raise ValueError("unsupported recording schema/kind/scope")
    identity = {key: value for key, value in manifest.items() if key != "session_id"}
    if manifest["session_id"] != _sha(canonical_json(identity).encode("utf-8")):
        raise ValueError("recording manifest/session hash mismatch")
    validate_provenance(manifest["source_uri"], manifest["permission"], manifest["license"])
    _name(manifest["source_name"])
    clock = None
    rate = None
    if exact:
        spec = manifest["video_clock"]
        if (
            manifest["source_kind"] != "mp4"
            or manifest["timebase"] != "video_presentation_ticks"
            or manifest["fps"] is not None
            or not isinstance(spec, dict)
            or set(spec) != {"stream_index", "numerator", "denominator"}
        ):
            raise ValueError("invalid video presentation clock")
        _integer(spec["stream_index"], "video stream index", 0, 2**31 - 1)
        clock = Fraction(
            _integer(spec["numerator"], "clock numerator", 1, 2**31 - 1),
            _integer(spec["denominator"], "clock denominator", 1, 2**31 - 1),
        )
    else:
        rate = _fps(manifest["fps"])
    count = _integer(manifest["frame_count"], "frame_count", 1, 10000)
    maximum = _integer(manifest["max_frames"], "max_frames", 1, 10000)
    width = _integer(manifest["width"], "width", 1, 8192)
    height = _integer(manifest["height"], "height", 1, 8192)
    if count > maximum or width * height > MAX_FRAME_PIXELS:
        raise ValueError("recording exceeds frame count or pixel limits")
    if not _valid_hash(manifest["source_sha256"]) or type(manifest["truncated"]) is not bool:
        raise ValueError("invalid recording source hash/truncation flag")
    hinted = manifest["total_source_frames"]
    if hinted is not None:
        _integer(hinted, "total_source_frames", 1, 2**53)
    if not isinstance(manifest["timebase"], str):
        raise ValueError("invalid recording timebase")
    kind = manifest["source_kind"]
    if kind == "png_directory":
        if (
            manifest["source_hash_basis"] != "selected_png_inputs"
            or manifest["timebase"] != "nominal_fps_user"
            or hinted is None
            or hinted < count
            or manifest["truncated"] != (hinted > count)
            or manifest["completion_reason"]
            != ("frame_limit" if hinted > count else "directory_complete")
        ):
            raise ValueError("inconsistent PNG directory timeline/provenance")
    elif kind == "mp4":
        if (
            manifest["source_hash_basis"] != "mp4_file"
            or (
                not exact
                and manifest["timebase"] not in {"nominal_fps_user", "nominal_fps_video_metadata"}
            )
            or manifest["completion_reason"]
            != ("frame_limit" if manifest["truncated"] else "decoder_stopped")
        ):
            raise ValueError("inconsistent MP4 timeline/provenance")
    else:
        raise ValueError("unsupported recording source kind")
    frames = manifest["frames"]
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError("frame count does not match manifest")
    expected_paths: set[str] = set()
    source_names: set[str] = set()
    total_bytes = 0
    first_pts = previous = 0
    fields = _FRAME_KEYS | {"pts"} if exact else _FRAME_KEYS
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("invalid recording frame entry")
        if clock is not None:
            pts = _integer(item["pts"], "PTS", -(2**63), 2**63 - 1)
            if index == 0:
                first_pts = pts
            elif pts <= previous:
                raise ValueError("presentation timestamps must increase strictly")
            previous = pts
            timestamp = round((pts - first_pts) * clock * 1000)
        else:
            assert rate is not None
            timestamp = _timestamp(index, rate)
        if (
            type(item["sequence"]) is not int
            or item["sequence"] != index
            or type(item["timestamp_ms"]) is not int
            or item["timestamp_ms"] != timestamp
            or item["file"] != f"frames/{index:06d}.png"
            or type(item["width"]) is not int
            or item["width"] != width
            or type(item["height"]) is not int
            or item["height"] != height
        ):
            raise ValueError("frame sequence, timeline, dimensions, or path is inconsistent")
        _name(item["source_name"])
        if item["source_name"] in source_names:
            raise ValueError("duplicate source frame name")
        source_names.add(item["source_name"])
        if not _valid_hash(item["sha256"]) or not _valid_hash(item["rgb_sha256"]):
            raise ValueError("invalid frame hash")
        if (kind == "png_directory" and not _valid_hash(item["source_sha256"])) or (
            kind == "mp4" and item["source_sha256"] is not None
        ):
            raise ValueError("invalid source frame hash")
        frame = _load_frame(root, item)
        if (frame.width, frame.height) != (width, height):
            raise ValueError("decoded dimensions differ from the manifest")
        expected_paths.add(item["file"])
        total_bytes += (root / item["file"]).stat().st_size
    _check_volume(count, width, height, total_bytes)
    frame_directory = root / "frames"
    if frame_directory.is_symlink() or not frame_directory.is_dir():
        raise ValueError("frame directory must be a real local directory")
    actual_paths = {
        path.relative_to(root).as_posix() for path in frame_directory.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("recording has extra or missing frame files")
    if kind == "png_directory":
        selected = [
            {"name": frame["source_name"], "sha256": frame["source_sha256"]} for frame in frames
        ]
        if _sha(canonical_json(selected).encode("utf-8")) != manifest["source_sha256"]:
            raise ValueError("source PNG aggregate hash mismatch")
    return manifest


def _load_frame(root: Path, item: dict[str, Any]) -> ImageFrame:
    path = root / item["file"]
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or path.stat().st_size > MAX_FRAME_BYTES
    ):
        raise ValueError("recording frame must be a bounded local regular file")
    data = path.read_bytes()
    if _sha(data) != item["sha256"]:
        raise ValueError("recording frame SHA256 mismatch")
    width, height, pixels = _png(data)
    if _sha(pixels) != item["rgb_sha256"]:
        raise ValueError("recording RGB hash mismatch")
    return ImageFrame(item["sequence"], item["timestamp_ms"], width, height, pixels)


def _summary(manifest: dict[str, Any]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "session_id",
            "scope",
            "source_kind",
            "frame_count",
            "fps",
            "timebase",
            "width",
            "height",
            "truncated",
            "completion_reason",
        )
    } | {
        "validated": True,
        "timing_accuracy": "source_pts" if manifest["schema_version"] == 2 else "nominal_fps_only",
    }


def validate_recording(recording: Path) -> dict[str, object]:
    return _summary(_validated_manifest(recording))


def iter_recording(recording: Path) -> Iterator[ImageFrame]:
    manifest = _validated_manifest(recording)
    for item in manifest["frames"]:
        # Recheck each file when consumed; no state or annotation travels with pixels.
        yield _load_frame(recording, item)


def observe_recording(
    recording: Path, output: Path, detector: Detector | None = None
) -> dict[str, object]:
    """Replay fixed frames into a detector; this cannot verify action outcomes."""
    root = Path(recording)
    manifest = _validated_manifest(root)
    baseline = PixelPerception() if detector is None else None
    source_id = baseline.source_id if baseline is not None else type(detector).__name__
    detector_metadata: dict[str, object] = {"implementation": source_id}
    model_manifest = getattr(detector, "manifest", None)
    if isinstance(model_manifest, ModelManifest):
        detector_metadata["model_manifest"] = asdict(model_manifest)
        detector_metadata["confidence_threshold"] = getattr(detector, "confidence_threshold", None)
        detector_metadata["iou_threshold"] = getattr(detector, "iou_threshold", None)
    statuses: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    with_detections = 0
    with _new_output(output, root) as (stage, destination):
        with (stage / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for item in manifest["frames"]:
                frame = _load_frame(root, item)
                if baseline is not None:
                    observation = baseline.observe(frame)
                else:
                    assert detector is not None
                    detections = detector.detect(frame)
                    if not isinstance(detections, tuple) or any(
                        not isinstance(value, Detection) for value in detections
                    ):
                        raise ValueError("detector must return immutable Detection values")
                    if any(d.bbox[2] > frame.width or d.bbox[3] > frame.height for d in detections):
                        raise ValueError("detector produced boxes outside the frame")
                    observation = VisualObservation(
                        frame.sequence,
                        frame.timestamp_ms,
                        Status.UNKNOWN,
                        0.0,
                        detections,
                        None,
                        None,
                        None,
                        source_id=source_id,
                    )
                statuses[observation.status.value] += 1
                classes.update(detection.class_id for detection in observation.detections)
                with_detections += bool(observation.detections)
                stream.write(
                    canonical_json(
                        {
                            "schema_version": 1,
                            "session_id": manifest["session_id"],
                            "frame_sha256": item["sha256"],
                            "observation": asdict(observation),
                        }
                    )
                    + "\n"
                )
        report: dict[str, object] = {
            "schema_version": 1,
            "session_id": manifest["session_id"],
            "mode": "perception_only_replay",
            "detector": source_id,
            "detector_metadata": detector_metadata,
            "observations_sha256": _file_hash(stage / "observations.jsonl"),
            "frames_processed": manifest["frame_count"],
            "frames_with_detections": with_detections,
            "status_counts": dict(sorted(statuses.items())),
            "detections_by_class": dict(sorted(classes.items())),
            "accuracy_metrics_available": False,
            "effect_success_rate": None,
            "action_execution": "none",
            "timing_accuracy": "source_pts"
            if manifest["schema_version"] == 2
            else "nominal_fps_only",
            "limitations": [
                "fixed recordings cannot verify counterfactual action outcomes",
                "no ground-truth labels: detection accuracy is not measured",
                "fixture palette/HUD rules do not establish real-game state or rules",
            ],
        }
        _write_json(stage / "report.json", report)
    return {**report, "output": str(destination)}
