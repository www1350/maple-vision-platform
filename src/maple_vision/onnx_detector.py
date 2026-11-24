"""Explicit, hash-checked YOLO ONNX inference on RGB frames, without network access.

Supported outputs are declared in the manifest; task/layout guessing is deliberately
avoided. This adapter is for detection exports, not segmentation or pose exports.
"""

import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .contracts import Detection
from .scene import ImageFrame


@dataclass(frozen=True)
class ModelManifest:
    model_sha256: str
    class_names: tuple[str, ...]
    input_width: int
    input_height: int
    output_format: str
    scope: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported model manifest version")
        if len(self.model_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.model_sha256
        ):
            raise ValueError("model_sha256 must be a lowercase SHA256")
        if (
            not self.class_names
            or any(not isinstance(c, str) or not c.strip() for c in self.class_names)
            or len(set(self.class_names)) != len(self.class_names)
        ):
            raise ValueError("class_names must be distinct nonempty strings")
        for size in (self.input_width, self.input_height):
            if type(size) is not int or not 32 <= size <= 4096:
                raise ValueError("model dimensions must be integers in [32, 4096]")
        if self.output_format not in {"raw_xywh", "end2end_xyxy"}:
            raise ValueError("unsupported output_format")
        if self.scope not in {"synthetic_fixture", "authorized_recordings"}:
            raise ValueError("model scope must describe its evaluation domain")

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = {
            "schema_version",
            "model_sha256",
            "class_names",
            "input_width",
            "input_height",
            "output_format",
            "scope",
        }
        if (
            not isinstance(data, dict)
            or set(data) != keys
            or not isinstance(data["class_names"], list)
        ):
            raise ValueError("model manifest has missing or unknown fields")
        try:
            return cls(**{**data, "class_names": tuple(data["class_names"])})
        except (TypeError, AttributeError) as exc:
            raise ValueError("invalid model manifest types") from exc


@dataclass(frozen=True)
class Letterbox:
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


def preprocess(frame: ImageFrame, width: int, height: int) -> tuple[NDArray[np.float32], Letterbox]:
    """RGB -> padded float32 NCHW. Track actual rounded resize scales for inversion."""
    scale = min(width / frame.width, height / frame.height)
    resized_width = max(1, round(frame.width * scale))
    resized_height = max(1, round(frame.height * scale))
    pad_x, pad_y = (width - resized_width) // 2, (height - resized_height) // 2
    original = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels)
    resized = original.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (width, height), (114, 114, 114))
    padded.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    transform = Letterbox(
        resized_width / frame.width,
        resized_height / frame.height,
        pad_x,
        pad_y,
        frame.width,
        frame.height,
    )
    return np.ascontiguousarray(tensor), transform


def intersection_over_union(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def decode(
    output: NDArray[Any],
    manifest: ModelManifest,
    transform: Letterbox,
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> tuple[Detection, ...]:
    for threshold in (confidence_threshold, iou_threshold):
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("thresholds must be finite in [0, 1]")
    if type(max_detections) is not int or not 1 <= max_detections <= 1000:
        raise ValueError("max_detections must be in [1, 1000]")
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("expected a single batch detection tensor")
    raw = manifest.output_format == "raw_xywh"
    if raw:
        if output.shape[1] != 4 + len(manifest.class_names):
            raise ValueError("raw output must be [1, 4 + classes, predictions]")
        rows = output[0].T
    else:
        if output.shape[2] != 6:
            raise ValueError("end-to-end output must be [1, predictions, 6]")
        rows = output[0]
    # Bound work before quadratic NMS. A malformed export must not exhaust the runner.
    if len(rows) > 100_000:
        raise ValueError("too many model predictions")
    # Reject empty/invalid candidates in vector operations before Python NMS.
    # A 320px raw export has thousands of anchors but normally only a few objects.
    valid = np.isfinite(rows).all(axis=1)
    if raw:
        scores = rows[:, 4:]
        valid &= (scores >= 0).all(axis=1) & (scores <= 1).all(axis=1)
        valid &= (rows[:, 2] > 0) & (rows[:, 3] > 0)
        valid &= scores.max(axis=1) >= confidence_threshold
    else:
        valid &= (rows[:, 4] >= confidence_threshold) & (rows[:, 4] <= 1)
    rows = rows[valid]
    candidates: list[Detection] = []
    for row in rows:
        if not np.isfinite(row).all():
            continue
        if raw:
            scores = row[4:]
            if (scores < 0).any() or (scores > 1).any():
                continue
            class_index = int(np.argmax(scores))
            confidence = float(scores[class_index])
            cx, cy, width, height = (float(value) for value in row[:4])
            if width <= 0 or height <= 0:
                continue
            x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        else:
            x1, y1, x2, y2, confidence, index = (float(value) for value in row)
            if not index.is_integer() or not 0 <= index < len(manifest.class_names):
                continue
            class_index = int(index)
        if not confidence_threshold <= confidence <= 1:
            continue
        x1 = max(0.0, min(transform.original_width, (x1 - transform.pad_x) / transform.scale_x))
        x2 = max(0.0, min(transform.original_width, (x2 - transform.pad_x) / transform.scale_x))
        y1 = max(0.0, min(transform.original_height, (y1 - transform.pad_y) / transform.scale_y))
        y2 = max(0.0, min(transform.original_height, (y2 - transform.pad_y) / transform.scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            Detection(manifest.class_names[class_index], (x1, y1, x2, y2), confidence)
        )
    candidates.sort(key=lambda item: (-item.confidence, item.class_id, item.bbox))
    selected: list[Detection] = []
    # End-to-end exports have already selected/NMS'ed their detections.
    for candidate in candidates[:3000]:
        if raw and any(
            candidate.class_id == previous.class_id
            and intersection_over_union(candidate, previous) > iou_threshold
            for previous in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_detections:
            break
    return tuple(selected)


class OnnxDetector:
    """CPU inference with explicit metadata and no automatic weights download."""

    @property
    def loaded_model_sha256(self) -> str:
        """Identity of the exact bytes supplied to the native inference session."""
        return self._loaded_model_sha256

    @property
    def runtime_settings(self) -> dict[str, object]:
        """Read actual native session options without changing the inference session."""
        options = self._session.get_session_options()
        return {
            "providers": list(self._session.get_providers()),
            "intra_op_num_threads": options.intra_op_num_threads,
            "inter_op_num_threads": options.inter_op_num_threads,
            "execution_mode": str(options.execution_mode),
        }

    def __init__(
        self,
        model: Path,
        manifest: Path,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.manifest = ModelManifest.load(manifest)
        model_bytes = model.read_bytes()
        loaded_hash = hashlib.sha256(model_bytes).hexdigest()
        if loaded_hash != self.manifest.model_sha256:
            raise ValueError("model checksum mismatch")
        self._loaded_model_sha256 = loaded_hash
        # Validate thresholds even before loading an optional native dependency.
        for threshold in (confidence_threshold, iou_threshold):
            if (
                isinstance(threshold, bool)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("thresholds must be in [0, 1]")
        # Set before import: calling only the API can miss initialization events.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        try:
            runtime = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ValueError("install maple-vision-platform[inference] for ONNX inference") from exc
        runtime.disable_telemetry_events()
        options = runtime.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = runtime.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("model requires exactly one float32 NCHW input")
        shape = inputs[0].shape
        expected = [1, 3, self.manifest.input_height, self.manifest.input_width]
        if len(shape) != 4 or any(
            isinstance(actual, int) and actual != target
            for actual, target in zip(shape, expected, strict=True)
        ):
            raise ValueError("model input shape does not match manifest")
        if len(self._session.get_outputs()) != 1:
            raise ValueError("only single-output detection exports are supported")
        self._input_name = inputs[0].name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: ImageFrame) -> tuple[Detection, ...]:
        tensor, transform = preprocess(frame, self.manifest.input_width, self.manifest.input_height)
        output = np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        return decode(
            output,
            self.manifest,
            transform,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
