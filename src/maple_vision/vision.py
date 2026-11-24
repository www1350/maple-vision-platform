"""Pixel-only fixture baseline, deliberately independent of simulator state.

Exact-color detection is a reproducible plumbing baseline, not a learned YOLO
model and not evidence of perception quality on real game recordings.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .contracts import Detection, Status, confidence_value, nonnegative_int
from .scene import (
    BAR_BG,
    BAR_WIDTH,
    BAR_X,
    DIALOGUE_COLOR,
    HP_COLOR,
    HP_Y,
    HUD_BG,
    MAP_MARKERS,
    MP_COLOR,
    MP_Y,
    PALETTE,
    ImageFrame,
)


class Detector(Protocol):
    def detect(self, frame: ImageFrame) -> tuple[Detection, ...]: ...


@dataclass(frozen=True)
class VisualObservation:
    sequence: int
    timestamp_ms: int
    status: Status
    confidence: float
    detections: tuple[Detection, ...]
    hp_ratio: float | None
    mp_ratio: float | None
    map_id: str | None
    dialogue_visible: bool = False
    source_id: str = "synthetic-palette-v2"

    def __post_init__(self) -> None:
        nonnegative_int(self.sequence, "sequence")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        confidence_value(self.confidence)
        if not isinstance(self.status, Status):
            raise ValueError("status must be a Status value")
        if not isinstance(self.detections, tuple) or any(
            not isinstance(detection, Detection) for detection in self.detections
        ):
            raise ValueError("detections must be immutable Detection values")
        for ratio in (self.hp_ratio, self.mp_ratio):
            if ratio is not None and (
                isinstance(ratio, bool)
                or not isinstance(ratio, (float, int))
                or not isfinite(ratio)
                or not 0 <= ratio <= 1
            ):
                raise ValueError("resource ratios must be finite in [0, 1] or unknown")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be nonempty")
        if self.map_id is not None and (not isinstance(self.map_id, str) or not self.map_id):
            raise ValueError("map_id must be a nonempty string or unknown")
        if type(self.dialogue_visible) is not bool:
            raise ValueError("dialogue_visible must be boolean")


def _components(mask: NDArray[np.bool_]) -> tuple[tuple[float, float, float, float], ...]:
    """4-connected components using row runs; avoid one Python node per pixel."""
    parents: list[int] = []
    boxes: list[list[int]] = []
    areas: list[int] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    previous: list[tuple[int, int, int]] = []
    for y in np.flatnonzero(np.any(mask, axis=1)):
        yi = int(y)
        if previous and boxes[find(previous[0][2])][3] != yi:
            previous = []
        row = mask[yi]
        edges = np.flatnonzero(row[1:] != row[:-1]) + 1
        if row[0]:
            edges = np.concatenate(([0], edges))
        if row[-1]:
            edges = np.concatenate((edges, [len(row)]))
        current: list[tuple[int, int, int]] = []
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            x1, x2 = int(start), int(end)
            neighbors = {find(label) for left, right, label in previous if left < x2 and right > x1}
            if neighbors:
                root = min(neighbors)
                for other in neighbors - {root}:
                    parents[other] = root
                    boxes[root] = [
                        min(boxes[root][0], boxes[other][0]),
                        min(boxes[root][1], boxes[other][1]),
                        max(boxes[root][2], boxes[other][2]),
                        max(boxes[root][3], boxes[other][3]),
                    ]
                    areas[root] += areas[other]
                boxes[root] = [
                    min(boxes[root][0], x1),
                    boxes[root][1],
                    max(boxes[root][2], x2),
                    yi + 1,
                ]
                areas[root] += x2 - x1
            else:
                root = len(parents)
                parents.append(root)
                boxes.append([x1, yi, x2, yi + 1])
                areas.append(x2 - x1)
            current.append((x1, x2, root))
        previous = current
    return tuple(
        (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for i, box in enumerate(boxes)
        if parents[i] == i and areas[i] >= 3
    )


def _merge_platforms(
    boxes: tuple[tuple[float, float, float, float], ...],
    foreground: NDArray[np.bool_],
) -> tuple[tuple[float, float, float, float], ...]:
    """Join aligned visible segments only when foreground explains every gap pixel.

    No simulator geometry, object IDs, or training labels are consulted. A gap
    containing even one background/unknown pixel remains a gap. This is a
    conservative synthetic-palette inference, not a guarantee of hidden terrain.
    """
    rows: dict[tuple[float, float], list[tuple[float, float, float, float]]] = {}
    for box in boxes:
        rows.setdefault((box[1], box[3]), []).append(box)
    merged: list[tuple[float, float, float, float]] = []
    for (top, bottom), segments in rows.items():
        segments.sort(key=lambda box: box[0])
        current = segments[0]
        for following in segments[1:]:
            gap_left, gap_right = int(current[2]), int(following[0])
            explained = gap_right > gap_left and bool(
                np.all(foreground[int(top) : int(bottom), gap_left:gap_right])
            )
            if explained:
                current = (current[0], top, following[2], bottom)
            else:
                merged.append(current)
                current = following
        merged.append(current)
    return tuple(merged)


class PixelPerception:
    source_id = "synthetic-palette-v2"

    def detect(self, frame: ImageFrame) -> tuple[Detection, ...]:
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        # Pack exact RGB once instead of reducing three channels for every class.
        packed = (
            (rgb[:, :, 0].astype(np.uint32) << 16)
            | (rgb[:, :, 1].astype(np.uint32) << 8)
            | rgb[:, :, 2].astype(np.uint32)
        )
        components: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
        foreground = np.zeros((frame.height, frame.width), dtype=np.bool_)
        for class_id, color in PALETTE.items():
            mask = packed == ((color[0] << 16) | (color[1] << 8) | color[2])
            components[class_id] = _components(mask)
            if class_id != "platform":
                foreground |= mask
        components["platform"] = _merge_platforms(components["platform"], foreground)
        return tuple(
            Detection(class_id, box, 1.0) for class_id, boxes in components.items() for box in boxes
        )

    def observe(self, frame: ImageFrame) -> VisualObservation:
        detections = self.detect(frame)
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        map_id: str | None = None
        if frame.width >= 160 and frame.height >= 80:
            marker = rgb[6:18, frame.width - 20 : frame.width - 8]
            for name, color in MAP_MARKERS.items():
                if np.all(marker == color):
                    map_id = name
        hp_ratio = self._bar(rgb, HP_Y, HP_COLOR)
        mp_ratio = self._bar(rgb, MP_Y, MP_COLOR)
        known = (
            sum(item.class_id == "player" for item in detections) == 1
            and map_id is not None
            and hp_ratio is not None
            and mp_ratio is not None
        )
        dialogue = False
        if frame.width >= 160 and frame.height >= 80:
            dialogue = bool(
                np.all(rgb[34, frame.width // 4 : frame.width * 3 // 4] == DIALOGUE_COLOR)
            )
        return VisualObservation(
            frame.sequence,
            frame.timestamp_ms,
            Status.KNOWN if known else Status.UNKNOWN,
            1.0 if known else 0.0,
            detections,
            hp_ratio,
            mp_ratio,
            map_id,
            dialogue,
        )

    @staticmethod
    def _bar(rgb: NDArray[np.uint8], y: int, color: tuple[int, int, int]) -> float | None:
        if rgb.shape[0] <= y + 4 or rgb.shape[1] < BAR_X + BAR_WIDTH:
            return None
        if not np.all(rgb[3, 3] == HUD_BG):
            return None
        line = rgb[y + 2, BAR_X : BAR_X + BAR_WIDTH]
        foreground = np.all(line == color, axis=1)
        background = np.all(line == BAR_BG, axis=1)
        if not np.all(foreground | background):
            return None
        count = int(np.count_nonzero(foreground))
        if not np.all(foreground[:count]) or np.any(foreground[count:]):
            return None
        return count / BAR_WIDTH
