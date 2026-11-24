"""Bounded local Tesseract CLI adapter, without downloads or language-model calls."""

from __future__ import annotations

import csv
import io
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .scene import ImageFrame

MAX_CROP_PIXELS = 4 * 1024 * 1024
MAX_TSV_BYTES = 8 * 1024 * 1024
MAX_WORDS = 10000
_LANGUAGE = re.compile(r"^[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*$")
_TSV_FIELDS = {"level", "left", "top", "width", "height", "conf", "text"}


@dataclass(frozen=True)
class OCRWord:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    sequence: int
    timestamp_ms: int
    language: str
    source_id: str
    crop: tuple[int, int, int, int]
    words: tuple[OCRWord, ...]
    text: str


def _crop(frame: ImageFrame, crop: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    value = (0, 0, frame.width, frame.height) if crop is None else crop
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("crop must be an immutable four-integer xyxy tuple")
    x1, y1, x2, y2 = value
    if not 0 <= x1 < x2 <= frame.width or not 0 <= y1 < y2 <= frame.height:
        raise ValueError("crop must have positive area inside the frame")
    if (x2 - x1) * (y2 - y1) > MAX_CROP_PIXELS:
        raise ValueError("OCR crop exceeds four megapixels")
    return value


def _image_input(frame: ImageFrame, crop: tuple[int, int, int, int]) -> bytes:
    x1, y1, x2, y2 = crop
    # Slice only the requested rows: avoid allocating another full-size source frame.
    pixels = b"".join(
        frame.pixels[(y * frame.width + x1) * 3 : (y * frame.width + x2) * 3] for y in range(y1, y2)
    )
    image = Image.frombytes("RGB", (x2 - x1, y2 - y1), pixels)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _run(arguments: list[str], *, timeout: float, data: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            arguments,
            input=data,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"local OCR command timed out after {timeout:g} seconds") from exc
    except OSError as exc:
        raise ValueError(f"local OCR command could not execute: {exc}") from exc
    if len(completed.stdout) > MAX_TSV_BYTES or len(completed.stderr) > MAX_TSV_BYTES:
        raise ValueError("local OCR output exceeds the 8 MiB limit")
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()[:500]
        raise ValueError(f"local OCR command failed ({completed.returncode}): {message}")
    return bytes(completed.stdout)


def _parse_tsv(data: bytes, crop: tuple[int, int, int, int]) -> tuple[OCRWord, ...]:
    if len(data) > MAX_TSV_BYTES:
        raise ValueError("OCR TSV exceeds the size limit")
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("OCR TSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
    if reader.fieldnames is None or not _TSV_FIELDS <= set(reader.fieldnames):
        raise ValueError("OCR TSV does not have the expected columns")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("OCR TSV has duplicate columns")
    x0, y0, x_end, y_end = crop
    width, height = x_end - x0, y_end - y0
    words: list[OCRWord] = []
    try:
        for row in reader:
            if None in row or any(row.get(field) is None for field in _TSV_FIELDS):
                raise ValueError("malformed OCR TSV row")
            if row["level"] not in {"1", "2", "3", "4", "5"}:
                raise ValueError("invalid OCR hierarchy level")
            if row["level"] != "5":
                continue
            text = row["text"].strip()
            confidence = float(row["conf"])
            if not math.isfinite(confidence) or not -1 <= confidence <= 100:
                raise ValueError("OCR confidence is not finite in [-1, 100]")
            if not text or confidence < 0:
                continue
            left, top, box_width, box_height = (
                int(row[key]) for key in ("left", "top", "width", "height")
            )
            if not (
                0 <= left < left + box_width <= width and 0 <= top < top + box_height <= height
            ):
                raise ValueError("OCR word box lies outside the crop")
            if len(text) > 4096 or any(ord(character) < 32 for character in text):
                raise ValueError("OCR word text exceeds limits or contains control characters")
            words.append(
                OCRWord(
                    text,
                    (left + x0, top + y0, left + box_width + x0, top + box_height + y0),
                    confidence / 100,
                )
            )
            if len(words) > MAX_WORDS:
                raise ValueError("OCR result exceeds the word limit")
    except (csv.Error, TypeError) as exc:
        raise ValueError("malformed OCR TSV") from exc
    return tuple(words)


class TesseractOCR:
    """Explicit local binary and installed languages; no automatic installation.

    Confidence values are Tesseract's scores divided by 100, not calibrated
    probabilities. The adapter returns text candidates, not verified game facts.
    """

    def __init__(
        self,
        binary: str | Path | None = None,
        language: str = "eng",
        *,
        tessdata_dir: Path | None = None,
        timeout_seconds: float = 5,
        psm: int = 6,
    ) -> None:
        if (
            not isinstance(language, str)
            or len(language) > 128
            or not _LANGUAGE.fullmatch(language)
        ):
            raise ValueError(
                "language must be installed language identifiers such as eng or eng+chi_sim"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0.1 <= timeout_seconds <= 30
        ):
            raise ValueError("timeout_seconds must be finite in [0.1, 30]")
        if type(psm) is not int or not 3 <= psm <= 13:
            raise ValueError("psm must be an integer in [3, 13]")
        if binary is None:
            found = shutil.which("tesseract")
            if found is None:
                raise ValueError("Tesseract is not installed; supply an existing local binary")
            resolved = Path(found).resolve()
        else:
            supplied = Path(binary)
            if not supplied.is_absolute():
                raise ValueError("explicit Tesseract binary must be an absolute path")
            resolved = supplied.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError("Tesseract binary is not an executable local file")
        if os.name == "nt" and resolved.suffix.lower() != ".exe":
            raise ValueError("Windows Tesseract binary must be an .exe, not a command script")
        self.binary = resolved
        self.language = language
        self.timeout_seconds = float(timeout_seconds)
        self.psm = psm
        self.tessdata_dir = Path(tessdata_dir).resolve() if tessdata_dir is not None else None
        if self.tessdata_dir is not None and not self.tessdata_dir.is_dir():
            raise ValueError("tessdata_dir must be an existing local directory")
        version = _run([str(resolved), "--version"], timeout=self.timeout_seconds).decode("utf-8")
        first_line = version.splitlines()[0] if version else ""
        if not first_line.lower().startswith("tesseract "):
            raise ValueError("resolved binary did not identify itself as Tesseract")
        self.version = first_line
        arguments = [str(resolved), "--list-langs", *self._data_args()]
        languages = _run(arguments, timeout=self.timeout_seconds).decode("utf-8")
        self.available_languages = tuple(
            sorted(
                {
                    line.strip()
                    for line in languages.splitlines()
                    if re.fullmatch(r"[A-Za-z0-9_]+", line.strip())
                }
            )
        )
        missing = set(language.split("+")) - set(self.available_languages)
        if missing:
            raise ValueError(
                f"requested OCR languages are not installed: {', '.join(sorted(missing))}"
            )

    def _data_args(self) -> list[str]:
        return ["--tessdata-dir", str(self.tessdata_dir)] if self.tessdata_dir is not None else []

    def recognize(
        self,
        frame: ImageFrame,
        crop: tuple[int, int, int, int] | None = None,
    ) -> OCRResult:
        area = _crop(frame, crop)
        arguments = [
            str(self.binary),
            "stdin",
            "stdout",
            "-l",
            self.language,
            "--psm",
            str(self.psm),
            *self._data_args(),
            "tsv",
        ]
        data = _run(arguments, timeout=self.timeout_seconds, data=_image_input(frame, area))
        words = _parse_tsv(data, area)
        return OCRResult(
            frame.sequence,
            frame.timestamp_ms,
            self.language,
            self.version,
            area,
            words,
            " ".join(word.text for word in words),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "binary": str(self.binary),
            "version": self.version,
            "language": self.language,
            "available_languages": list(self.available_languages),
            "psm": self.psm,
            "timeout_seconds": self.timeout_seconds,
            "tessdata_dir": str(self.tessdata_dir) if self.tessdata_dir else None,
        }
