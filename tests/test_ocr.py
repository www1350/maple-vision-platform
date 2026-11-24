import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from maple_vision.ocr import TesseractOCR
from maple_vision.scene import ImageFrame

HEADER = (
    b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    b"left\ttop\twidth\theight\tconf\ttext\n"
)
WORD = b"5\t1\t1\t1\t1\t1\t2\t3\t10\t5\t87.5\tMAPLE\n"


@pytest.fixture
def fake_tesseract(monkeypatch):
    calls = []
    state = {"tsv": HEADER + WORD, "returncode": 0, "timeout": False}

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if "--version" in arguments:
            stdout = b"tesseract 5.0.0\n"
        elif "--list-langs" in arguments:
            stdout = b'List of available languages in "/local" (2):\neng\nosd\n'
        else:
            if state["timeout"]:
                raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])
            return subprocess.CompletedProcess(
                arguments, state["returncode"], state["tsv"], b"fixture error"
            )
        return subprocess.CompletedProcess(arguments, 0, stdout, b"")

    monkeypatch.setattr(subprocess, "run", run)
    return calls, state


@pytest.fixture
def frame():
    return ImageFrame(7, 350, 100, 60, b"\xff\xff\xff" * 100 * 60)


def adapter():
    return TesseractOCR(Path(sys.executable).resolve(), "eng", timeout_seconds=2)


def test_local_command_crop_and_coordinates(fake_tesseract, frame):
    calls, _ = fake_tesseract
    ocr = adapter()
    result = ocr.recognize(frame, (20, 10, 70, 40))
    assert result.text == "MAPLE"
    assert result.words[0].bbox == (22, 13, 32, 18)
    assert result.words[0].confidence == pytest.approx(0.875)
    assert result.sequence == 7 and result.timestamp_ms == 350
    arguments, options = calls[-1]
    assert arguments[0] == str(Path(sys.executable).resolve())
    assert arguments[1:3] == ["stdin", "stdout"]
    assert arguments[-1] == "tsv"
    assert options["shell"] is False and options["timeout"] == 2
    with Image.open(io.BytesIO(options["input"])) as image:
        assert image.size == (50, 30)
    assert ocr.describe()["available_languages"] == ["eng", "osd"]


@pytest.mark.parametrize("language", ["eng;echo", "../eng", "-x", "eng+", "", True])
def test_language_arguments_cannot_inject_command_options(language):
    with pytest.raises(ValueError, match="language"):
        TesseractOCR(Path(sys.executable).resolve(), language)


def test_missing_language_does_not_download(fake_tesseract):
    calls, _ = fake_tesseract
    with pytest.raises(ValueError, match="not installed"):
        TesseractOCR(Path(sys.executable).resolve(), "chi_sim")
    assert len(calls) == 2


def test_missing_binary_and_relative_override_rejected(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="not installed"):
        TesseractOCR()
    with pytest.raises(ValueError, match="absolute"):
        TesseractOCR("tesseract")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 31},
        {"timeout_seconds": True},
        {"timeout_seconds": float("nan")},
        {"psm": 2},
        {"psm": True},
        {"psm": 14},
    ],
)
def test_invalid_runtime_bounds(kwargs):
    with pytest.raises(ValueError):
        TesseractOCR(Path(sys.executable).resolve(), **kwargs)


@pytest.mark.parametrize(
    "crop", [(-1, 0, 10, 10), (0, 0, 101, 10), (10, 0, 10, 10), (0, 0, True, 10), [0, 0, 10, 10]]
)
def test_invalid_crop_does_not_start_ocr(fake_tesseract, frame, crop):
    calls, _ = fake_tesseract
    ocr = adapter()
    with pytest.raises(ValueError, match="crop"):
        ocr.recognize(frame, crop)
    assert len(calls) == 2


def test_pixel_budget_applies_to_crop_not_source(fake_tesseract, frame, monkeypatch):
    import maple_vision.ocr as module

    monkeypatch.setattr(module, "MAX_CROP_PIXELS", 100)
    ocr = adapter()
    with pytest.raises(ValueError, match="megapixels"):
        ocr.recognize(frame)


def test_timeout_and_process_failure_are_errors(fake_tesseract, frame):
    _, state = fake_tesseract
    ocr = adapter()
    state["timeout"] = True
    with pytest.raises(ValueError, match="timed out"):
        ocr.recognize(frame)
    state["timeout"] = False
    state["returncode"] = 1
    with pytest.raises(ValueError, match="failed"):
        ocr.recognize(frame)


@pytest.mark.parametrize(
    "tsv",
    [
        b"bad columns\n",
        HEADER + WORD.replace(b"87.5", b"nan"),
        HEADER + WORD.replace(b"87.5", b"101"),
        HEADER + WORD.replace(b"\t2\t3\t10\t5\t", b"\t95\t3\t10\t5\t"),
        HEADER + WORD.replace(b"MAPLE", b"\xff"),
    ],
)
def test_malformed_tsv_cannot_become_success(fake_tesseract, frame, tsv):
    _, state = fake_tesseract
    state["tsv"] = tsv
    with pytest.raises(ValueError):
        adapter().recognize(frame)


def test_empty_ocr_result_is_explicit(fake_tesseract, frame):
    _, state = fake_tesseract
    state["tsv"] = HEADER
    result = adapter().recognize(frame)
    assert result.words == () and result.text == ""


def test_tessdata_dir_is_explicit_local_argument(fake_tesseract, tmp_path, frame):
    calls, _ = fake_tesseract
    ocr = TesseractOCR(Path(sys.executable).resolve(), tessdata_dir=tmp_path)
    ocr.recognize(frame)
    assert calls[-1][0][-3:] == ["--tessdata-dir", str(tmp_path.resolve()), "tsv"]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="optional local Tesseract binary")
def test_real_authored_text_smoke():
    image = Image.new("RGB", (500, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "MAPLE 123", fill="black", font=ImageFont.load_default(size=48))
    frame = ImageFrame(0, 0, image.width, image.height, image.tobytes())
    ocr = TesseractOCR(language="eng", timeout_seconds=5, psm=6)
    result = ocr.recognize(frame)
    assert "MAPLE" in result.text and "123" in result.text
    assert all(0 <= word.confidence <= 1 for word in result.words)
    assert all(0 <= word.bbox[0] < word.bbox[2] <= frame.width for word in result.words)
