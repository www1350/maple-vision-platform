from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from maple_vision.contracts import Status
from maple_vision.scene import (
    BAR_BG,
    DIALOGUE_COLOR,
    HP_COLOR,
    HUD_BG,
    MAP_MARKERS,
    MP_COLOR,
    PALETTE,
    ImageFrame,
)
from maple_vision.vision import PixelPerception, VisualObservation, _components


def _image() -> Image.Image:
    # Construct pixels directly, without a simulator or a renderer implementation.
    image = Image.new("RGB", (320, 192), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 33, 121), fill=PALETTE["player"])
    draw.rectangle((100, 106, 117, 121), fill=PALETTE["monster"])
    draw.rectangle((150, 106, 167, 121), fill=PALETTE["monster"])
    draw.rectangle((3, 3, 115, 26), fill=HUD_BG)
    draw.rectangle((8, 7, 107, 11), fill=BAR_BG)
    draw.rectangle((8, 7, 57, 11), fill=HP_COLOR)
    draw.rectangle((8, 15, 107, 19), fill=MP_COLOR)
    draw.rectangle((300, 6, 311, 17), fill=MAP_MARKERS["cavern"])
    return image


def _frame(image: Image.Image) -> ImageFrame:
    return ImageFrame(9, 450, image.width, image.height, image.tobytes())


def test_pixel_detector_observes_disconnected_instances_and_resources() -> None:
    result = PixelPerception().observe(_frame(_image()))
    assert result.status is Status.KNOWN
    assert result.map_id == "cavern"
    assert (result.hp_ratio, result.mp_ratio) == (0.5, 1.0)
    assert result.sequence == 9 and result.timestamp_ms == 450
    assert {d.bbox for d in result.detections if d.class_id == "monster"} == {
        (100, 106, 118, 122),
        (150, 106, 168, 122),
    }


def test_blank_frame_is_unknown_and_never_invents_resources() -> None:
    result = PixelPerception().observe(ImageFrame(0, 0, 320, 192, bytes(320 * 192 * 3)))
    assert result.status is Status.UNKNOWN
    assert result.confidence == 0
    assert result.detections == ()
    assert result.hp_ratio is None and result.mp_ratio is None and result.map_id is None


def test_zero_hp_remains_a_known_measurement_not_missing() -> None:
    image = _image()
    ImageDraw.Draw(image).rectangle((8, 7, 107, 11), fill=BAR_BG)
    assert PixelPerception().observe(_frame(image)).hp_ratio == 0


def test_corrupted_hp_bar_is_unknown_instead_of_partial_count() -> None:
    image = _image()
    ImageDraw.Draw(image).rectangle((30, 8, 40, 10), fill=(220, 10, 10))
    result = PixelPerception().observe(_frame(image))
    assert result.hp_ratio is None
    assert result.status is Status.UNKNOWN


def test_map_marker_must_be_complete() -> None:
    image = _image()
    image.putpixel((300, 6), (0, 0, 0))
    assert PixelPerception().observe(_frame(image)).map_id is None


def test_dialogue_is_read_from_visible_panel_pixels() -> None:
    image = _image()
    assert not PixelPerception().observe(_frame(image)).dialogue_visible
    ImageDraw.Draw(image).rectangle((80, 34, 240, 68), fill=DIALOGUE_COLOR)
    assert PixelPerception().observe(_frame(image)).dialogue_visible


def test_components_connect_bridges_but_not_diagonal_or_separated_rows() -> None:
    mask = np.zeros((12, 12), dtype=np.bool_)
    mask[0:2, 0:2] = True
    mask[0:2, 4:6] = True
    mask[2, 0:6] = True  # bridge merges two components
    mask[5:7, 0:2] = True  # row gap keeps a new component
    mask[7:9, 2:4] = True  # diagonal only; still separate
    assert set(_components(mask)) == {(0, 0, 6, 3), (0, 5, 2, 7), (2, 7, 4, 9)}


def test_components_reject_single_pixel_noise() -> None:
    mask = np.eye(4, dtype=np.bool_)
    assert _components(mask) == ()


def test_components_match_independent_flood_fill_on_seeded_masks() -> None:
    rng = np.random.default_rng(738)
    for _ in range(20):
        mask = rng.random((12, 17)) < 0.35
        unseen = {(int(x), int(y)) for y, x in zip(*np.nonzero(mask), strict=True)}
        expected = set()
        while unseen:
            point = unseen.pop()
            component, pending = [point], [point]
            while pending:
                x, y = pending.pop()
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        pending.append(neighbor)
            if len(component) >= 3:
                xs, ys = zip(*component, strict=True)
                expected.add((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
        assert set(_components(mask)) == expected


def _platform_boxes(image: Image.Image):
    return {
        detection.bbox
        for detection in PixelPerception().detect(_frame(image))
        if detection.class_id == "platform"
    }


def test_platform_segments_with_real_background_gap_are_not_merged() -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 49, 103), fill=PALETTE["platform"])
    draw.rectangle((60, 100, 99, 103), fill=PALETTE["platform"])
    assert _platform_boxes(image) == {(20, 100, 50, 104), (60, 100, 100, 104)}


@pytest.mark.parametrize("occluder", ["ladder", "player", "monster", "npc", "portal", "loot"])
def test_platform_occluded_by_known_foreground_is_merged(occluder) -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 99, 103), fill=PALETTE["platform"])
    draw.rectangle((50, 80, 59, 120), fill=PALETTE[occluder])
    assert _platform_boxes(image) == {(20, 100, 100, 104)}


def test_single_unexplained_pixel_in_occlusion_gap_prevents_merge() -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 99, 103), fill=PALETTE["platform"])
    draw.rectangle((50, 80, 59, 120), fill=PALETTE["ladder"])
    image.putpixel((55, 102), (1, 2, 3))
    assert _platform_boxes(image) == {(20, 100, 50, 104), (60, 100, 100, 104)}


def test_unknown_foreground_color_does_not_invent_hidden_platform() -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 99, 103), fill=PALETTE["platform"])
    draw.rectangle((50, 80, 59, 120), fill=(250, 250, 250))
    assert _platform_boxes(image) == {(20, 100, 50, 104), (60, 100, 100, 104)}


def test_multiple_fully_explained_gaps_merge_as_one_platform() -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 149, 103), fill=PALETTE["platform"])
    draw.rectangle((50, 80, 59, 120), fill=PALETTE["ladder"])
    draw.rectangle((90, 80, 103, 120), fill=PALETTE["player"])
    assert _platform_boxes(image) == {(20, 100, 150, 104)}


def test_foreground_does_not_merge_platforms_at_different_heights() -> None:
    image = Image.new("RGB", (320, 192), (1, 2, 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 100, 49, 103), fill=PALETTE["platform"])
    draw.rectangle((60, 101, 99, 104), fill=PALETTE["platform"])
    draw.rectangle((50, 80, 59, 120), fill=PALETTE["ladder"])
    assert _platform_boxes(image) == {(20, 100, 50, 104), (60, 101, 100, 105)}


@pytest.mark.parametrize("ratio", [-0.1, float("nan"), True])
def test_observation_rejects_invalid_ratios(ratio) -> None:
    valid = VisualObservation(0, 0, Status.UNKNOWN, 0, (), None, None, None)
    with pytest.raises(ValueError):
        replace(valid, hp_ratio=ratio)
