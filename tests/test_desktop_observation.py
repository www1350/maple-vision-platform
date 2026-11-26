from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from maple_vision.contracts import Status
from maple_vision.desktop_observation import (
    fixture_desktop_observation_session,
    verify_desktop_observation_evidence,
    write_desktop_observation_evidence,
)
from maple_vision.desktop_window import DesktopWindowSession, WindowIdentity
from maple_vision.scene import (
    BAR_BG,
    BAR_WIDTH,
    BAR_X,
    HP_COLOR,
    HP_Y,
    HUD_BG,
    MAP_MARKERS,
    MP_COLOR,
    MP_Y,
    PALETTE,
)

IDENTITY = WindowIdentity(
    71, 72, 73, 74, "OwnedFixture", "Fixture Observation", (0, 0, 320, 192), 96
)


def fixture_image() -> Image.Image:
    image = Image.new("RGB", (320, 192), "#1e394c")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 24), fill=HUD_BG)
    draw.rectangle((BAR_X, HP_Y + 2, BAR_X + BAR_WIDTH - 1, HP_Y + 2), fill=BAR_BG)
    draw.rectangle((BAR_X, MP_Y + 2, BAR_X + BAR_WIDTH - 1, MP_Y + 2), fill=BAR_BG)
    draw.rectangle((BAR_X, HP_Y + 2, BAR_X + 79, HP_Y + 2), fill=HP_COLOR)
    draw.rectangle((BAR_X, MP_Y + 2, BAR_X + 69, MP_Y + 2), fill=MP_COLOR)
    draw.rectangle((300, 6, 311, 17), fill=MAP_MARKERS["meadow"])
    draw.rectangle((100, 90, 111, 113), fill=PALETTE["player"])
    draw.rectangle((170, 98, 185, 117), fill=PALETTE["monster"])
    return image


class FakeDesktop:
    def __init__(self, image: Image.Image) -> None:
        self.image = image

    def list_windows(self) -> tuple[WindowIdentity, ...]:
        return (IDENTITY,)

    def describe(self, handle: int) -> WindowIdentity | None:
        return IDENTITY if handle == IDENTITY.handle else None

    def foreground_handle(self) -> int | None:
        return IDENTITY.handle

    def capture(self, identity: WindowIdentity) -> Image.Image:
        assert identity == IDENTITY
        return self.image.copy()


def desktop_for(image: Image.Image) -> DesktopWindowSession:
    desktop = DesktopWindowSession(
        FakeDesktop(image), release_key=lambda _: None, clock=lambda: 1_000_000_000
    )
    desktop.select(IDENTITY.handle)
    desktop.confirm()
    return desktop


def test_external_pixels_feed_perception_and_state_on_the_same_frame() -> None:
    desktop = desktop_for(fixture_image())
    result = fixture_desktop_observation_session(desktop).observe()
    assert result.observation.status is Status.KNOWN
    assert result.observation.map_id == "meadow"
    assert result.observation.hp_ratio == 0.8
    assert result.observation.mp_ratio == 0.7
    assert result.observation.source_id == "external-window:synthetic_fixture_palette_v2"
    assert result.belief.actionable
    assert result.belief.observation is result.observation
    assert desktop.events[-1]["frame_rgb_sha256"] == result.frame.rgb_sha256


def test_unrecognized_external_pixels_remain_unknown() -> None:
    result = fixture_desktop_observation_session(
        desktop_for(Image.new("RGB", (320, 192), "#123456"))
    ).observe()
    assert result.observation.status is Status.UNKNOWN
    assert not result.belief.actionable
    assert result.belief.reason == "unknown_observation"


def test_fixture_mode_rejects_dimension_drift_and_pauses() -> None:
    desktop = desktop_for(Image.new("RGB", (321, 192), "#123456"))
    with pytest.raises(ValueError, match="does not match"):
        fixture_desktop_observation_session(desktop).observe()
    assert (desktop.state, desktop.reason) == ("paused", "capture_size_changed")


def test_observation_bundle_binds_capture_overlay_and_belief(tmp_path: Path) -> None:
    desktop = desktop_for(fixture_image())
    receipt = desktop.events[-1]
    result = fixture_desktop_observation_session(desktop).observe()
    desktop.close()
    output = tmp_path / "observation"
    report = write_desktop_observation_evidence(
        output, result, receipt=receipt, events=desktop.events
    )
    assert report["observation"]["sequence"] == report["frame"]["sequence"]
    assert verify_desktop_observation_evidence(output)["status"] == "known"
    Image.new("RGB", (320, 192), "#000000").save(output / "recognized-overlay.png")
    with pytest.raises(ValueError, match="overlay.png changed"):
        verify_desktop_observation_evidence(output)
