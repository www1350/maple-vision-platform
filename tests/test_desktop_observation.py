from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from maple_vision.contracts import Status
from maple_vision.desktop_observation import (
    DesktopObservationSession,
    DetectorOnlyPerception,
    fixture_desktop_observation_session,
    onnx_desktop_observation_session,
    verify_desktop_observation_evidence,
    write_desktop_observation_evidence,
)
from maple_vision.desktop_window import DesktopWindowSession, WindowIdentity
from maple_vision.onnx_detector import OnnxDetector
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
from maple_vision.state_estimation import TemporalStateEstimator

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


def test_detector_boxes_are_visible_but_never_invent_hud_or_actionability() -> None:
    class Detector:
        def detect(self, frame):
            from maple_vision.contracts import Detection

            return (Detection("player", (10, 20, 30, 60), 0.9),)

    desktop = desktop_for(Image.new("RGB", (640, 480), "#123456"))
    result = DesktopObservationSession(
        desktop,
        DetectorOnlyPerception(Detector()),
        estimator=TemporalStateEstimator(retain_detection_hypotheses=True),
        perception_mode="detector_test",
        perception_identity={"implementation": "test.Detector"},
        limitations=("test detector only",),
    ).observe()
    assert len(result.observation.detections) == 1
    assert result.observation.status is Status.UNKNOWN
    assert result.observation.map_id is None
    assert result.observation.hp_ratio is None
    assert result.observation.mp_ratio is None
    assert not result.belief.actionable
    assert result.belief.reason == "fresh_detection_hypotheses"
    assert result.belief.field("player_bbox").value == (10, 20, 30, 60)
    assert len(result.belief.tracks) == 1
    assert result.belief.tracks[0].class_id == "player"
    assert result.belief.last_accepted_sequence is None


def test_shipped_hash_bound_onnx_model_can_observe_external_pixels() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "lifecycle-model"
    detector = OnnxDetector(root / "model.onnx", root / "manifest.json")
    result = onnx_desktop_observation_session(desktop_for(fixture_image()), detector).observe()
    assert result.perception_mode == "onnx_detector_only_v1"
    assert result.perception_identity["loaded_model_sha256"] == detector.loaded_model_sha256
    assert result.observation.status is Status.UNKNOWN
    assert not result.belief.actionable
    assert result.belief.reason == "fresh_detection_hypotheses"
    assert {track.class_id for track in result.belief.tracks} >= {"player", "monster"}


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
    assert report["schema_version"] == 2
    assert report["observation"]["sequence"] == report["frame"]["sequence"]
    verified = verify_desktop_observation_evidence(output)
    assert verified["status"] == "known"
    assert verified["perception_identity"]["model_manifest"] is None
    Image.new("RGB", (320, 192), "#000000").save(output / "recognized-overlay.png")
    with pytest.raises(ValueError, match="overlay.png changed"):
        verify_desktop_observation_evidence(output)
