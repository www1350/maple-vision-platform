from dataclasses import fields, replace

import numpy as np
import pytest

from maple_vision.scene import ImageFrame, SceneRenderer
from maple_vision.sim_types import (
    NPC,
    DropState,
    MonsterState,
    Platform,
    PlayerState,
    Portal,
    SimSnapshot,
)


def _snapshot() -> SimSnapshot:
    player = PlayerState(
        30, 146, 14, 22, 0, 0, 100, 100, 30, 60, 80, 3, 3, 0, (), True, 1, "assassin", True
    )
    return SimSnapshot(
        7,
        350,
        "meadow",
        320,
        192,
        player,
        (MonsterState("m1", 120, 152, 18, 16, 30, 30),),
        (DropState("d1", 200, 162, 6, 6, "coin", 1),),
        (Platform(0, 320, 168), Platform(80, 145, 105)),
        (Portal("p", 292, 140, 16, 28, "cavern"),),
        (NPC("n", 65, 146, 14, 22),),
        (),
        (),
        None,
    )


def test_pixels_have_no_truth_or_simulator_handles() -> None:
    snapshot = _snapshot()
    frame = SceneRenderer().render(snapshot)
    assert {field.name for field in fields(frame)} == {
        "sequence",
        "timestamp_ms",
        "width",
        "height",
        "pixels",
    }
    assert frame.sequence == 7 and frame.timestamp_ms == 350
    assert len(frame.pixels) == 320 * 192 * 3
    assert frame.pixels == SceneRenderer().render(snapshot).pixels


@pytest.mark.parametrize("width,height,pixels", [(0, 1, b""), (1, True, b"abc"), (2, 2, b"abc")])
def test_rgb_contract_rejects_malformed_buffers(width: int, height: int, pixels: bytes) -> None:
    with pytest.raises(ValueError):
        ImageFrame(0, 0, width, height, pixels)


def test_visible_labels_have_independently_known_bounds() -> None:
    sample = SceneRenderer().render_sample(_snapshot())
    boxes = {item.class_id: item.bbox for item in sample.annotations if item.class_id != "platform"}
    assert boxes["player"] == (30, 146, 44, 168)
    assert boxes["monster"] == (120, 152, 138, 168)
    assert boxes["loot"] == (200, 162, 206, 168)
    assert boxes["portal"] == (292, 140, 308, 168)


def test_camera_clips_labels_and_removes_offscreen_entities() -> None:
    snapshot = _snapshot()
    snapshot = replace(snapshot, player=replace(snapshot.player, x=160))
    renderer = SceneRenderer(width=160)
    assert renderer.camera_x(snapshot) == 87
    sample = renderer.render_sample(snapshot)
    player = next(item for item in sample.annotations if item.class_id == "player")
    assert player.bbox == (73, 146, 87, 168)
    assert not any(item.class_id == "portal" for item in sample.annotations)
    platform = next(
        item for item in sample.annotations if item.class_id == "platform" and item.bbox[1] == 105
    )
    assert platform.bbox == (0, 105, 58, 109)
    assert all(0 <= item.bbox[0] < item.bbox[2] <= 160 for item in sample.annotations)


def test_fully_occluded_monster_is_not_labeled_through_player() -> None:
    snapshot = _snapshot()
    monster = MonsterState("hidden", 32, 149, 9, 13, 30, 30)
    snapshot = replace(snapshot, monsters=(monster,))
    assert not any(d.class_id == "monster" for d in SceneRenderer().annotations(snapshot))


def test_png_roundtrip_has_exact_pixels(tmp_path) -> None:
    frame = SceneRenderer().render(_snapshot())
    path = tmp_path / "frame.png"
    path.write_bytes(frame.png_bytes())
    loaded = ImageFrame.from_file(path, sequence=7, timestamp_ms=350)
    assert loaded == frame


def test_style_variation_changes_background_without_changing_labels() -> None:
    a = SceneRenderer(style_seed=1).render_sample(_snapshot())
    b = SceneRenderer(style_seed=2).render_sample(_snapshot())
    assert a.frame.pixels != b.frame.pixels
    assert a.annotations == b.annotations
    aa = np.frombuffer(a.frame.pixels, dtype=np.uint8).reshape(192, 320, 3)
    bb = np.frombuffer(b.frame.pixels, dtype=np.uint8).reshape(192, 320, 3)
    assert np.array_equal(aa[146:168, 30:44], bb[146:168, 30:44])
