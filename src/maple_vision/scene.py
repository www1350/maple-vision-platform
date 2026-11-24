"""Deterministic, authored fixture rendering; labels never enter the image contract.

The palette is intentionally simple so a pixel baseline can exercise a complete
closed loop before users supply authorized recordings and a trained detector.
It is not a renderer of proprietary game assets or an estimate of game rules.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .contracts import Detection, nonnegative_int
from .map_pack import MapPack
from .sprite_pack import SpritePack

if TYPE_CHECKING:
    from .sim_types import SimSnapshot

CLASS_NAMES: Final = ("player", "monster", "loot", "portal", "npc", "platform", "ladder")
PALETTE: Final = {
    "player": (59, 221, 255),
    "monster": (238, 82, 113),
    "loot": (248, 207, 65),
    "portal": (183, 108, 255),
    "npc": (71, 204, 126),
    "platform": (122, 90, 60),
    "ladder": (213, 153, 82),
}
MAP_MARKERS: Final = {"meadow": (35, 163, 95), "cavern": (108, 83, 169)}
HUD_BG: Final = (17, 23, 36)
BAR_BG: Final = (35, 42, 60)
HP_COLOR: Final = (239, 68, 68)
MP_COLOR: Final = (80, 140, 255)
DIALOGUE_COLOR: Final = (219, 229, 242)
BAR_X: Final = 8
BAR_WIDTH: Final = 100
HP_Y: Final = 7
MP_Y: Final = 15


@dataclass(frozen=True)
class ImageFrame:
    """RGB24 bytes and capture identity only; no labels, state, or hidden handles."""

    sequence: int
    timestamp_ms: int
    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        nonnegative_int(self.sequence, "sequence")
        nonnegative_int(self.timestamp_ms, "timestamp_ms")
        for value in (self.width, self.height):
            if type(value) is not int or not 1 <= value <= 8192:
                raise ValueError("image dimensions must be integers in [1, 8192]")
        if not isinstance(self.pixels, bytes) or len(self.pixels) != self.width * self.height * 3:
            raise ValueError("pixels must contain exactly width * height * 3 RGB bytes")

    def to_image(self) -> Image.Image:
        return Image.frombytes("RGB", (self.width, self.height), self.pixels)

    def png_bytes(self) -> bytes:
        stream = BytesIO()
        self.to_image().save(stream, format="PNG")
        return stream.getvalue()

    @classmethod
    def from_file(cls, path: str | Path, sequence: int = 0, timestamp_ms: int = 0) -> "ImageFrame":
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            return cls(sequence, timestamp_ms, rgb.width, rgb.height, rgb.tobytes())


@dataclass(frozen=True)
class RenderedSample:
    """Training/validation artifact. Policies receive ``frame`` alone."""

    frame: ImageFrame
    annotations: tuple[Detection, ...]


class SceneRenderer:
    def __init__(
        self,
        width: int = 320,
        height: int = 192,
        style_seed: int = 0,
        *,
        sprite_pack: SpritePack | None = None,
        map_pack: MapPack | None = None,
    ) -> None:
        if any(type(v) is not int or not 160 <= v <= 4096 for v in (width, height)):
            raise ValueError("viewport dimensions must be integers in [160, 4096]")
        if type(style_seed) is not int or style_seed < 0:
            raise ValueError("style_seed must be a non-negative integer")
        if sprite_pack is not None and not isinstance(sprite_pack, SpritePack):
            raise ValueError("sprite_pack must be a verified SpritePack")
        if map_pack is not None and not isinstance(map_pack, MapPack):
            raise ValueError("map_pack must be a verified MapPack")
        if map_pack is not None and (width, height) != (320, 192):
            raise ValueError("external map packs require a 320 x 192 viewport")
        self.width = width
        self.height = height
        self.style_seed = style_seed
        self.sprite_pack = sprite_pack
        self.map_pack = map_pack
        self.map_markers = MAP_MARKERS if map_pack is None else map_pack.markers

    def camera_x(self, snapshot: "SimSnapshot") -> int:
        center = snapshot.player.x + snapshot.player.w / 2
        return max(0, min(max(0, snapshot.width - self.width), round(center - self.width / 2)))

    def render(self, snapshot: "SimSnapshot") -> ImageFrame:
        return self._draw(snapshot, with_labels=False).frame

    def annotations(self, snapshot: "SimSnapshot") -> tuple[Detection, ...]:
        return self.render_sample(snapshot).annotations

    def render_sample(self, snapshot: "SimSnapshot") -> RenderedSample:
        return self._draw(snapshot, with_labels=True)

    def _draw(self, snapshot: "SimSnapshot", *, with_labels: bool) -> RenderedSample:
        if snapshot.map_id not in self.map_markers:
            raise ValueError("renderer does not know this fixture map")
        if self.map_pack is not None and (snapshot.width, snapshot.height) != (320, 192):
            raise ValueError("external map snapshots require a 320 x 192 world")
        offset = self.camera_x(snapshot)
        cavern = (
            snapshot.map_id == "cavern"
            if self.map_pack is None
            else self.map_pack.get(snapshot.map_id).theme == "cavern"
        )
        sky = (27, 25, 48) if cavern else (30, 57, 76)
        image = Image.new("RGB", (self.width, self.height), sky)
        draw = ImageDraw.Draw(image)
        # Seeded background decoration is independent of object class/position.
        rng = Random(self.style_seed)
        for _ in range(30):
            x = (rng.randrange(self.width * 2) - offset // 3) % self.width
            y = rng.randrange(30, max(31, self.height - 40))
            color = (rng.randrange(40, 70), rng.randrange(65, 90), rng.randrange(85, 110))
            draw.rectangle((x, y, x + rng.randrange(2, 8), y + rng.randrange(1, 4)), fill=color)
        for x in range(-100, snapshot.width + 100, 70):
            px = x - offset // 2
            draw.polygon([(px, 165), (px + 35, 65), (px + 70, 165)], fill=(36, 67, 72))

        mask = Image.new("I", (self.width, self.height), 0)
        masks = ImageDraw.Draw(mask)
        instances: list[tuple[int, str, int]] = []
        visible_masks: list[Image.Image] = []

        def erase_labels(box: tuple[int, int, int, int]) -> None:
            if self.sprite_pack is None:
                masks.rectangle(box, fill=0)
            else:
                for visible in visible_masks:
                    ImageDraw.Draw(visible).rectangle(box, fill=0)

        def rectangle(class_id: str, x: float, y: float, w: float, h: float) -> None:
            x1, y1 = round(x - offset), round(y)
            x2, y2 = x1 + max(1, round(w)), y1 + max(1, round(h))
            instance_id = len(instances) + 1
            clipped_area = max(0, min(self.width, x2) - max(0, x1)) * max(
                0, min(self.height, y2) - max(0, y1)
            )
            asset = self.sprite_pack.get(class_id) if self.sprite_pack is not None else None
            if asset is None:
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=PALETTE[class_id])
                alpha = Image.new("L", (x2 - x1, y2 - y1), 255)
            else:
                sprite = asset.image().resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
                alpha = sprite.getchannel("A")
                image.paste(sprite.convert("RGB"), (x1, y1), alpha)
            if with_labels and self.sprite_pack is not None:
                # Partial alpha contributes visible pixels but does not completely
                # hide the underlying object. Only fully opaque pixels erase it.
                visible = Image.new("L", (self.width, self.height), 0)
                opaque = Image.new("L", (self.width, self.height), 0)
                visible.paste(alpha.point([0] + [255] * 255), (x1, y1))
                opaque.paste(alpha.point([0] * 255 + [255]), (x1, y1))
                clipped_area = int(np.count_nonzero(np.asarray(visible)))
                for index, previous in enumerate(visible_masks):
                    visible_masks[index] = ImageChops.subtract(previous, opaque)
                visible_masks.append(visible)
            elif with_labels:
                masks.rectangle((x1, y1, x2 - 1, y2 - 1), fill=instance_id)
            instances.append((instance_id, class_id, clipped_area))

        for platform in snapshot.platforms:
            if platform.y >= snapshot.height - 30:
                draw.rectangle(
                    (round(platform.x1) - offset, round(platform.y) + 4, self.width, self.height),
                    fill=(57, 48, 47),
                )
            rectangle("platform", platform.x1, platform.y, platform.x2 - platform.x1, 4)
        for ladder in snapshot.ladders:
            rectangle("ladder", ladder.x, ladder.y, ladder.w, ladder.h)
        for portal in snapshot.portals:
            rectangle("portal", portal.x, portal.y, portal.w, portal.h)
        for npc in snapshot.npcs:
            rectangle("npc", npc.x, npc.y, npc.w, npc.h)
        for drop in snapshot.drops:
            rectangle("loot", drop.x, drop.y, drop.w, drop.h)
        for monster in snapshot.monsters:
            if monster.hp > 0:
                rectangle("monster", monster.x, monster.y, monster.w, monster.h)
        player = snapshot.player
        if player.alive:
            rectangle("player", player.x, player.y, player.w, player.h)
            if self.sprite_pack is None or self.sprite_pack.get("player") is None:
                # The authored fixture eye must not paint over a registered sprite.
                eye_x = round(player.x - offset + (player.w - 4 if player.facing > 0 else 2))
                eye_y = round(player.y + 4)
                draw.rectangle((eye_x, eye_y, eye_x + 1, eye_y + 1), fill=(245, 250, 255))

        draw.rectangle((3, 3, 115, 26), fill=HUD_BG)
        if with_labels:
            erase_labels((3, 3, 115, 26))
        for y, value, maximum, color in (
            (HP_Y, player.hp, player.max_hp, HP_COLOR),
            (MP_Y, player.mp, player.max_mp, MP_COLOR),
        ):
            draw.rectangle((BAR_X, y, BAR_X + BAR_WIDTH - 1, y + 4), fill=BAR_BG)
            length = round(BAR_WIDTH * max(0, min(1, value / maximum))) if maximum > 0 else 0
            if length:
                draw.rectangle((BAR_X, y, BAR_X + length - 1, y + 4), fill=color)
        draw.rectangle(
            (self.width - 20, 6, self.width - 9, 17), fill=self.map_markers[snapshot.map_id]
        )
        if with_labels:
            erase_labels((self.width - 20, 6, self.width - 9, 17))
        if snapshot.dialogue is not None:
            panel = (self.width // 4, 34, self.width * 3 // 4, 68)
            draw.rectangle(panel, fill=DIALOGUE_COLOR)
            draw.text((panel[0] + 4, 40), "NPC: " + snapshot.dialogue[:22], fill=(28, 35, 52))
            if with_labels:
                erase_labels(panel)

        annotations: list[Detection] = []
        if with_labels:
            instance_pixels = np.asarray(mask)
            for instance_id, class_id, area in instances:
                visible_pixels = (
                    instance_pixels == instance_id
                    if self.sprite_pack is None
                    else np.asarray(visible_masks[instance_id - 1]) > 0
                )
                ys, xs = np.nonzero(visible_pixels)
                # Severe occlusion is excluded rather than labeled through a wall.
                if len(xs) < 3 or not area or len(xs) / area < 0.25:
                    continue
                bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                annotations.append(Detection(class_id, bbox, 1.0))
        frame = ImageFrame(
            snapshot.sequence, snapshot.timestamp_ms, self.width, self.height, image.tobytes()
        )
        return RenderedSample(frame, tuple(annotations))
