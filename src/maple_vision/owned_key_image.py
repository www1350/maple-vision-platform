"""Recognize the A–Z key tiles rendered in the project's own desktop settings panel."""

from functools import lru_cache
from string import ascii_uppercase

from PIL import Image, ImageDraw, ImageFont

_SIZE = (48, 48)


@lru_cache(maxsize=26)
def _tile_bytes(key: str) -> bytes:
    if len(key) != 1 or key not in ascii_uppercase:
        raise ValueError("expected one uppercase ASCII letter")
    tile = Image.new("RGB", _SIZE, (30, 36, 48))
    ImageDraw.Draw(tile).text(
        (24, 24), key, font=ImageFont.load_default(size=30), fill="white", anchor="mm"
    )
    return tile.tobytes()


def key_tile(key: str) -> Image.Image:
    return Image.frombytes("RGB", _SIZE, _tile_bytes(key))


def read_key_tile(image: Image.Image) -> str | None:
    if image.size != _SIZE or image.mode not in {"RGB", "RGBA"}:
        return None
    if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
        return None
    pixels = image.convert("RGB").tobytes()
    return next((key for key in ascii_uppercase if _tile_bytes(key) == pixels), None)
