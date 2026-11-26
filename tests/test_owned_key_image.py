from string import ascii_uppercase

import pytest
from PIL import Image

from maple_vision.owned_key_image import key_tile, read_key_tile


def test_displayed_key_tiles_are_distinct_and_recognized():
    for key in ascii_uppercase:
        assert read_key_tile(key_tile(key).convert("RGBA")) == key


def test_unreadable_key_never_falls_back_to_default_binding():
    assert read_key_tile(Image.new("RGB", (48, 48))) is None
    tile = key_tile("K")
    tile.putpixel((20, 20), (255, 0, 0))
    assert read_key_tile(tile) is None
    tile = key_tile("K").convert("RGBA")
    tile.putalpha(0)
    assert read_key_tile(tile) is None
    assert read_key_tile(key_tile("K").resize((96, 96))) is None
    with pytest.raises(ValueError):
        key_tile("Ctrl")
