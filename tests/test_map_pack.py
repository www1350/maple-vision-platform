"""Map data validates independently of rendering, policy, and the simulator."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from maple_vision.map_pack import MAX_MAP_PACK_BYTES, MapDefinition, MapPack, load_map_pack
from maple_vision.sim_types import Ladder, Platform

EXAMPLE = Path(__file__).parents[1] / "examples" / "maps" / "three-map-fixture.json"


def source():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_authored_three_map_fixture_has_complete_cycle_and_independent_digest():
    raw = source()
    pack = load_map_pack(EXAMPLE)
    assert pack.map_ids == ("grove", "quarry", "ruins")
    assert pack.entry_map == "grove"
    assert (pack.width, pack.height, pack.ground) == (320, 192, 168)
    assert pack.to_dict() == raw
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert pack.content_sha256 == digest
    assert pack.manifest() == {"schema_version": 1, "content": raw, "content_sha256": digest}
    for index, map_id in enumerate(pack.map_ids):
        definition = pack.get(map_id)
        assert definition.portals[0].destination == pack.map_ids[(index + 1) % 3]
        assert definition.spawn_y == 146
        assert (definition.width, definition.height, definition.ground) == (320, 192, 168)
        assert len(definition.monster_x) == 3
    assert dict(pack.markers) == {item["id"]: tuple(item["marker"]) for item in raw["maps"]}
    assert MapPack.from_manifest(pack.manifest()) == pack


def test_map_order_is_normalized_by_explicit_route_and_whitespace_does_not_change_identity(
    tmp_path,
):
    raw = source()
    raw["maps"].reverse()
    path = tmp_path / "maps.json"
    path.write_text(json.dumps(raw, indent=7), encoding="utf-8")
    loaded = load_map_pack(path)
    expected = load_map_pack(EXAMPLE)
    assert loaded.to_dict() == expected.to_dict()
    assert loaded.content_sha256 == expected.content_sha256
    assert loaded.map_ids == expected.map_ids
    assert loaded == expected == MapPack.from_manifest(loaded.manifest())
    assert replace(expected, maps=tuple(reversed(expected.maps))) == expected


def test_route_order_is_semantic_and_part_of_content_identity():
    raw = source()
    original = MapPack.from_dict(raw)
    raw["route"] = ["quarry", "ruins", "grove"]
    shifted = MapPack.from_dict(raw)
    assert shifted.entry_map == "quarry"
    assert shifted.content_sha256 != original.content_sha256


def test_data_is_deeply_immutable_and_serialized_copies_do_not_mutate_it():
    raw = source()
    pack = MapPack.from_dict(raw)
    definition = pack.get("grove")
    with pytest.raises(FrozenInstanceError):
        pack.version = "2.0.0"
    with pytest.raises(FrozenInstanceError):
        definition.spawn_x = 100
    with pytest.raises(FrozenInstanceError):
        definition.platforms[0].x1 = 1
    with pytest.raises(TypeError):
        pack.markers["grove"] = (1, 2, 3)
    raw["maps"][0]["spawn_x"] = 100
    exported = pack.to_dict()
    exported["maps"][0]["spawn_x"] = 200
    manifest = pack.manifest()
    manifest["content"]["route"].reverse()
    assert pack.get("grove").spawn_x == 24
    assert pack.map_ids == ("grove", "quarry", "ruins")


def test_fixture_paint_colors_cannot_be_map_markers():
    # A marker painted with an object color would invent an extra detection in
    # the HUD. Keep the data-loader contract aligned with the actual renderer.
    from maple_vision.scene import (
        BAR_BG,
        DIALOGUE_COLOR,
        HP_COLOR,
        HUD_BG,
        MAP_MARKERS,
        MP_COLOR,
        PALETTE,
    )

    definition = load_map_pack(EXAMPLE).get("grove")
    colors = {
        *PALETTE.values(),
        *MAP_MARKERS.values(),
        BAR_BG,
        DIALOGUE_COLOR,
        HP_COLOR,
        HUD_BG,
        MP_COLOR,
    }
    for color in colors:
        with pytest.raises(ValueError, match="reserved fixture color"):
            replace(definition, marker=color)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("width", 640),
        ("width", 320.0),
        ("height", 193),
        ("ground", 169),
        ("scope", "game_reference"),
        ("scope", ["synthetic_fixture"]),
        ("id", "../external"),
        ("id", "x" * 41),
        ("id", "Uppercase"),
        ("version", "next"),
        ("version", "1.0"),
        ("version", 1),
        ("source_uri", "relative.json"),
        ("source_uri", "https://user:secret@example.com/x"),
        ("permission", "unverified"),
        ("license", ""),
        ("route", ["grove", "grove", "ruins"]),
        ("route", ["grove", "quarry"]),
        ("route", ["grove", "quarry", "missing"]),
        ("route", ["grove"]),
    ],
)
def test_invalid_pack_metadata_dimensions_and_route_are_rejected(field, value):
    raw = source()
    raw[field] = value
    with pytest.raises(ValueError):
        MapPack.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "meadow"),
        ("id", "cavern"),
        ("id", "../map"),
        ("theme", "desert"),
        ("theme", ["meadow"]),
        ("marker", [1, 2]),
        ("marker", [1, 2, 256]),
        ("marker", [1, True, 3]),
        ("marker", [59, 221, 255]),
        ("marker", [35, 163, 95]),
        ("spawn_x", -1),
        ("spawn_x", 307),
        ("spawn_x", 24.0),
        ("monster_x", []),
        ("monster_x", [1]),
        ("monster_x", [300]),
        ("monster_x", [120, 120]),
        ("monster_x", [120, 143]),
        ("monster_x", [True]),
        ("monster_x", [100.0]),
        ("monster_x", [10, 50, 90, 130, 170, 210, 250]),
        ("npcs", []),
        ("portals", []),
        ("ladders", []),
        ("platforms", []),
    ],
)
def test_invalid_map_definition_is_rejected(field, value):
    raw = source()["maps"][0]
    raw[field] = value
    with pytest.raises(ValueError):
        MapDefinition.from_dict(raw)


@pytest.mark.parametrize(
    ("collection", "index", "field", "value"),
    [
        ("platforms", 0, "x1", 1),
        ("platforms", 0, "y", 167),
        ("platforms", 1, "x1", -1),
        ("platforms", 1, "x2", 321),
        ("platforms", 1, "x2", 70),
        ("platforms", 1, "y", 71),
        ("platforms", 1, "y", 168),
        ("platforms", 1, "y", float("nan")),
        ("portals", 0, "x", 301),
        ("portals", 0, "y", 131),
        ("portals", 0, "w", 21),
        ("portals", 0, "h", 35),
        ("portals", 0, "destination", "no-such-map"),
        ("npcs", 0, "x", 307),
        ("npcs", 0, "y", 143),
        ("npcs", 0, "w", 14.0),
        ("npcs", 0, "h", 25),
        ("npcs", 0, "id", "to_quarry"),
        ("ladders", 0, "x", 0),
        ("ladders", 0, "y", 125),
        ("ladders", 0, "w", 11),
        ("ladders", 0, "h", 45),
    ],
)
def test_geometry_and_destination_constraints_are_enforced(collection, index, field, value):
    raw = source()
    raw["maps"][0][collection][index][field] = value
    with pytest.raises(ValueError):
        MapPack.from_dict(raw)


@pytest.mark.parametrize("x", [0, 1, 309, 310])
def test_ladders_must_leave_room_for_centered_player_in_parser_and_constructor(x):
    raw = source()
    definition = raw["maps"][0]
    support = {"x1": 0 if x < 2 else 256, "x2": 64 if x < 2 else 320, "y": 124}
    definition["platforms"][1] = support
    definition["ladders"][0]["x"] = x
    with pytest.raises(ValueError, match=r"ladder x.*\[2, 308\]"):
        MapPack.from_dict(raw)
    original = load_map_pack(EXAMPLE).get("grove")
    with pytest.raises(ValueError, match=r"ladder x.*\[2, 308\]"):
        replace(
            original,
            platforms=(original.platforms[0], Platform(**support), original.platforms[2]),
            ladders=(Ladder(x, 124, 10, 44),),
        )


@pytest.mark.parametrize("x", [2, 308])
def test_ladder_player_margin_boundaries_are_accepted(x):
    raw = source()
    definition = raw["maps"][0]
    definition["platforms"][1] = {
        "x1": 0 if x == 2 else 256,
        "x2": 64 if x == 2 else 320,
        "y": 124,
    }
    definition["ladders"][0]["x"] = x
    pack = MapPack.from_dict(raw)
    assert pack.get("grove").ladders[0].x == x
    assert MapPack.from_manifest(pack.manifest()) == pack


def test_duplicate_map_ids_markers_and_route_disconnection_are_rejected():
    for change in ("duplicate_id", "duplicate_marker", "disconnected", "reversed_route"):
        raw = source()
        if change == "duplicate_id":
            raw["maps"][1]["id"] = "grove"
        elif change == "duplicate_marker":
            raw["maps"][1]["marker"] = raw["maps"][0]["marker"]
        elif change == "disconnected":
            raw["maps"][0]["portals"][0]["destination"] = "grove"
        else:
            raw["route"].reverse()
        with pytest.raises(ValueError):
            MapPack.from_dict(raw)


def test_overlap_duplicate_ladders_and_entity_count_budgets_are_rejected():
    for field in ("platforms", "ladders", "portals", "npcs", "maps"):
        raw = source()
        if field == "maps":
            raw[field] *= 3
        else:
            raw["maps"][0][field].append(copy.deepcopy(raw["maps"][0][field][-1]))
        with pytest.raises(ValueError):
            MapPack.from_dict(raw)


def test_unknown_fields_at_every_schema_layer_are_rejected():
    for target in ("pack", "map", "platforms", "portals", "npcs", "ladders", "manifest"):
        raw = source()
        if target == "manifest":
            manifest = MapPack.from_dict(raw).manifest()
            manifest["external_script"] = "run.py"
            with pytest.raises(ValueError, match="invalid fields"):
                MapPack.from_manifest(manifest)
            continue
        item = raw if target == "pack" else raw["maps"][0]
        if target not in {"pack", "map"}:
            item = item[target][0]
        item["external_script"] = "run.py"
        with pytest.raises(ValueError, match="invalid fields"):
            MapPack.from_dict(raw)


def test_direct_dataclass_construction_does_not_bypass_validation():
    pack = load_map_pack(EXAMPLE)
    with pytest.raises(ValueError, match="immutable"):
        replace(pack, maps=list(pack.maps))
    with pytest.raises(ValueError, match="immutable"):
        replace(pack, route=list(pack.route))
    definition = pack.get("grove")
    with pytest.raises(ValueError, match="immutable"):
        replace(definition, platforms=list(definition.platforms))
    with pytest.raises(ValueError, match="integer"):
        replace(definition, platforms=(Platform(0.0, 320, 168), *definition.platforms[1:]))
    with pytest.raises(ValueError, match="unknown map"):
        pack.get("missing")


def test_embedded_content_hash_rejects_tampered_data_and_invalid_hashes():
    original = load_map_pack(EXAMPLE).manifest()
    tampered = copy.deepcopy(original)
    tampered["content"]["maps"][0]["spawn_x"] += 1
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MapPack.from_manifest(tampered)
    for bad in (None, "x" * 64, "A" * 64, 1, "0" * 64):
        changed = {**original, "content_sha256": bad}
        with pytest.raises(ValueError, match="SHA256"):
            MapPack.from_manifest(changed)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b'{"schema_version":Infinity}',
        b"\xff",
        b"[" * 2000 + b"]" * 2000,
        b" " * (MAX_MAP_PACK_BYTES + 1),
    ],
)
def test_file_loader_rejects_duplicate_keys_nonfinite_encoding_recursion_and_size(
    tmp_path, payload
):
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_map_pack(path)


def test_unpaired_unicode_is_rejected_with_validation_error():
    raw = source()
    raw["license"] = "unpaired \ud800 surrogate"
    with pytest.raises(ValueError, match="invalid text"):
        MapPack.from_dict(raw)


def test_loader_rejects_nonfiles_and_symlink_before_read(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="regular file"):
        load_map_pack(tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        load_map_pack(tmp_path / "missing")
    # Exercise the rejection branch without relying on Windows symlink privileges.
    monkeypatch.setattr(Path, "is_symlink", lambda _: True)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("source was read"))
    with pytest.raises(ValueError, match="symlink"):
        load_map_pack(EXAMPLE)
