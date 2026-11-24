"""External map physics and replay, with frozen legacy byte compatibility."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.contracts import Status
from maple_vision.map_pack import MapPack, load_map_pack
from maple_vision.scene import MAP_MARKERS, PALETTE, SceneRenderer
from maple_vision.sim_types import CLASS_SPECS, SimAction
from maple_vision.simulation import MAP_IDS, MAP_SPECS, Simulator
from maple_vision.vision import PixelPerception

EXAMPLE = Path(__file__).parents[1] / "examples/maps/three-map-fixture.json"


@pytest.fixture
def pack():
    return load_map_pack(EXAMPLE)


@pytest.mark.parametrize(
    "job,expected",
    [
        ("assassin", "6bc41703f46ec319a9901fc181394ab84f04c47d15a886330bee86b2942a5ccc"),
        ("dragon_knight", "c288999eee435e32da039a7d30a5dc020f5f43a9de591d746489159bfe9975e4"),
        ("cleric", "1ebe8e7db7ea1951a817bca10852baf26536cc37d86e4010588540cc8f467d94"),
    ],
)
def test_default_state_and_pixels_match_v03_bytes(job, expected):
    # Captured independently from main 493b7d24688c6aad22d7583f11e8098a28965567,
    # before injecting MapPack into any of the simulator/render/perception code.
    sim = Simulator(seed=42, job=job)
    digest = hashlib.sha256()
    actions = [SimAction("buff"), SimAction("move", 1), SimAction("jump_attack", 1)]
    actions += [SimAction()] * 25
    actions += [SimAction("attack", 1), SimAction("potion_mp"), SimAction("move", -1)]
    for action in actions:
        sim.step(action)
        digest.update(
            json.dumps(sim.save(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(SceneRenderer().render(sim.snapshot()).pixels)
    assert digest.hexdigest() == expected
    assert sim.save()["schema_version"] == 1
    assert sim.map_pack is None
    assert sim.map_ids == MAP_IDS


@pytest.mark.parametrize("job", CLASS_SPECS)
def test_pack_supplies_each_map_geometry_spawn_and_seeded_monsters(pack, job):
    for map_id in pack.map_ids:
        definition = pack.get(map_id)
        sim = Simulator(seed=19, job=job, map_id=map_id, map_pack=pack)
        snapshot = sim.snapshot()
        assert snapshot.map_id == map_id
        assert (snapshot.width, snapshot.height) == (320, 192)
        assert (snapshot.player.x, snapshot.player.y) == (definition.spawn_x, 146)
        assert snapshot.platforms == definition.platforms
        assert snapshot.npcs == definition.npcs
        assert snapshot.portals == definition.portals
        assert snapshot.ladders == definition.ladders
        assert len(snapshot.monsters) == len(definition.monster_x)
        assert all(
            abs(monster.x - x) <= 3
            for monster, x in zip(snapshot.monsters, definition.monster_x, strict=True)
        )
        assert sim.map_pack == pack
        assert sim.map_ids == pack.route
    assert tuple(MAP_SPECS) == MAP_IDS
    assert set(MAP_MARKERS) == set(MAP_IDS)
    assert Simulator().snapshot().map_id == "meadow"


@pytest.mark.parametrize("with_profile", [False, True])
def test_embedded_pack_and_profile_restore_without_original_file(tmp_path, with_profile):
    source = tmp_path / "map.json"
    source.write_bytes(EXAMPLE.read_bytes())
    pack = load_map_pack(source)
    profile = replace(CLASS_SPECS["assassin"], walk_speed=96) if with_profile else None
    sim = Simulator(seed=41, map_id=pack.entry_map, map_pack=pack, class_spec=profile)
    sim.step(SimAction("jump", 1))
    save = json.loads(json.dumps(sim.save(), allow_nan=False))
    assert save["schema_version"] == 3
    assert ("class_profile" in save) == with_profile
    assert save["map_pack"] == pack.manifest()
    source.unlink()
    restored = Simulator.restore(save)
    actions = [SimAction()] * 25 + [SimAction("move", 1), SimAction("attack", 1)]
    assert [sim.step(action) for action in actions] == [restored.step(action) for action in actions]
    assert sim.save() == restored.save()
    assert SceneRenderer(map_pack=restored.map_pack).render(restored.snapshot()) == SceneRenderer(
        map_pack=pack
    ).render(sim.snapshot())
    restored.reset()
    assert (
        restored.save()
        == Simulator(seed=41, map_id=pack.entry_map, map_pack=pack, class_spec=profile).save()
    )


def test_save_does_not_alias_pack_and_reset_preserves_selected_map(pack):
    selected = pack.map_ids[-1]
    sim = Simulator(seed=5, map_id=selected, map_pack=pack)
    before = sim.save()
    changed = sim.save()
    changed["map_pack"]["content"]["route"].reverse()
    assert sim.save() == before
    sim.step(SimAction("move", 1))
    sim.reset()
    assert sim.save() == before
    assert sim.snapshot().map_id == selected


def test_all_portals_form_the_authored_cycle_and_arrive_at_target_spawn(pack):
    sim = Simulator(seed=3, map_id=pack.entry_map, map_pack=pack)
    visited = []
    for current in pack.map_ids:
        assert sim.snapshot().map_id == current
        portal = pack.get(current).portals[0]
        state = sim.save()
        state["player"]["x"] = float(portal.x + portal.w / 2 - 7)
        state["player"]["y"] = 146.0
        sim = Simulator.restore(state)
        outcome = sim.step(SimAction("portal", target=portal.id))
        assert outcome.accepted
        assert "map_changed" in outcome.events
        assert outcome.snapshot.map_id == portal.destination
        assert outcome.snapshot.player.x == pack.get(portal.destination).spawn_x
        assert outcome.snapshot.player.y == 146
        visited.append(current)
    assert tuple(visited) == pack.map_ids
    assert sim.snapshot().map_id == pack.entry_map


@pytest.mark.parametrize(
    "kind,target_field,object_y", [("portal", "portals", 132), ("interact", "npcs", 144)]
)
@pytest.mark.parametrize("gap", [0, 4])
def test_external_interaction_cannot_reach_down_from_a_nonoverlapping_platform(
    pack, kind, target_field, object_y, gap
):
    content = pack.to_dict()
    definition = content["maps"][0]
    target = definition[target_field][0]
    # Add a legal platform above the target; the first ladder's own platform
    # remains untouched. Merely touching the target's top is not overlap.
    definition["platforms"].append(
        {"x1": max(0, target["x"] - 4), "x2": min(320, target["x"] + 28), "y": object_y - gap}
    )
    elevated_pack = MapPack.from_dict(content)
    state = Simulator(map_id=elevated_pack.entry_map, map_pack=elevated_pack).save()
    state["player"]["x"] = float(target["x"])
    state["player"]["y"] = float(object_y - gap - 22)
    state["maps"][elevated_pack.entry_map]["monsters"] = []
    sim = Simulator.restore(state)
    result = sim.step(SimAction(kind, target=target["id"]))
    assert not result.accepted
    assert result.snapshot.map_id == elevated_pack.entry_map
    assert result.snapshot.dialogue is None
    assert Simulator.restore(sim.save()).snapshot() == sim.snapshot()


def test_external_portal_still_requires_a_grounded_player(pack):
    state = Simulator(map_id=pack.entry_map, map_pack=pack).save()
    portal = pack.get(pack.entry_map).portals[0]
    state["player"]["x"] = float(portal.x)
    state["player"]["y"] = 140.0
    state["player"]["grounded"] = False
    sim = Simulator.restore(state)
    result = sim.step(SimAction("portal", target=portal.id))
    assert not result.accepted
    assert result.snapshot.map_id == pack.entry_map


def test_npcs_and_ladders_use_external_geometry(pack):
    for map_id in pack.map_ids:
        definition = pack.get(map_id)
        state = Simulator(map_id=map_id, map_pack=pack).save()
        state["maps"][map_id]["monsters"] = []
        npc = definition.npcs[0]
        state["player"]["x"] = float(npc.x)
        sim = Simulator.restore(state)
        result = sim.step(SimAction("interact", target=npc.id))
        assert result.accepted
        assert result.snapshot.dialogue.startswith(npc.id + ":")
        ladder = definition.ladders[0]
        state = sim.save()
        state["player"]["x"] = float(ladder.x + ladder.w / 2 - 7)
        sim = Simulator.restore(state)
        for _ in range(60):
            result = sim.step(SimAction("climb", -1))
            assert result.accepted
        assert result.snapshot.player.y == ladder.y - 22
        assert result.snapshot.player.grounded
        assert Simulator.restore(sim.save()).snapshot() == sim.snapshot()


@pytest.mark.parametrize("ladder_x", [2, 308])
def test_edge_ladders_keep_player_in_bounds_and_restore_during_climb(pack, ladder_x):
    content = pack.to_dict()
    definition = content["maps"][0]
    definition["ladders"][0]["x"] = ladder_x
    definition["platforms"][1]["x1"] = 0 if ladder_x == 2 else 256
    definition["platforms"][1]["x2"] = 64 if ladder_x == 2 else 320
    edge_pack = MapPack.from_dict(content)
    state = Simulator(map_id=edge_pack.entry_map, map_pack=edge_pack).save()
    state["player"]["x"] = float(ladder_x - 2)
    state["maps"][edge_pack.entry_map]["monsters"] = []
    sim = Simulator.restore(state)
    for _ in range(20):
        result = sim.step(SimAction("climb", -1))
        assert result.accepted
        assert result.snapshot.player.x == ladder_x - 2
        assert 0 <= result.snapshot.player.x <= 306
        sim = Simulator.restore(json.loads(json.dumps(sim.save())))
        assert sim.snapshot() == result.snapshot
    assert sim.snapshot().player.grounded
    assert sim.snapshot().player.y == edge_pack.get(edge_pack.entry_map).ladders[0].y - 22


def test_external_marker_perception_observes_only_pixels_and_static_colors(pack):
    markers = dict(pack.markers)
    perception = PixelPerception(map_markers=markers)
    markers.clear()
    for map_id in pack.map_ids:
        frame = SceneRenderer(map_pack=pack).render(
            Simulator(map_id=map_id, map_pack=pack).snapshot()
        )
        observation = perception.observe(frame)
        assert observation.status is Status.KNOWN
        assert observation.map_id == map_id
        assert observation.hp_ratio == observation.mp_ratio == 1
        assert PixelPerception().observe(frame).map_id is None
        assert perception.observe_hud(frame).detections == ()
    with pytest.raises(TypeError):
        perception.map_markers[pack.entry_map] = (1, 2, 3)
    assert not hasattr(perception, "map_pack")
    assert not hasattr(perception, "platforms")


@pytest.mark.parametrize("mode", ["hash", "content", "missing", "maps", "profile"])
def test_restore_rejects_inconsistent_external_map_evidence(pack, mode):
    sim = Simulator(map_id=pack.entry_map, map_pack=pack, class_spec=CLASS_SPECS["assassin"])
    state = copy.deepcopy(sim.save())
    if mode == "hash":
        state["map_pack"]["content_sha256"] = "0" * 64
    elif mode == "content":
        state["map_pack"]["content"]["route"].reverse()
    elif mode == "missing":
        del state["map_pack"]
    elif mode == "maps":
        del state["maps"][pack.map_ids[-1]]
    else:
        state["class_profile"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        Simulator.restore(state)


def test_legacy_schemas_cannot_smuggle_map_data(pack):
    for schema in (1, 2):
        state = Simulator(class_spec=CLASS_SPECS["assassin"] if schema == 2 else None).save()
        state["map_pack"] = pack.manifest()
        with pytest.raises(ValueError):
            Simulator.restore(state)


def test_pack_renderer_rejects_unknown_map_and_incompatible_viewport(pack):
    with pytest.raises(ValueError, match="unknown fixture map"):
        Simulator(map_pack=pack)
    with pytest.raises(ValueError, match="320 x 192"):
        SceneRenderer(width=160, map_pack=pack)
    with pytest.raises(ValueError, match="does not know"):
        SceneRenderer(map_pack=pack).render(Simulator().snapshot())
    with pytest.raises(ValueError, match="does not know"):
        SceneRenderer().render(Simulator(map_id=pack.entry_map, map_pack=pack).snapshot())
    snapshot = Simulator(map_id=pack.entry_map, map_pack=pack).snapshot()
    with pytest.raises(ValueError, match="320 x 192"):
        SceneRenderer(map_pack=pack).render(replace(snapshot, width=640))
    for constructor in (Simulator, SceneRenderer):
        with pytest.raises(ValueError, match="verified MapPack"):
            constructor(map_pack={})


@pytest.mark.parametrize(
    "markers",
    [
        {},
        {"": (1, 2, 3)},
        {"one": (True, 2, 3)},
        {"one": (256, 2, 3)},
        {"one": [1, 2, 3]},
        {"one": (1, 2, 3), "two": (1, 2, 3)},
        {"one": PALETTE["monster"]},
        {str(index): (index, 0, 0) for index in range(9)},
    ],
)
def test_pixel_perception_rejects_ambiguous_or_malformed_markers(markers):
    with pytest.raises(ValueError):
        PixelPerception(map_markers=markers)
