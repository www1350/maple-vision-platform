"""Independent topology, hand-calculated kinematics and actual simulator execution."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import maple_vision.navigation_graph as navigation
from maple_vision.catalog import canonical_json
from maple_vision.map_pack import MapDefinition, MapPack
from maple_vision.navigation_graph import (
    GraphProvenance,
    MotionState,
    MovementCapabilities,
    NavigationLadder,
    NavigationPath,
    PlatformGeometry,
    PlatformGraph,
    PlatformSurface,
    SearchBudget,
    WorldMapGraph,
    WorldPortal,
    build_platform_graph,
    load_platform_geometry,
    load_world_graph,
    simulate_actions,
)
from maple_vision.sim_types import CLASS_SPECS, NPC, Ladder, Platform, Portal, SimAction
from maple_vision.simulation import Simulator


def _provenance():
    return GraphProvenance(
        "urn:maple-vision:self-authored-navigation", "self_authored", "CC0-1.0", "fixture-v1"
    )


def _world():
    # Cheap branch beats the direct door; return direction is deliberately absent.
    return WorldMapGraph(
        "branch_fixture",
        ("start", "left", "right", "goal", "island"),
        (
            WorldPortal("start", "direct", "goal", 9),
            WorldPortal("start", "left", "left", 2),
            WorldPortal("start", "right", "right", 2),
            WorldPortal("left", "finish", "goal", 2),
            WorldPortal("right", "finish", "goal", 2),
        ),
        _provenance(),
    )


def _geometry(*platforms, ladders=()):
    return PlatformGeometry(
        "fixture",
        (PlatformSurface("ground", 0, 320, 168), *platforms),
        ladders,
        _provenance(),
    )


def _pack():
    maps = []
    for i, (name, destination) in enumerate((("field", "camp"), ("camp", "field"))):
        maps.append(
            MapDefinition(
                id=name,
                theme="meadow" if i == 0 else "cavern",
                marker=(11 + i * 57, 29 + i * 14, 47 - i * 28),
                spawn_x=24,
                platforms=(
                    Platform(0, 320, 168),
                    Platform(70, 140, 126),
                    Platform(180, 250, 104),
                    Platform(0, 32, 72),
                ),
                portals=(Portal("exit", 290, 132, 20, 36, destination),),
                npcs=(NPC("guide", 270, 144, 14, 24),),
                ladders=(Ladder(100, 126, 10, 42),),
                monster_x=(280,),
            )
        )
    return MapPack(
        "navigation_fixture",
        "1.0.0",
        "urn:maple-vision:authored-map-fixture",
        "self_authored",
        "CC0-1.0",
        tuple(maps),
        ("field", "camp"),
    )


def _simulator(pack, start, capabilities):
    profile = replace(
        CLASS_SPECS["assassin"],
        walk_speed=capabilities.walk_speed,
        jump_speed=capabilities.jump_speed,
    )
    sim = Simulator(
        seed=614,
        map_id="field",
        step_ms=capabilities.step_ms,
        class_spec=profile,
        map_pack=pack,
    )
    state = sim.save()
    state["player"].update(start.to_dict())
    # Isolate locomotion through the existing validated save API; physics is untouched.
    for value in state["maps"].values():
        value["monsters"] = []
    return Simulator.restore(state)


def _execute_and_compare(pack, geometry, capabilities, start, actions):
    oracle = simulate_actions(geometry, capabilities, start, actions)
    sim = _simulator(pack, start, capabilities)
    for index, action in enumerate(actions):
        outcome = sim.step(action)
        actual = sim.save()["player"]
        expected = oracle.states[index + 1]
        assert outcome.accepted == oracle.accepted[index]
        assert {key: actual[key] for key in expected.to_dict()} == expected.to_dict()
    return sim


def test_world_graph_weighted_branches_directionality_and_diagnostics():
    graph = _world()
    route = graph.shortest_path("start", "goal")
    assert route is not None
    assert route.total_cost == 4
    assert [p.portal_id for p in route.portals] == ["left", "finish"]
    assert graph.shortest_path("goal", "start") is None
    assert graph.shortest_path("start", "island") is None
    same = graph.shortest_path("start", "start")
    assert same is not None and same.total_cost == 0 and same.portals == ()
    report = graph.diagnostics("start")
    assert report.reachable == ("goal", "left", "right", "start")
    assert report.unreachable == report.isolated == ("island",)
    assert report.dead_ends == ("goal", "island")


def test_world_tie_prefers_fewer_doors_then_stable_portal_sequence():
    graph = _world()
    graph = replace(graph, portals=graph.portals + (WorldPortal("start", "z_shortcut", "goal", 4),))
    route = graph.shortest_path("start", "goal")
    assert route is not None and len(route.portals) == 1
    assert route.portals[0].portal_id == "z_shortcut"
    reordered = replace(
        graph, nodes=tuple(reversed(graph.nodes)), portals=tuple(reversed(graph.portals))
    )
    assert reordered == graph
    assert reordered.content_sha256 == graph.content_sha256
    assert reordered.shortest_path("start", "goal") == route


def test_world_cycle_does_not_loop_or_invent_reverse_edges():
    graph = replace(_world(), portals=_world().portals + (WorldPortal("goal", "back", "left"),))
    assert graph.shortest_path("goal", "start") is None
    route = graph.shortest_path("goal", "left")
    assert route is not None and route.total_cost == 1
    assert graph.diagnostics("goal").reachable == ("goal", "left")


@pytest.mark.parametrize("change", ["duplicate-node", "duplicate-door", "bad-target", "bad-source"])
def test_world_rejects_identity_and_foreign_key_conflicts(change):
    graph = _world()
    nodes, portals = graph.nodes, graph.portals
    if change == "duplicate-node":
        nodes += (nodes[0],)
    elif change == "duplicate-door":
        portals += (replace(portals[0], target="island"),)
    else:
        portals += (
            WorldPortal(
                "missing" if change == "bad-source" else "start",
                "new",
                "missing" if change == "bad-target" else "goal",
            ),
        )
    with pytest.raises(ValueError):
        replace(graph, nodes=nodes, portals=portals)


def test_map_pack_adaptation_preserves_only_existing_cycle_and_source_binding():
    pack = _pack()
    world = WorldMapGraph.from_map_pack(pack)
    assert len(world.portals) == 2
    assert [(p.source, p.portal_id, p.target) for p in world.portals] == [
        ("camp", "exit", "field"),
        ("field", "exit", "camp"),
    ]
    assert world.provenance.source_sha256 == pack.content_sha256
    geometry = PlatformGeometry.from_map_pack(pack, "field")
    assert geometry.provenance.source_sha256 == pack.content_sha256
    assert geometry.platform_ids == ("platform_0", "platform_1", "platform_2", "platform_3")
    assert geometry.ladders[0] == NavigationLadder("ladder_0", 100, 126, 42)
    assert _world().shortest_path("start", "right") is not None
    # A branched WorldMapGraph is independent data, never masquerading as a MapPack.
    with pytest.raises(ValueError):
        MapPack.from_dict(_world().to_dict())


def test_strict_world_and_geometry_roundtrip_and_hash_binding(tmp_path):
    for value, filename, loader, parser in (
        (_world(), "world.json", load_world_graph, WorldMapGraph.from_manifest),
        (
            _geometry(PlatformSurface("shelf", 80, 120, 104)),
            "terrain.json",
            load_platform_geometry,
            PlatformGeometry.from_manifest,
        ),
    ):
        path = tmp_path / filename
        path.write_text(canonical_json(value.to_dict()))
        assert loader(path) == value
        path.write_text(canonical_json(value.manifest()))
        assert loader(path) == value
        manifest = value.manifest()
        manifest["content"]["provenance"]["game_version"] = "another-version"
        with pytest.raises(ValueError, match="SHA256"):
            parser(manifest)
        manifest["content_sha256"] = hashlib.sha256(
            canonical_json(manifest["content"]).encode()
        ).hexdigest()
        revised = parser(manifest)
        assert revised.content_sha256 != value.content_sha256
        assert revised.provenance.game_version == "another-version"


@pytest.mark.parametrize("document", ["world", "geometry"])
def test_graph_load_is_bounded_before_json_parse(tmp_path, monkeypatch, document):
    path = tmp_path / "graph.json"
    source, loader = (
        (_world(), load_world_graph)
        if document == "world"
        else (_geometry(), load_platform_geometry)
    )
    path.write_text(canonical_json(source.to_dict()))
    monkeypatch.setattr(navigation, "MAX_GRAPH_BYTES", 16)
    monkeypatch.setattr(
        navigation, "strict_json", lambda _: pytest.fail("oversized input reached JSON parser")
    )
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("unbounded file read"))
    with pytest.raises(ValueError, match="budget"):
        loader(path)


def test_world_array_count_guard_precedes_portal_construction(monkeypatch):
    data = _world().to_dict()
    data["nodes"] = ["a"] * 129
    monkeypatch.setattr(
        WorldPortal, "from_dict", lambda _: pytest.fail("oversized nodes reached portal parsing")
    )
    with pytest.raises(ValueError, match="world nodes"):
        WorldMapGraph.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("walk_speed", True),
        ("walk_speed", float("nan")),
        ("jump_speed", 401),
        ("step_ms", 0),
        ("can_climb", 1),
        ("can_drop", "true"),
    ],
)
def test_capabilities_reject_invalid_types_and_ranges(field, value):
    with pytest.raises(ValueError):
        MovementCapabilities(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_expanded_states", True),
        ("max_generated_edges", 0),
        ("max_simulated_steps", 2_000_001),
        ("max_primitive_steps", float("nan")),
        ("max_path_steps", 65537),
    ],
)
def test_search_budget_is_strict_and_bounded(field, value):
    with pytest.raises(ValueError):
        SearchBudget(**{field: value})


@pytest.mark.parametrize(
    "arguments",
    [
        {"x": True, "y": 146},
        {"x": 10**1000, "y": 146},
        {"x": 0, "y": float("nan")},
        {"x": 0, "y": 146, "grounded": 1},
        {"x": 0, "y": 146, "vy": 1},
        {"x": 0, "y": 146, "climbing": True},
        {"x": 0, "y": 80, "climbing": True, "grounded": False, "vx": 1},
    ],
)
def test_motion_state_rejects_nonfinite_or_inconsistent_preconditions(arguments):
    with pytest.raises(ValueError):
        MotionState(**arguments)


def test_immutable_documents_have_detached_serialization():
    graph = build_platform_graph(_geometry(), MovementCapabilities())
    raw = graph.to_dict()
    assert PlatformGraph.from_dict(raw) == graph
    raw["geometry"]["content"]["platforms"][0]["x1"] = 100
    assert graph.geometry.platforms[0].x1 == 0
    with pytest.raises(FrozenInstanceError):
        graph.capabilities.walk_speed = 1
    with pytest.raises(ValueError):
        PlatformGraph.from_dict(raw)
    with pytest.raises(ValueError):
        replace(graph.geometry, platforms=list(graph.geometry.platforms))


@pytest.mark.parametrize(
    "platforms,ladders",
    [
        ((PlatformSurface("ground", 0, 319, 168),), ()),
        ((PlatformSurface("ground", 0, 320, 168), PlatformSurface("ground", 20, 70, 100)), ()),
        (
            (
                PlatformSurface("ground", 0, 320, 168),
                PlatformSurface("a", 20, 70, 100),
                PlatformSurface("b", 70, 90, 100),
            ),
            (),
        ),
        ((PlatformSurface("ground", 0, 320, 168),), (NavigationLadder("ladder", 100, 100, 68),)),
    ],
)
def test_geometry_rejects_unsupported_collision_and_ladder_layouts(platforms, ladders):
    with pytest.raises(ValueError):
        PlatformGeometry("fixture", platforms, ladders, _provenance())


def test_hand_calculated_vertical_jump_rises_53_5_pixels_and_passes_one_way_floor():
    geometry = _geometry(PlatformSurface("shelf", 70, 140, 126))
    start = MotionState(90, 146)
    actions = (SimAction("jump"),) + (SimAction(),) * 11
    trace = simulate_actions(geometry, MovementCapabilities(), start, actions)
    # Discrete Euler: feet(n) = 168 - 14*n + 0.8125*n*(n+1).
    # n=8 is the apex; the shelf is crossed upward at n=5 and landed on at n=12.
    assert trace.states[5].y + 22 == 122.375 and not trace.states[5].grounded
    assert trace.states[8].y + 22 == 114.5
    assert trace.states[8].vy == -20
    assert trace.states[9].vy == 12.5
    assert trace.states[11].grounded is False
    assert trace.states[12] == MotionState(90, 104)
    assert all(trace.accepted)
    actual = _execute_and_compare(
        _pack(),
        PlatformGeometry.from_map_pack(_pack(), "field"),
        MovementCapabilities(),
        start,
        actions,
    )
    assert actual.snapshot().player.y == 104


def test_ceiling_collision_resets_velocity_and_matches_actual_fixture_simulator():
    pack = _pack()
    geometry = PlatformGeometry.from_map_pack(pack, "field")
    caps = MovementCapabilities(jump_speed=400)
    start = MotionState(8, 50)
    actions = (SimAction("jump"),) + (SimAction(),) * 10
    trace = simulate_actions(geometry, caps, start, actions)
    # y=50 ->31.625 ->14.875 ->-0.25 before clamp at the world's ceiling.
    assert [state.y for state in trace.states[:4]] == [50, 31.625, 14.875, 0]
    assert trace.states[3].vy == 0
    assert trace.states[4].y == 1.625 and trace.states[4].vy == 32.5
    assert trace.states[-1].grounded and trace.states[-1].y == 50
    _execute_and_compare(pack, geometry, caps, start, actions)


def test_fast_descent_sweeps_all_platform_tops_instead_of_tunneling():
    pack = _pack()
    # Descending over the elevated right platform at y=104, before ground y=168.
    geometry = PlatformGeometry.from_map_pack(pack, "field")
    caps = MovementCapabilities(step_ms=100)
    start = MotionState(200, 20, vy=1000, grounded=False)
    trace = simulate_actions(geometry, caps, start, (SimAction(),))
    # Free movement would put feet at 148.5. The sweep must land at 104 instead.
    assert trace.states[-1] == MotionState(200, 82)
    _execute_and_compare(pack, geometry, caps, start, (SimAction(),))


def test_walk_drop_edge_has_actual_one_way_reachability_and_no_upward_teleport():
    geometry = _geometry(PlatformSurface("shelf", 80, 120, 104))
    graph = build_platform_graph(geometry, MovementCapabilities(can_jump=False, can_climb=False))
    top = MotionState(119, 82)
    down = graph.find_path(top, "ground")
    assert down.status == "found" and down.path is not None
    assert [edge.kind for edge in down.path.edges] == ["drop"]
    assert down.path.end.y == 146
    assert down.path.actions[0] == SimAction("move", 1)
    assert graph.verify_path(down.path)["verified"] is True
    up = graph.find_path(down.path.end, "shelf")
    assert up.status == "unreachable_in_primitive_graph" and up.path is None
    assert up.reachable_platforms == ("ground",)


def test_drop_disabled_does_not_publish_a_falling_edge():
    geometry = _geometry(PlatformSurface("shelf", 80, 120, 104))
    graph = build_platform_graph(
        geometry, MovementCapabilities(can_jump=False, can_climb=False, can_drop=False)
    )
    expansion = graph.edges_from(MotionState(119, 82))
    assert {edge.kind for edge in expansion.edges} == {"walk"}
    assert all(edge.end.y == 82 for edge in expansion.edges)


def test_jump_capacity_changes_reachable_graph_without_changing_geometry():
    geometry = _geometry(PlatformSurface("shelf", 80, 120, 104))
    start = MotionState(90, 146)
    low = build_platform_graph(geometry, MovementCapabilities(can_walk=False, can_climb=False))
    high = replace(low, capabilities=replace(low.capabilities, jump_speed=400))
    unavailable = low.find_path(start, "shelf")
    assert unavailable.status == "unreachable_in_primitive_graph"
    available = high.find_path(start, "shelf")
    assert available.status == "found" and available.path is not None
    assert high.geometry is low.geometry
    assert high.content_sha256 != low.content_sha256
    assert high.verify_path(available.path)["verified"] is True
    with pytest.raises(ValueError, match="bound"):
        low.verify_path(available.path)


def test_ladder_up_down_and_mid_ladder_start_have_hand_calculated_endpoints():
    geometry = _geometry(
        PlatformSurface("shelf", 80, 140, 72),
        ladders=(NavigationLadder("ladder", 100, 72, 96),),
    )
    graph = build_platform_graph(geometry, MovementCapabilities(can_walk=False, can_jump=False))
    start = MotionState(98, 146)
    up = graph.find_path(start, "shelf")
    assert up.path is not None and up.path.steps == 32  # 96 pixels / 3 pixels per step.
    assert up.path.end == MotionState(98, 50)
    assert {action for action in up.path.actions} == {SimAction("climb", -1)}
    down = graph.find_path(up.path.end, "ground")
    assert down.path is not None and down.path.steps == 32 and down.path.end == start
    mid = MotionState(98, 98, grounded=False, climbing=True)
    rest = graph.find_path(mid, "shelf")
    assert rest.path is not None and rest.path.steps == 16 and rest.path.initial_state == mid
    assert graph.verify_path(rest.path, initial_state=mid)["verified"] is True


def test_ladder_acquisition_uses_declared_first_match_order():
    geometry = _geometry(
        PlatformSurface("low", 70, 150, 126),
        PlatformSurface("high", 70, 150, 104),
        ladders=(
            NavigationLadder("low_ladder", 100, 126, 42),
            NavigationLadder("high_ladder", 105, 104, 64),
        ),
    )
    graph = build_platform_graph(geometry, MovementCapabilities(can_walk=False, can_jump=False))
    edges = graph.edges_from(MotionState(100, 146)).edges
    assert [edge.target_platforms for edge in edges] == [("low",), ("ground",)]
    assert edges[0].primitives[0].action == SimAction("climb", -1)
    assert edges[1].end == MotionState(98, 146)  # Acquiring down snaps to the first ladder.
    reversed_geometry = replace(geometry, ladders=tuple(reversed(geometry.ladders)))
    reversed_edges = (
        replace(graph, geometry=reversed_geometry).edges_from(MotionState(100, 146)).edges
    )
    assert [edge.target_platforms for edge in reversed_edges] == [("high",), ("ground",)]
    assert reversed_edges[1].end == MotionState(103, 146)
    assert geometry.content_sha256 != reversed_geometry.content_sha256


def test_actual_map_pack_ladder_and_reachable_route_execute_every_step():
    pack = _pack()
    geometry = PlatformGeometry.from_map_pack(pack, "field")
    caps = MovementCapabilities()
    start = MotionState(24, 146)
    graph = build_platform_graph(geometry, caps)
    found = graph.find_path(start, "platform_2")
    assert found.status == "found" and found.path is not None
    path = found.path
    assert any(edge.kind == "jump" for edge in path.edges)
    assert any(edge.kind == "walk" for edge in path.edges)
    assert len({p for edge in path.edges for p in edge.target_platforms}) >= 3
    assert all(left.end == right.start for left, right in zip(path.edges, path.edges[1:]))
    assert graph.verify_path(path, initial_state=start)["verified"] is True
    actual = _execute_and_compare(pack, geometry, caps, start, path.actions)
    assert actual.snapshot().player.y == 82
    # A separate no-jump profile forces the actual ladder path to the lower ledge.
    climb_caps = replace(caps, can_jump=False)
    climb_graph = replace(graph, capabilities=climb_caps)
    climb = climb_graph.find_path(MotionState(98, 146), "platform_1")
    assert climb.path is not None and climb.path.steps == 14
    _execute_and_compare(pack, geometry, climb_caps, climb.path.initial_state, climb.path.actions)
    back = climb_graph.find_path(climb.path.end, "platform_0")
    assert back.path is not None
    _execute_and_compare(pack, geometry, climb_caps, back.path.initial_state, back.path.actions)


def test_existing_velocity_and_airborne_start_are_bound_and_adapted():
    geometry = _geometry(PlatformSurface("shelf", 80, 120, 126))
    graph = build_platform_graph(geometry, MovementCapabilities(can_climb=False))
    moving = MotionState(24, 146, vx=80)
    expanded = graph.edges_from(moving)
    assert len(expanded.edges) == 1
    assert expanded.edges[0].kind == "settle"
    assert expanded.edges[0].start == moving and expanded.edges[0].end == MotionState(24, 146)
    airborne = MotionState(40, 100, vx=80, vy=50, grounded=False)
    found = graph.find_path(airborne, "ground")
    assert found.path is not None and found.path.initial_state == airborne
    assert found.path.edges[0].kind == "settle"
    assert graph.verify_path(found.path, initial_state=airborne)["verified"] is True
    with pytest.raises(ValueError, match="precondition"):
        graph.verify_path(found.path, initial_state=replace(airborne, vx=0))


@pytest.mark.parametrize("speed,jump,step_ms", [(37, 351, 33), (120, 240, 10), (61, 400, 75)])
def test_fixed_step_motion_matches_existing_simulator_for_fractional_profiles(speed, jump, step_ms):
    pack = _pack()
    geometry = PlatformGeometry.from_map_pack(pack, "field")
    capabilities = MovementCapabilities(walk_speed=speed, jump_speed=jump, step_ms=step_ms)
    actions = (
        (SimAction("move", 1),) * 5
        + (SimAction("jump", 1),)
        + (SimAction(),) * 40
        + (SimAction("move", -1),) * 3
        + (SimAction(),)
    )
    _execute_and_compare(pack, geometry, capabilities, MotionState(24, 146), actions)


@pytest.mark.parametrize(
    "budget",
    [
        SearchBudget(max_expanded_states=1),
        SearchBudget(max_generated_edges=1),
        SearchBudget(max_simulated_steps=1),
        SearchBudget(max_primitive_steps=1),
        SearchBudget(max_path_steps=1),
    ],
)
def test_budget_exhaustion_is_unknown_not_physical_unreachability(budget):
    geometry = _geometry(PlatformSurface("shelf", 80, 120, 126))
    graph = build_platform_graph(geometry, MovementCapabilities(can_climb=False), budget)
    result = graph.find_path(MotionState(24, 146), "shelf")
    assert result.status == "unknown_budget" and result.path is None
    assert result.incomplete_primitives
    assert result.expanded_states <= budget.max_expanded_states
    assert result.generated_edges <= budget.max_generated_edges
    assert result.simulated_steps <= budget.max_simulated_steps


def test_shortest_path_is_repeatedly_identical_and_json_roundtrips():
    geometry = _geometry(
        PlatformSurface("shelf", 80, 140, 126), ladders=(NavigationLadder("ladder", 100, 126, 42),)
    )
    graph = build_platform_graph(geometry, MovementCapabilities(can_jump=False))
    first = graph.find_path(MotionState(94, 146), "shelf")
    assert first.path is not None and first.optimal_within_primitive_graph
    assert graph.find_path(MotionState(94, 146), "shelf").to_dict() == first.to_dict()
    restored = NavigationPath.from_dict(json.loads(canonical_json(first.path.to_dict())))
    assert restored == first.path
    assert graph.verify_path(restored)["verified"] is True


def test_hash_repair_cannot_hide_motion_or_kind_tampering():
    graph = build_platform_graph(
        _geometry(PlatformSurface("shelf", 80, 120, 104)),
        MovementCapabilities(can_jump=False, can_climb=False),
    )
    found = graph.find_path(MotionState(119, 82), "ground")
    assert found.path is not None
    for field, replacement in (
        ("kind", "walk"),
        ("trajectory_sha256", "0" * 64),
        ("target_platforms", ["shelf"]),
    ):
        payload = copy.deepcopy(found.path.to_dict())
        payload["edges"][0][field] = replacement
        altered = NavigationPath.from_dict(payload)
        assert altered.content_sha256 != found.path.content_sha256
        with pytest.raises(ValueError, match="reproduce"):
            graph.verify_path(altered)


def test_path_cannot_combine_different_exact_motion_preconditions():
    graph = build_platform_graph(_geometry(), MovementCapabilities(can_jump=False, can_climb=False))
    start = MotionState(24, 146)
    left, right = graph.edges_from(start).edges
    with pytest.raises(ValueError, match="incompatible"):
        NavigationPath(graph.content_sha256, "fixture", start, "ground", (left, right))


def test_invalid_ground_or_ladder_precondition_is_rejected_before_search():
    graph = build_platform_graph(_geometry(), MovementCapabilities())
    with pytest.raises(ValueError, match="supporting"):
        graph.find_path(MotionState(24, 100), "ground")
    with pytest.raises(ValueError, match="ladder"):
        graph.find_path(MotionState(24, 100, grounded=False, climbing=True), "ground")
    with pytest.raises(ValueError, match="authored"):
        graph.find_path(MotionState(24, 146), "missing")


def test_replay_does_not_accept_combat_actions_or_disabled_movement():
    geometry = _geometry()
    with pytest.raises(ValueError, match="only"):
        simulate_actions(
            geometry, MovementCapabilities(), MotionState(24, 146), (SimAction("attack"),)
        )
    with pytest.raises(ValueError, match="disabled"):
        simulate_actions(
            geometry,
            MovementCapabilities(can_walk=False),
            MotionState(24, 146),
            (SimAction("move", 1),),
        )


def test_no_geometry_module_reads_live_simulator_or_runtime_observation():
    # The only simulator access in this package is the independent oracle in this test file.
    source = Path(navigation.__file__).read_text()
    assert "from .simulation import" not in source
    assert "VisualObservation" not in source and "SimSnapshot" not in source


def test_navigation_cli_entry_points(capsys):
    from maple_vision.cli import main

    root = Path(__file__).resolve().parents[1] / "examples" / "navigation"
    assert main(["maps", "route", str(root / "branch-world.json"), "camp", "lookout"]) == 0
    assert json.loads(capsys.readouterr().out)["route"]["total_cost"] == 4
    assert (
        main(
            [
                "maps",
                "platform-route",
                str(root / "platform-geometry.json"),
                str(root / "movement.json"),
                "--start",
                "24",
                "146",
                "--target",
                "platform_2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verification"]["steps"] == 44
