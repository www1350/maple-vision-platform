from pathlib import Path

import pytest

from maple_vision.lab import RunConfig, run_episode, verify_perception
from maple_vision.map_pack import load_map_pack
from maple_vision.mission_program import load_mission_program
from maple_vision.navigation_graph import WorldMapGraph
from maple_vision.runtime_pipeline import RuntimeConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "mission,visited",
    [
        ("grove-shortcut", ["grove", "ruins"]),
        ("three-map-visits", ["grove", "quarry", "ruins"]),
    ],
)
def test_visually_selects_requested_branch_and_reinfers(tmp_path, mission, visited):
    pack = load_map_pack(EXAMPLES / "maps/branch-grove.json")
    graph = WorldMapGraph.from_map_pack(pack)
    route = graph.shortest_path("grove", "ruins")
    assert route is not None and route.portals[0].portal_id == "shortcut"
    program = load_mission_program(EXAMPLES / "missions" / (mission + ".json"))
    root = tmp_path / "run"
    result = run_episode(root, RunConfig(), runtime=RuntimeConfig(mission=program), map_pack=pack)
    assert result["status"] == "completed" and result["maps_visited"] == visited
    assert result["events"]["map_changed"] == len(visited) - 1
    assert verify_perception(root)["verified"]
