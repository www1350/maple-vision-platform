"""Reproduce navigation witnesses and execute the local route in the unchanged simulator."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from maple_vision.catalog import canonical_json
from maple_vision.map_pack import load_map_pack
from maple_vision.navigation_graph import (
    MotionState,
    PlatformGeometry,
    build_platform_graph,
    load_movement_capabilities,
    load_platform_geometry,
    load_world_graph,
    simulate_actions,
)
from maple_vision.sim_types import CLASS_SPECS
from maple_vision.simulation import Simulator


def verify_fixture() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    world = load_world_graph(root / "branch-world.json")
    route = world.shortest_path("camp", "lookout")
    if route is None or route.total_cost != 4:
        raise ValueError("the authored branch fixture did not reproduce its expected route")
    world_check = world.verify_route(route)
    pack = load_map_pack(root / "simulator-maps.json")
    geometry = load_platform_geometry(root / "platform-geometry.json")
    if geometry != PlatformGeometry.from_map_pack(pack, "field"):
        raise ValueError("platform fixture is not bound to the archived MapPack")
    capabilities = load_movement_capabilities(root / "movement.json")
    graph = build_platform_graph(geometry, capabilities)
    start = MotionState(24, 146)
    result = graph.find_path(start, "platform_2")
    if result.status != "found" or result.path is None:
        raise ValueError(f"navigation fixture did not find an executable route: {result.reason}")
    path = result.path
    verification = graph.verify_path(path, initial_state=start)
    expected = simulate_actions(geometry, capabilities, start, path.actions)
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
    for index, action in enumerate(path.actions):
        outcome = sim.step(action)
        actual = sim.save()["player"]
        predicted = expected.states[index + 1].to_dict()
        if not outcome.accepted or {key: actual[key] for key in predicted} != predicted:
            raise ValueError(
                f"unchanged simulator disagrees with motion witness at step {index + 1}"
            )
    final_state = sim.save()
    return {
        "schema_version": 1,
        "scope": "synthetic_fixture",
        "world_route": route.to_dict(),
        "world_verification": world_check,
        "world_diagnostics": world.diagnostics("camp").to_dict(),
        "platform_search": result.to_dict(),
        "platform_verification": verification,
        "simulator_executed_steps": sim.snapshot().sequence,
        "simulator_all_motion_states_matched": True,
        "simulator_final_state_sha256": hashlib.sha256(
            canonical_json(final_state).encode()
        ).hexdigest(),
        "world_branch_execution_supported_by_existing_simulator": False,
        "dynamic_obstacles_evaluated": False,
        "real_domain_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="optional new JSON report file; never overwrite"
    )
    args = parser.parse_args()
    report = canonical_json(verify_fixture()) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
