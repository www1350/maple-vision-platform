import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.class_package import load_class_package
from maple_vision.cli import main
from maple_vision.input_bindings import load_input_profile
from maple_vision.lab import (
    CapabilityRunConfig,
    RunConfig,
    _execute_episode,
    run_episode,
    verify_perception,
    verify_run,
)
from maple_vision.mission_program import (
    MissionProgram,
    MissionStage,
    MissionTransition,
    NavigationObjective,
    load_mission_program,
)
from maple_vision.mission_verification import MissionOracle
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "phases", [[], ["fly"], ["talk", "talk"], ["collect"], ["collect", "fight"]]
)
def test_reject_invalid_or_unachievable_objectives(phases):
    with pytest.raises(ValueError):
        MissionStage("meadow", tuple(phases))


def test_schema_route_and_configuration_boundaries():
    program = MissionProgram((MissionStage("meadow", ("talk",)),))
    default = RuntimeConfig()
    assert default.to_dict()["schema_version"] == 1 and "mission" not in default.to_dict()
    config = RuntimeConfig(mission=program)
    assert RuntimeConfig.from_manifest(config.manifest()) == config
    with pytest.raises(ValueError, match="prefix"):
        program.bind_route(("cavern", "meadow"))
    raw = config.manifest()
    raw["settings"]["mission"]["stages"][0]["phases"] = ["fight"]
    with pytest.raises(ValueError, match="checksum"):
        RuntimeConfig.from_manifest(raw)
    raw = default.to_dict()
    raw["mission"] = program.to_dict()
    with pytest.raises(ValueError):
        RuntimeConfig.from_dict(raw)
    with pytest.raises(ValueError):
        MissionProgram((program.stages[0], program.stages[0]))


def test_navigation_mission_schema_is_explicit_and_backward_compatible():
    legacy = MissionProgram((MissionStage("meadow", ("fight",)),))
    assert legacy.to_dict() == {
        "schema_version": 1,
        "stages": [{"map_id": "meadow", "phases": ["fight"]}],
    }
    layered = MissionProgram(
        (MissionStage("forest", ("navigate", "fight"), NavigationObjective("platform_1")),)
    )
    assert layered.to_dict()["schema_version"] == 2
    assert MissionProgram.from_dict(layered.to_dict()) == layered
    with pytest.raises(ValueError, match="requires"):
        MissionStage("forest", ("navigate",))
    with pytest.raises(ValueError, match="first"):
        MissionStage("forest", ("fight", "navigate"), NavigationObjective("platform_1"))


def test_conditional_mission_schema_is_a_bounded_forward_graph():
    program = MissionProgram(
        (
            MissionStage(
                "grove",
                ("fight",),
                transitions=(
                    MissionTransition("loot_visible", "quarry"),
                    MissionTransition("no_loot_visible", "ruins"),
                ),
            ),
            MissionStage("quarry", ("talk",), transitions=(MissionTransition("always", "ruins"),)),
            MissionStage("ruins", ("talk",)),
        )
    )
    assert program.to_dict()["schema_version"] == 3
    assert MissionProgram.from_dict(program.to_dict()) == program
    assert program.transition("grove", loot_visible=True).target_map == "quarry"
    assert program.transition("grove", loot_visible=False).target_map == "ruins"
    assert program.transition("ruins", loot_visible=False) is None
    with pytest.raises(ValueError, match="cover"):
        MissionProgram(
            (
                MissionStage(
                    "grove",
                    ("talk",),
                    transitions=(
                        MissionTransition("always", "quarry"),
                        MissionTransition("loot_visible", "ruins"),
                    ),
                ),
                MissionStage("quarry", ("talk",)),
                MissionStage("ruins", ("talk",)),
            )
        )
    with pytest.raises(ValueError, match="later"):
        MissionProgram(
            (
                MissionStage(
                    "grove",
                    ("talk",),
                    transitions=(MissionTransition("always", "ruins"),),
                ),
                MissionStage("ruins", ("talk",)),
                MissionStage(
                    "quarry",
                    ("talk",),
                    transitions=(MissionTransition("always", "grove"),),
                ),
            )
        )


def test_one_conditional_program_selects_both_visual_branches_and_replays(tmp_path):
    from maple_vision.map_pack import load_map_pack

    pack = load_map_pack(EXAMPLES / "maps/branch-grove.json")
    program = load_mission_program(EXAMPLES / "missions/conditional-grove.json")
    normal_root = tmp_path / "with-loot"
    normal = run_episode(
        normal_root,
        RunConfig(),
        map_pack=pack,
        runtime=RuntimeConfig(mission=program),
    )
    assert normal["status"] == "completed" and normal["independent_goal_check"]
    assert normal["maps_visited"] == ["grove", "quarry", "ruins"]
    branches = normal["mission_evidence"]["branches"]
    assert [(row["source_map"], row["condition"], row["target_map"]) for row in branches] == [
        ("grove", "loot_visible", "quarry"),
        ("quarry", "always", "ruins"),
    ]
    assert [row["sequence"] for row in branches] == sorted(row["sequence"] for row in branches)
    decisions = normal["runtime_evidence"]["terminal"]["state"]["mission_branching"]["decisions"]
    assert [(row["condition"], row["target_map"]) for row in decisions] == [
        ("loot_visible", "quarry"),
        ("always", "ruins"),
    ]
    assert verify_run(normal_root)["verified"]
    assert verify_perception(normal_root)["verified"]

    initial = Simulator(map_id=pack.entry_map, map_pack=pack).save()
    initial["maps"]["grove"]["monsters"] = []
    empty = _execute_episode(
        config=RunConfig(),
        initial_state=initial,
        map_pack=pack,
        runtime=RuntimeConfig(mission=program),
    )
    assert empty["status"] == "completed" and empty["independent_goal_check"]
    assert empty["maps_visited"] == ["grove", "ruins"]
    assert [
        (row["condition"], row["target_map"]) for row in empty["mission_evidence"]["branches"]
    ] == [("no_loot_visible", "ruins")]
    empty_decisions = empty["runtime_evidence"]["terminal"]["state"]["mission_branching"][
        "decisions"
    ]
    assert [(row["condition"], row["target_map"]) for row in empty_decisions] == [
        ("no_loot_visible", "ruins")
    ]

    summary_path = normal_root / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary["mission_evidence"]["branches"][0]["target_map"] = "ruins"
    summary_path.write_text(json.dumps(summary))
    manifest_path = normal_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="witnesses"):
        verify_run(normal_root)


def test_visual_loot_false_negative_is_rejected_by_truth_branch_and_reinfers(tmp_path):
    from maple_vision.map_pack import load_map_pack

    root = tmp_path / "loot-false-negative"
    result = run_episode(
        root,
        RunConfig(fault="loot_false_negative"),
        map_pack=load_map_pack(EXAMPLES / "maps/branch-grove.json"),
        runtime=RuntimeConfig(
            mission=load_mission_program(EXAMPLES / "missions/conditional-grove.json")
        ),
    )
    assert result["status"] == "failed"
    assert result["reason"] == "conditional_branch_check_failed"
    assert not result["independent_goal_check"]
    assert result["maps_visited"] == ["grove", "ruins"]
    assert result["mission_evidence"]["invalid_transition"]
    check = result["conditional_branch_check"]
    assert check == {
        "schema_version": 1,
        "scope": "runtime_visual_branch_vs_simulator_truth",
        "matched": False,
        "mismatch_index": 0,
        "runtime_path": [
            {
                "source_map": "grove",
                "condition": "no_loot_visible",
                "target_map": "ruins",
            }
        ],
        "oracle_path": [
            {
                "source_map": "grove",
                "condition": "loot_visible",
                "target_map": "quarry",
            }
        ],
    }
    replay = verify_run(root)
    assert replay["verified"] and not replay["independent_goal_check"]
    reinfer = verify_perception(root)
    assert reinfer["verified"] and reinfer["policy_reproduced"]
    assert not reinfer["independent_goal_check"]


@pytest.mark.parametrize("job", ["assassin", "dragon_knight", "cleric"])
def test_ordered_program_executes_and_replays_after_source_removal(tmp_path, job):
    source = tmp_path / "mission.json"
    source.write_bytes((EXAMPLES / "missions/talk-then-clear.json").read_bytes())
    program = load_mission_program(source)
    source.unlink()
    root = tmp_path / "run"
    result = run_episode(root, RunConfig(job=job), runtime=RuntimeConfig(mission=program))
    assert result["status"] == "completed" and result["independent_goal_check"]
    witnesses = result["mission_evidence"]["witnesses"]
    assert [(w["map_id"], w["phase"]) for w in witnesses] == [
        ("meadow", "talk"),
        ("meadow", "fight"),
        ("meadow", "collect"),
        ("cavern", "talk"),
    ]
    assert [w["sequence"] for w in witnesses] == sorted(w["sequence"] for w in witnesses)
    assert result["events"]["monster_defeated"] == 3
    assert result["drops_collected"] == 3
    assert verify_run(root)["verified"] and verify_perception(root)["verified"]
    summary = json.loads((root / "summary.json").read_bytes())
    summary["mission_evidence"]["witnesses"][0]["phase"] = "fight"
    (root / "summary.json").write_text(json.dumps(summary))
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"]["summary.json"] = hashlib.sha256(
        (root / "summary.json").read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="witnesses"):
        verify_run(root)


def test_cli_single_map_program_and_wrong_map_preflight(tmp_path):
    source = tmp_path / "mission.json"
    source.write_text(
        json.dumps({"schema_version": 1, "stages": [{"map_id": "meadow", "phases": ["talk"]}]})
    )
    root = tmp_path / "run"
    assert main(["lab", "run", "--mission-program", str(source), "--output", str(root)]) == 0
    result = json.loads((root / "summary.json").read_bytes())
    assert result["maps_visited"] == ["meadow"] and result["drops_collected"] == 0
    assert result["independent_goal_check"]
    assert (
        main(
            [
                "lab",
                "run",
                "--mission-program",
                str(source),
                "--runtime",
                "legacy",
                "--output",
                str(tmp_path / "legacy"),
            ]
        )
        == 2
    )
    source.write_text(
        json.dumps({"schema_version": 1, "stages": [{"map_id": "cavern", "phases": ["talk"]}]})
    )
    assert (
        main(["lab", "run", "--mission-program", str(source), "--output", str(tmp_path / "wrong")])
        == 2
    )
    assert not (tmp_path / "wrong").exists()


def test_oracle_rejects_missing_dialogue_and_premature_map_change():
    program = MissionProgram((MissionStage("meadow", ("talk", "fight")),))
    simulator = Simulator()
    before = simulator.snapshot()
    oracle = MissionOracle(program)
    # A scene clearing before the required dialogue cannot satisfy the ordered mission.
    cleared = replace(before, monsters=())
    oracle.observe(before, cleared, ())
    assert oracle.result(cleared)["witnesses"] == []
    oracle.observe(before, replace(cleared, map_id="cavern"), ("map_changed",))
    assert oracle.result(cleared)["invalid_transition"]
    assert not oracle.result(cleared)["completed"]


def test_oracle_checks_truth_while_visual_failures_remain_failures(tmp_path):
    program = load_mission_program(EXAMPLES / "missions/talk-then-clear.json")
    result = run_episode(
        tmp_path / "fault", RunConfig(fault="freeze_input"), runtime=RuntimeConfig(mission=program)
    )
    assert result["status"] != "completed" and not result["independent_goal_check"]
    assert verify_run(tmp_path / "fault")["verified"]
    assert verify_perception(tmp_path / "fault")["verified"]


def test_talk_completion_does_not_imply_clearing():
    program = MissionProgram((MissionStage("meadow", ("talk",)),))
    simulator = Simulator()
    before = simulator.snapshot()
    after = simulator.step(SimAction("interact"))
    oracle = MissionOracle(program)
    oracle.observe(before, after.snapshot, after.events)
    assert after.snapshot.monsters and oracle.result(after.snapshot)["completed"]


@pytest.mark.parametrize("fault", ["none", "freeze_input"])
def test_two_forest_combat_drill(tmp_path, fault):
    from maple_vision.map_pack import load_map_pack

    root = tmp_path / "forest"
    program = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    result = run_episode(
        root,
        RunConfig(job="dragon_knight", fault=fault),
        map_pack=pack,
        runtime=RuntimeConfig(mission=program),
    )
    if fault == "freeze_input":
        assert result["status"] != "completed"
        assert not result["mission_evidence"]["completed"]
        assert result["events"].get("monster_defeated", 0) == 0
        return
    assert result["status"] == "completed"
    rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    milestones = []
    for row in rows:
        for event in row["execution"]["events"]:
            if event in {"monster_defeated", "map_changed"}:
                milestones.append((row["observation"]["map_id"], event))
    assert milestones == [
        ("drill_forest_1", "monster_defeated"),
        ("drill_forest_1", "map_changed"),
        ("drill_forest_2", "monster_defeated"),
    ]
    for map_id in program.maps:
        local = [row for row in rows if row["observation"]["map_id"] == map_id]
        first_hit = next(
            i for i, row in enumerate(local) if "attack_hit" in row["execution"]["events"]
        )

        def player_x(row):
            return next(
                d["bbox"][0] for d in row["observation"]["detections"] if d["class_id"] == "player"
            )

        assert abs(player_x(local[first_hit]) - player_x(local[0])) > 20
    assert verify_run(root)["verified"]


def test_throwing_star_climbs_both_forest_maps_then_reacquires_and_attacks(tmp_path):
    from maple_vision.map_pack import load_map_pack

    program = load_mission_program(EXAMPLES / "missions/dead-forest-layered-hunt.json")
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    package = load_class_package(EXAMPLES / "classes/throwing_star.json")
    profile = load_input_profile(EXAMPLES / "input-bindings/throwing-star.json")
    root = tmp_path / "layered-star"
    result = run_episode(
        root,
        CapabilityRunConfig(seed=17, job="throwing_star"),
        map_pack=pack,
        runtime=RuntimeConfig(mission=program),
        class_package=package,
        input_profile=profile,
    )
    assert result["status"] == "completed" and result["independent_goal_check"]
    assert result["actions"]["climb"] == 32
    assert result["actions"]["cast"] == 4
    assert result["events"]["attack_hit"] == 8
    assert result["events"]["monster_defeated"] == 2
    assert result["input_dispatch_summary"] == {
        "scope": "offline_simulated_skill_dispatch",
        "attempts": 4,
        "accepted": 4,
        "denied": 0,
        "reasons": {"bound_skill": 4},
    }
    assert json.loads((root / "final-state.json").read_text())["player"]["ammo"] == 192
    witnesses = result["mission_evidence"]["witnesses"]
    assert [(row["map_id"], row["phase"]) for row in witnesses] == [
        ("drill_forest_1", "navigate"),
        ("drill_forest_1", "fight"),
        ("drill_forest_2", "navigate"),
        ("drill_forest_2", "fight"),
    ]
    navigation = result["runtime_evidence"]["terminal"]["state"]["navigation"]["records"]
    assert [row["status"] for row in navigation] == ["completed", "completed"]
    assert [row["accepted_steps"] for row in navigation] == [41, 83]
    assert all(row["ladder_edges"] == 1 and row["verification"]["verified"] for row in navigation)
    assert verify_run(root)["verified"]
    reinfer = verify_perception(root)
    assert reinfer["verified"] and reinfer["frames_reinferred"] == 267


def test_layered_throwing_star_rejects_wrong_skill_binding_after_climb(tmp_path):
    from maple_vision.map_pack import load_map_pack

    root = tmp_path / "wrong-binding"
    result = run_episode(
        root,
        CapabilityRunConfig(seed=17, job="throwing_star"),
        map_pack=load_map_pack(EXAMPLES / "maps/dead-forest-drill.json"),
        runtime=RuntimeConfig(
            mission=load_mission_program(EXAMPLES / "missions/dead-forest-layered-hunt.json")
        ),
        class_package=load_class_package(EXAMPLES / "classes/throwing_star.json"),
        input_profile=load_input_profile(EXAMPLES / "input-bindings/spear-fighter.json"),
    )
    assert result["status"] == "failed" and result["reason"] == "combat_progress_timeout"
    assert result["input_dispatch_summary"]["accepted"] == 0
    assert result["input_dispatch_summary"]["denied"] == 12
    assert result["events"].get("attack_hit", 0) == 0
    assert result["mission_evidence"]["witnesses"] == [
        {
            "map_id": "drill_forest_1",
            "phase": "navigate",
            "sequence": 41,
            "target_platform": "platform_1",
        }
    ]
    assert verify_perception(root)["verified"]
