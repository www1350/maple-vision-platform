import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from maple_vision.cli import main
from maple_vision.lab import RunConfig, run_episode, verify_perception, verify_run
from maple_vision.mission_program import MissionProgram, MissionStage, load_mission_program
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
