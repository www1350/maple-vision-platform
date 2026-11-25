"""Full task execution must preserve the configured skill-key evidence chain."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from maple_vision.failure_bundle import export_failure_bundle, verify_failure_bundle
from maple_vision.input_bindings import InputBindingProfile, KeyBinding
from maple_vision.lab import RunConfig, run_episode, verify_perception, verify_run
from maple_vision.map_pack import load_map_pack
from maple_vision.mission_program import load_mission_program
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import SimAction

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def profile(*bindings: tuple[str, str]) -> InputBindingProfile:
    return InputBindingProfile("小白兔 / 大美人乖乖", tuple(KeyBinding(*item) for item in bindings))


def test_profile_normalizes_keys_and_resolves_only_supported_simulated_skills():
    configured = profile(("Control", "普通攻击"), ("Ａ", "模拟范围攻击"))
    skill, action = configured.action_for_key("ctrl", "dragon_knight", None)
    assert skill == "普通攻击" and action == SimAction("attack")
    skill, action = configured.action_for_key("a", "dragon_knight", None)
    assert skill == "模拟范围攻击" and action == SimAction("attack", skill="area")
    assert configured.action_for_key("s", "dragon_knight", None) == (None, None)
    with pytest.raises(ValueError, match="duplicate normalized keys"):
        profile(("Control", "普通攻击"), ("ctrl", "模拟范围攻击"))
    with pytest.raises(ValueError, match="duplicate normalized skills"):
        profile(("A", "普通攻击"), ("S", " 普通攻击 "))


def test_bound_dragon_knight_skill_completes_two_maps_and_reinfers(tmp_path):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    mission = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    configured = profile(("A", "模拟范围攻击"))
    output = tmp_path / "bound"
    report = run_episode(
        output,
        RunConfig(job="dragon_knight"),
        map_pack=pack,
        runtime=RuntimeConfig(mission=mission),
        input_profile=configured,
    )
    assert report["status"] == "completed" and report["independent_goal_check"]
    assert report["input_profile"] == configured.manifest()
    assert report["input_dispatch_summary"] == {
        "scope": "offline_simulated_skill_dispatch",
        "attempts": 4,
        "accepted": 4,
        "denied": 0,
        "reasons": {"bound_skill": 4},
    }
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    dispatches = [
        row["execution"]["input_dispatch"]
        for row in rows
        if row["execution"]["input_dispatch"] is not None
    ]
    assert len(dispatches) == 4
    assert {item["key"] for item in dispatches} == {"A"}
    assert {item["action_skill"] for item in dispatches} == {"模拟范围攻击"}
    assert all(item["accepted"] and item["reason"] == "bound_skill" for item in dispatches)
    assert verify_run(output)["verified"]
    assert verify_perception(output)["frames_reinferred"] == report["steps"] + 1
    bundle = tmp_path / "bundle"
    export_failure_bundle(output, bundle)
    assert verify_failure_bundle(bundle)["action_replay"]["verified"]


def test_unbound_skill_is_denied_as_idle_and_cannot_complete_mission(tmp_path):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    mission = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    output = tmp_path / "unbound"
    report = run_episode(
        output,
        RunConfig(job="dragon_knight", max_steps=120),
        map_pack=pack,
        runtime=RuntimeConfig(mission=mission),
        input_profile=profile(("A", "普通攻击")),
    )
    assert report["status"] == "failed" and not report["independent_goal_check"]
    assert report["input_dispatch_summary"]["denied"] > 0
    assert report["input_dispatch_summary"]["accepted"] == 0
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    denied = [
        row
        for row in rows
        if row["execution"]["input_dispatch"] is not None
        and not row["execution"]["input_dispatch"]["accepted"]
    ]
    assert denied
    assert {row["execution"]["input_dispatch"]["reason"] for row in denied} == {"unbound_skill"}
    assert all(
        row["execution"]["applied_action"]
        == {"kind": "idle", "direction": 0, "skill": "basic", "target": ""}
        for row in denied
    )
    assert verify_run(output)["verified"]
    assert verify_perception(output)["recorded_status"] == "failed"


def test_replay_recomputes_dispatch_instead_of_trusting_rehashed_trace(tmp_path):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    mission = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    original = tmp_path / "original"
    run_episode(
        original,
        RunConfig(job="dragon_knight"),
        map_pack=pack,
        runtime=RuntimeConfig(mission=mission),
        input_profile=profile(("A", "模拟范围攻击")),
    )
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    lines = (changed / "trace.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    first = next(row for row in rows if row["execution"]["input_dispatch"] is not None)
    first["execution"]["input_dispatch"]["key"] = "S"
    (changed / "trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    manifest = json.loads((changed / "manifest.json").read_text())
    manifest["files"]["trace.jsonl"] = hashlib.sha256(
        (changed / "trace.jsonl").read_bytes()
    ).hexdigest()
    (changed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="input dispatch disagrees"):
        verify_run(changed)
