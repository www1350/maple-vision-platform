"""Full task execution must preserve the configured skill-key evidence chain."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from maple_vision.class_package import load_class_package
from maple_vision.cli import main
from maple_vision.failure_bundle import export_failure_bundle, verify_failure_bundle
from maple_vision.input_bindings import (
    InputBindingProfile,
    KeyBinding,
    confirm_input_profile,
    load_input_profile,
    load_input_profile_confirmation,
    verify_input_profile_confirmation,
)
from maple_vision.lab import (
    CapabilityRunConfig,
    RunConfig,
    run_episode,
    verify_perception,
    verify_run,
)
from maple_vision.map_pack import load_map_pack
from maple_vision.mission_program import load_mission_program
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator

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


def test_user_confirmation_receipt_keeps_manual_comparison_separate_from_recognition():
    configured = profile(("J", "lucky_seven"))
    image = {
        "sha256": "a" * 64,
        "bytes": 12345,
        "media_type": "image/png",
        "width": 1536,
        "height": 864,
    }
    receipt = confirm_input_profile(configured, "browser_selected_image_digest", image)
    assert receipt["profile"] == configured.manifest()
    assert receipt["source_image"] == {
        **image,
        "digest_origin": "client_reported_browser_web_crypto",
        "image_bytes_uploaded": False,
    }
    assert receipt["confirmation"] == {"user_confirmed": True}
    assert receipt["recognition"] == {
        "method": "manual_visible_comparison",
        "automatic_skill_recognition": False,
        "candidates_generated": False,
    }
    assert len(receipt["profile_sha256"]) == len(receipt["receipt_sha256"]) == 64
    with pytest.raises(ValueError, match="exact image metadata"):
        confirm_input_profile(configured, "browser_selected_image_digest")
    with pytest.raises(ValueError, match="cannot carry"):
        confirm_input_profile(configured, "live_window_manual_comparison", image)
    with pytest.raises(ValueError, match="lowercase"):
        confirm_input_profile(
            configured,
            "browser_selected_image_digest",
            {**image, "sha256": "A" * 64},
        )


def test_confirmation_receipt_loader_is_bounded_and_rejects_tampering(tmp_path):
    configured = profile(("J", "lucky_seven"))
    receipt = confirm_input_profile(configured, "live_window_manual_comparison")
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert load_input_profile_confirmation(path) == (configured, receipt)
    receipt["receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        load_input_profile_confirmation(path)
    path.write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="byte budget"):
        load_input_profile_confirmation(path)


def test_user_reviewed_key_grid_is_bound_to_selected_image_receipt():
    configured = profile(("F", "lucky_seven"))
    image = {
        "sha256": "c" * 64,
        "bytes": 2048,
        "media_type": "image/jpeg",
        "width": 1536,
        "height": 864,
    }
    review = {
        "schema_version": 1,
        "row": "ASDFGHJKL",
        "anchors": [{"key": "A", "x": 641, "y": 430}, {"key": "L", "x": 831, "y": 430}],
        "algorithm_candidate_keys": ["A", "S", "D", "G"],
        "confirmed_content_keys": ["A", "S", "D", "F"],
        "user_reviewed": True,
    }
    receipt = confirm_input_profile(configured, "browser_selected_image_digest", image, review)
    assert receipt["schema_version"] == 2
    assert receipt["key_grid_review"]["source_image_sha256"] == image["sha256"]
    assert receipt["key_grid_review"]["tile_size"] == 21
    assert receipt["key_grid_review"]["heuristic"]["candidate_keys"] == ["A", "S", "D", "G"]
    assert receipt["key_grid_review"]["player_review"] == {
        "user_reviewed": True,
        "content_keys": ["A", "S", "D", "F"],
        "added_keys": ["F"],
        "removed_keys": ["G"],
    }
    assert receipt["recognition"]["automatic_skill_recognition"] is False
    assert receipt["recognition"]["skill_candidates_generated"] is False
    assert verify_input_profile_confirmation(receipt) == (configured, receipt)
    changed = json.loads(json.dumps(receipt))
    changed["key_grid_review"]["player_review"]["content_keys"].append("G")
    with pytest.raises(ValueError, match="modified"):
        verify_input_profile_confirmation(changed)
    with pytest.raises(ValueError, match="selected source image"):
        confirm_input_profile(configured, "live_window_manual_comparison", None, review)
    with pytest.raises(ValueError, match="row order"):
        confirm_input_profile(
            configured,
            "browser_selected_image_digest",
            image,
            {**review, "confirmed_content_keys": ["F", "A"]},
        )


def test_bound_dragon_knight_skill_completes_two_maps_and_reinfers(tmp_path):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    mission = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    configured = profile(("A", "模拟范围攻击"))
    confirmation = confirm_input_profile(configured, "live_window_manual_comparison")
    output = tmp_path / "bound"
    report = run_episode(
        output,
        RunConfig(job="dragon_knight"),
        map_pack=pack,
        runtime=RuntimeConfig(mission=mission),
        input_profile=configured,
        input_confirmation=confirmation,
    )
    assert report["status"] == "completed" and report["independent_goal_check"]
    assert report["input_profile"] == configured.manifest()
    assert report["input_confirmation"]["receipt_sha256"] == confirmation["receipt_sha256"]
    assert (
        json.loads((output / "input-binding-confirmation.json").read_text(encoding="utf-8"))
        == confirmation
    )
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


def test_bound_external_multi_hit_skill_completes_two_maps_and_reinfers(tmp_path):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    mission = load_mission_program(EXAMPLES / "missions/dead-forest-hunt.json")
    package = load_class_package(EXAMPLES / "classes/spear_fighter.json")
    configured = InputBindingProfile.from_manifest(
        json.loads((EXAMPLES / "input-bindings/spear-fighter.json").read_text())
    )
    assert configured.action_for_key("s", package.class_id, package) == (
        "triple_thrust",
        SimAction("cast", skill="triple_thrust"),
    )
    output = tmp_path / "bound-multi-hit"
    report = run_episode(
        output,
        CapabilityRunConfig(seed=17, job=package.class_id),
        map_pack=pack,
        class_package=package,
        runtime=RuntimeConfig(mission=mission),
        input_profile=configured,
    )
    assert report["status"] == "completed" and report["independent_goal_check"]
    assert report["maps_visited"] == ["drill_forest_1", "drill_forest_2"]
    assert report["events"]["cast"] == 2
    assert report["events"]["attack_hit"] == 6
    assert report["events"]["monster_defeated"] == 2
    assert report["input_dispatch_summary"] == {
        "scope": "offline_simulated_skill_dispatch",
        "attempts": 2,
        "accepted": 2,
        "denied": 0,
        "reasons": {"bound_skill": 2},
    }
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    dispatches = [
        row["execution"]["input_dispatch"]
        for row in rows
        if row["execution"]["input_dispatch"] is not None
    ]
    assert {item["key"] for item in dispatches} == {"S"}
    assert {item["action_skill"] for item in dispatches} == {"triple_thrust"}
    assert verify_run(output)["verified"]
    assert verify_perception(output)["frames_reinferred"] == report["steps"] + 1


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


def test_cli_loads_the_same_bounded_profile_and_rejects_invalid_files(tmp_path, capsys):
    configured = profile(("A", "模拟范围攻击"))
    bindings = tmp_path / "role-keys.json"
    bindings.write_text(json.dumps(configured.manifest()), encoding="utf-8")
    output = tmp_path / "cli-run"
    args = [
        "lab",
        "run",
        "--output",
        str(output),
        "--job",
        "dragon_knight",
        "--map-pack",
        str(EXAMPLES / "maps/dead-forest-drill.json"),
        "--mission-program",
        str(EXAMPLES / "missions/dead-forest-hunt.json"),
        "--runtime",
        "structured",
        "--input-bindings",
        str(bindings),
    ]
    assert main(args) == 0
    capsys.readouterr()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["input_profile"] == configured.manifest()
    assert verify_run(output)["verified"]

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    invalid_output = tmp_path / "invalid-run"
    invalid_args = [*args]
    invalid_args[invalid_args.index(str(output))] = str(invalid_output)
    invalid_args[invalid_args.index(str(bindings))] = str(invalid)
    assert main(invalid_args) == 2 and not invalid_output.exists()
    assert "invalid input binding profile fields" in capsys.readouterr().err

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (64 * 1024 + 1))
    invalid_args[invalid_args.index(str(invalid))] = str(oversized)
    assert main(invalid_args) == 2 and not invalid_output.exists()
    assert "exceeds byte budget" in capsys.readouterr().err


@pytest.mark.parametrize("map_id", ["drill_forest_1", "drill_forest_2"])
@pytest.mark.parametrize("facing", [-1, 1], ids=["left", "right"])
@pytest.mark.parametrize(
    ("forward_px", "on_platform", "hit_count"),
    [(57, False, 3), (58, False, 3), (59, False, 0), (-1, False, 0), (30, True, 0)],
    ids=["inside", "limit", "outside", "behind", "upper-platform"],
)
def test_synthetic_keyed_spear_range_on_each_map(
    map_id, facing, forward_px, on_platform, hit_count
):
    pack = load_map_pack(EXAMPLES / "maps/dead-forest-drill.json")
    package = load_class_package(EXAMPLES / "classes/spear_fighter.json")
    bindings = load_input_profile(EXAMPLES / "input-bindings/spear-fighter.json")
    sim = Simulator(
        seed=17, job=package.class_id, class_package=package, map_pack=pack, map_id=map_id
    )
    before = sim.snapshot()
    state = sim.save()
    player_x = 100
    platform = before.platforms[1] if on_platform else before.platforms[0]
    state["player"].update(x=player_x, y=platform.y - before.player.h, facing=facing)
    target = state["maps"][map_id]["monsters"][0]
    target["x"] = player_x + (before.player.w - before.monsters[0].w) / 2 + facing * forward_px
    target["hp"] = target["max_hp"] = 100
    sim = Simulator.restore(state)
    before = sim.snapshot()
    skill, action = bindings.action_for_key("S", package.class_id, package)
    assert skill == "triple_thrust" and action is not None
    cast = sim.step(action)
    assert cast.accepted and cast.events.count("cast") == 1
    assert cast.events.count("attack_hit") == hit_count
    assert before.monsters[0].hp - cast.snapshot.monsters[0].hp == 12 * hit_count
    assert before.player.mp - cast.snapshot.player.mp == 7
    retry = sim.step(action)
    assert not retry.accepted and retry.reason == "capability_cooldown"
    assert retry.snapshot.player.mp == cast.snapshot.player.mp
