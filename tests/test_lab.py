import json

import pytest

from maple_vision.lab import RunConfig, chat_draft, run_episode, run_soak, verify_run


@pytest.mark.parametrize("job", ["assassin", "dragon_knight", "cleric"])
def test_visual_mission_completes_two_maps_and_independent_goals(job):
    result = run_episode(config=RunConfig(seed=7, job=job))
    assert result["status"] == "completed", result
    assert result["independent_goal_check"] is True
    assert result["events"]["monster_defeated"] == 6
    assert result["events"]["loot"] == 6
    assert result["events"]["dialogue"] == 2
    assert result["maps_visited"] == ["cavern", "meadow"]
    if job == "assassin":
        assert result["events"]["jump"] >= 2
    if job == "dragon_knight":
        assert result["events"]["potion_mp"] >= 1


@pytest.mark.parametrize("fault", ["brief_dropout", "duplicate_frames"])
def test_temporary_bad_frames_pause_then_recover(fault):
    result = run_episode(config=RunConfig(fault=fault))
    assert result["status"] == "completed"
    assert result["paused_steps"] == 6


def test_persistent_bad_frames_stop_instead_of_claiming_success(tmp_path):
    bundle = tmp_path / "failure"
    result = run_episode(bundle, RunConfig(fault="persistent_dropout"))
    assert result["status"] == "stopped"
    assert result["steps"] == 31
    assert result["independent_goal_check"] is False
    replay = verify_run(bundle)
    assert replay["verified"] and replay["steps_replayed"] == 31
    assert replay["perception_reproduced"] is False


def test_lost_inputs_trigger_bounded_combat_progress_failure():
    result = run_episode(config=RunConfig(fault="freeze_input"))
    assert result["status"] == "failed"
    assert result["reason"] == "combat_progress_timeout"
    assert result["steps"] < 100
    assert not result["independent_goal_check"]


def test_completed_bundle_replays_and_never_overwrites(tmp_path):
    output = tmp_path / "run"
    result = run_episode(output)
    assert result["status"] == "completed"
    replay = verify_run(output)
    assert replay["verified"] and replay["independent_goal_check"]
    assert replay["runtime_identity_matches"]
    with pytest.raises(FileExistsError):
        run_episode(output)


def test_chat_only_builds_an_explicit_human_review_draft():
    draft = chat_draft("在吗？", "稍等，我正在看消息。")
    assert draft["sent"] is False
    assert draft["requires_human_confirmation"] is True
    with pytest.raises(ValueError):
        chat_draft("", "reply")


def test_soak_counts_wall_time_not_accelerated_simulation(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("maple_vision.lab.time.monotonic", lambda: clock[0])

    def episode(*, config):
        clock[0] += 0.4
        return {"status": "completed", "simulated_seconds": 86400, "job": config.job}

    monkeypatch.setattr("maple_vision.lab.run_episode", episode)
    result = run_soak(tmp_path / "soak", seconds=1)
    assert result["episodes"] == 3
    assert result["actual_wall_seconds"] == pytest.approx(1.2)
    assert result["simulated_seconds"] == 259200
    assert result["24h_wall_soak"] == "not_completed"
    assert json.loads((tmp_path / "soak" / "soak.json").read_text())["passed"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"max_steps": 0},
        {"record_stride": 0},
        {"fault": "injected"},
        {"job": "unknown"},
    ],
)
def test_run_config_rejects_invalid_boundaries(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)
