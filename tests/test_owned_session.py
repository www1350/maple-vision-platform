import json
from pathlib import Path

import pytest

from maple_vision.input_bindings import InputBindingProfile, KeyBinding, confirm_input_profile
from maple_vision.owned_live_route import verify_live_route_execution
from maple_vision.owned_route_execution import OwnedRouteExecution, verify_route_execution
from maple_vision.owned_route_playback import (
    build_route_playback,
    finalize_route_playback,
    verify_route_playback,
)
from maple_vision.owned_session import OwnedSession

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    "asset",
    [
        "classes/throwing_star.json",
        "maps/dead-forest-drill.json",
        "missions/dead-forest-layered-hunt.json",
    ],
)
def test_bundled_practice_matches_authored_example(asset):
    from importlib.resources import files

    bundled = files("maple_vision").joinpath("resources", "practice", *asset.split("/"))
    assert bundled.read_bytes() == (EXAMPLES / asset).read_bytes()


def confirmed_profile(key: str = "J") -> tuple[InputBindingProfile, dict]:
    profile = InputBindingProfile("小白兔 / 无影人", (KeyBinding(key, "lucky_seven"),))
    return profile, confirm_input_profile(profile, "live_window_manual_comparison")


def test_player_confirmation_is_required_to_match_visible_key_and_enters_cast_receipt():
    profile, confirmation = confirmed_profile()
    session = OwnedSession(EXAMPLES)
    assert session.load_confirmation(confirmation) == "J"
    with pytest.raises(ValueError, match="does not match"):
        session.confirm("S")
    session.confirm("J")
    token = session.arm()
    result = session.press("J", focused=True, generation=token)
    assert result["accepted"] and result["profile_id"] == profile.profile_id
    assert result["confirmation_receipt_sha256"] == confirmation["receipt_sha256"]
    confirmed = next(row for row in session.events if row["event"] == "confirm")
    assert confirmed["confirmation_comparison_source"] == "live_window_manual_comparison"

    changed = {**confirmation, "receipt_sha256": "0" * 64}
    with pytest.raises(ValueError, match="modified"):
        OwnedSession(EXAMPLES).load_confirmation(changed)


def test_confirmed_nondefault_key_runs_one_cast_until_explicit_reconfirmation():
    session = OwnedSession(EXAMPLES)
    with pytest.raises(ValueError, match="confirm"):
        session.arm()
    session.confirm("j")
    first = session.arm()
    before = session.simulator.save()
    assert session.press("S", focused=True)["reason"] == "unbound_key"
    assert session.simulator.save() == before
    result = session.press("j", focused=True, generation=first)
    assert result["accepted"] and result["skill"] == "lucky_seven"
    assert result["hit_segments"] == 2
    assert (result["key"], result["ammo_before"], result["ammo_after"]) == ("J", 200, 198)
    assert result["confirmation_receipt_sha256"] is not None
    assert session.held_key == "J"
    after = session.simulator.save()
    assert not session.press("J", focused=True)["accepted"]
    session.release("s")
    assert session.held_key == "J"
    session.release("j")
    assert session.held_key is None and session.state == "completed"
    assert not session.press("J", focused=True)["accepted"]
    assert session.simulator.save() == after


@pytest.mark.parametrize("interruption", ["focus_lost", "manual_takeover"])
def test_interruption_releases_held_key_and_stale_callbacks_cannot_operate(interruption):
    session = OwnedSession(EXAMPLES)
    session.confirm("Q")
    old = session.arm()
    session.press("q", focused=True)
    if interruption == "focus_lost":
        assert session.press("q", focused=False)["reason"] == "focus_lost"
    else:
        session.takeover()
    assert any(
        row["event"] == "takeover" and row["released_key"] == "Q" and row["reason"] == interruption
        for row in session.events
    )
    assert session.held_key is None and session.state == "paused"
    with pytest.raises(ValueError, match="confirm"):
        session.arm()
    session.confirm("Q")
    current = session.arm()
    assert session.press("Q", focused=True, generation=old)["reason"] == "stale_dispatch"
    assert session.press("Q", focused=True, generation=current)["accepted"]
    session.release("Q", generation=old)
    assert session.held_key == "Q"
    session.release("Q", generation=current)
    assert session.held_key is None


def test_confirmed_nondefault_key_runs_the_existing_layered_route(tmp_path):
    session = OwnedSession(EXAMPLES)
    session.confirm("J")
    review = session.run_layered_route(tmp_path / "route")
    assert review["status"] == "completed"
    assert review["confirmed_key"] == "J"
    assert [row["sequence"] for row in review["navigation"]] == [41, 233]
    assert [(row["source_sequence"], row["map_id"], row["key"]) for row in review["casts"]] == [
        (71, "drill_forest_1", "J"),
        (103, "drill_forest_1", "J"),
        (250, "drill_forest_2", "J"),
        (262, "drill_forest_2", "J"),
    ]
    summary = json.loads((tmp_path / "route/summary.json").read_text())
    assert (
        summary["input_confirmation"]["receipt_sha256"]
        == review["input_confirmation_receipt_sha256"]
    )
    assert summary["input_dispatch_summary"]["accepted"] == 4
    assert (tmp_path / "route/preview.gif").is_file()
    assert set(review["source_files"]) == {"summary.json", "trace.jsonl", "preview.gif"}
    schedule = build_route_playback(tmp_path / "route", review)
    assert [item.source_sequence for item in schedule if item.labels] == [
        41,
        71,
        103,
        233,
        250,
        262,
    ]
    execution = OwnedRouteExecution(tmp_path / "route", review)
    shown_rows = []
    while execution.last_source_sequence != 72:
        item = execution.step()
        if item.source_sequence % 4 == 0 or item.labels:
            shown_rows.append((item.source_sequence, item.image))
    executed = execution.cancel("manual_takeover")
    assert (executed["executed_steps"], executed["planned_steps"]) == (73, 266)
    assert executed["last_source_sequence"] == 72
    assert verify_route_execution(tmp_path / "route", review | {"execution": executed}) == {
        "verified": True,
        "status": "taken_over",
        "executed_steps": 73,
        "planned_steps": 266,
        "last_source_sequence": 72,
        "cancelled_steps": 193,
        "journal_sha256": executed["journal_sha256"],
    }
    shown = tuple(shown_rows)
    updated = finalize_route_playback(
        tmp_path / "route",
        review | {"execution": executed},
        shown,
        status="taken_over",
        reason="manual_takeover",
    )
    assert [item["kind"] for item in updated["timeline"]] == [
        "layer_reached",
        "skill_dispatched",
        "takeover",
    ]
    assert updated["playback"]["last_source_sequence"] == 72
    assert (tmp_path / "route/route-takeover.gif").is_file()
    assert verify_route_playback(tmp_path / "route") == {
        "verified": True,
        "scope": "project_owned_synthetic_route",
        "status": "taken_over",
        "last_source_sequence": 72,
        "timeline_events": 3,
        "displayed_frames": len(shown),
        "replay_sha256": updated["playback"]["replay_sha256"],
        "execution": {
            "verified": True,
            "status": "taken_over",
            "executed_steps": 73,
            "planned_steps": 266,
            "last_source_sequence": 72,
            "cancelled_steps": 193,
            "journal_sha256": executed["journal_sha256"],
        },
    }
    journal = tmp_path / "route/route-execution.jsonl"
    journal.write_bytes(journal.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="execution summary"):
        verify_route_playback(tmp_path / "route")


def test_invalid_rebinding_and_focus_loss_cancel_an_unused_arm():
    session = OwnedSession(EXAMPLES)
    session.confirm("q")
    session.arm()
    with pytest.raises(ValueError, match="one visible letter"):
        session.confirm("shift")
    assert not session.press("q", focused=True)["accepted"]
    session.confirm("s")
    session.arm()
    assert not session.press("s", focused=False)["accepted"]
    assert not session.press("s", focused=True)["accepted"]
    assert session.simulator.snapshot().player.ammo == 200


def test_stepwise_route_can_execute_the_entire_verified_plan(tmp_path):
    session = OwnedSession(EXAMPLES)
    session.confirm("J")
    route = tmp_path / "complete-route"
    review = session.run_layered_route(route)
    execution = OwnedRouteExecution(route, review)
    shown = []
    while execution.active:
        item = execution.step()
        if item.source_sequence % 4 == 0 or item.labels or not execution.active:
            shown.append((item.source_sequence, item.image))
    assert execution.summary is not None
    assert execution.summary["status"] == "completed"
    assert execution.summary["executed_steps"] == execution.summary["planned_steps"] == 266
    updated = finalize_route_playback(
        route,
        review | {"execution": execution.summary},
        tuple(shown),
        status="completed",
        reason=None,
    )
    verified = verify_route_playback(route)
    assert verified["status"] == "completed"
    assert verified["timeline_events"] == 6
    assert verified["execution"]["cancelled_steps"] == 0
    assert (route / updated["playback"]["replay"]).name == "route-playback.gif"


def test_live_route_decides_only_the_executed_prefix_before_takeover(tmp_path):
    session = OwnedSession(EXAMPLES)
    session.confirm("J")
    route = tmp_path / "live-route"
    execution = session.start_live_layered_route(route)
    assert (route / "trace.jsonl").read_text() == ""
    assert not (route / "summary.json").exists()
    assert not (route / "player-review.json").exists()

    shown = []
    last = None
    while execution.last_source_sequence != 72:
        last = execution.step()
        assert last is not None
        if last.source_sequence % 4 == 0 or last.labels:
            shown.append((last.source_sequence, last.image))
    assert last is not None
    if shown[-1][0] != last.source_sequence:
        shown.append((last.source_sequence, last.image))
    summary = execution.cancel("manual_takeover")
    assert summary["executed_steps"] == summary["decision_cycles"] == 73
    assert summary["remaining_steps"] == "not_precomputed"
    cycles = execution.decision_cycles
    with pytest.raises(ValueError, match="not active"):
        execution.step()
    assert execution.decision_cycles == cycles
    assert [item["sequence"] for item in execution.navigation] == [41]
    assert [(item["source_sequence"], item["key"]) for item in execution.casts] == [(71, "J")]
    assert execution.review is not None
    updated = finalize_route_playback(
        route,
        execution.review,
        tuple(shown),
        status="taken_over",
        reason="manual_takeover",
    )
    assert [item["kind"] for item in updated["timeline"]] == [
        "layer_reached",
        "skill_dispatched",
        "takeover",
    ]
    assert verify_live_route_execution(route)["executed_steps"] == 73
    verified = verify_route_playback(route)
    assert verified["scope"] == "project_owned_live_synthetic_route"
    assert verified["execution"]["remaining_steps"] == "not_precomputed"
    trace = route / "trace.jsonl"
    trace.write_bytes(trace.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        verify_route_playback(route)


def test_live_route_can_be_taken_over_before_its_first_decision(tmp_path):
    session = OwnedSession(EXAMPLES)
    session.confirm("J")
    route = tmp_path / "zero-step-live-route"
    execution = session.start_live_layered_route(route)
    summary = execution.cancel("focus_lost")
    assert summary["executed_steps"] == 0
    assert summary["decision_cycles"] == 0
    assert summary["last_source_sequence"] is None
    assert summary["ammo_before"] == summary["ammo_after"] == 200
    assert verify_live_route_execution(route)["executed_steps"] == 0


def test_live_route_completes_from_current_frames_without_a_plan(tmp_path):
    session = OwnedSession(EXAMPLES)
    session.confirm("J")
    route = tmp_path / "complete-live-route"
    execution = session.start_live_layered_route(route)
    shown = []
    last = None
    while execution.active:
        item = execution.step()
        if item is not None:
            last = item
            if item.source_sequence % 4 == 0 or item.labels:
                shown.append((item.source_sequence, item.image))
    assert last is not None
    if shown[-1][0] != last.source_sequence:
        shown.append((last.source_sequence, last.image))
    assert execution.status == "completed"
    assert execution.index == 266
    assert execution.decision_cycles == 267
    assert [item["sequence"] for item in execution.navigation] == [41, 233]
    assert [item["key"] for item in execution.casts] == ["J"] * 4
    assert execution.review is not None
    finalize_route_playback(
        route,
        execution.review,
        tuple(shown),
        status="completed",
        reason=None,
    )
    verified = verify_route_playback(route)
    assert verified["timeline_events"] == 6
    assert verified["execution"]["executed_steps"] == 266
