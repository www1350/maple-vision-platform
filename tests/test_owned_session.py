import json
from pathlib import Path

import pytest

from maple_vision.input_bindings import InputBindingProfile, KeyBinding, confirm_input_profile
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
