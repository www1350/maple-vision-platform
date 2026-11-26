from pathlib import Path

import pytest

from maple_vision.owned_session import OwnedSession

EXAMPLES = Path(__file__).parents[1] / "examples"


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
