"""Multiple visible people are normal observations, not a selected owner.

These independent boxes exercise the opt-in read-only tracker. They do not
represent annotated real-game detections or establish a controlled character.
"""

from dataclasses import replace

import pytest

from maple_vision.contracts import Detection, Status
from maple_vision.state_estimation import TemporalStateEstimator
from maple_vision.vision import VisualObservation


def _player(x, y=100, confidence=0.95):
    return Detection("player", (x, y, x + 16, y + 32), confidence)


def _observation(sequence, *, players=None, timestamp_ms=None, source="multi-player-pixels"):
    return VisualObservation(
        sequence=sequence,
        timestamp_ms=sequence * 50 if timestamp_ms is None else timestamp_ms,
        status=Status.UNKNOWN,
        confidence=0.95,
        detections=(tuple(players) if players is not None else (_player(100), _player(260)))
        + (
            Detection("rope", (106, 40, 110, 180), 0.95),
            Detection("platform", (70, 180, 190, 188), 0.95),
        ),
        hp_ratio=None,
        mp_ratio=None,
        map_id=None,
        source_id=source,
    )


def _tracker():
    return TemporalStateEstimator(
        retain_detection_hypotheses=True, retain_multiple_player_hypotheses=True
    )


def _current_players(belief):
    return tuple(
        track for track in belief.tracks if track.class_id == "player" and track.observed_this_frame
    )


@pytest.mark.parametrize("invalid", [0, 1, None, "yes"])
def test_multiple_player_flag_requires_an_actual_boolean(invalid):
    with pytest.raises(ValueError, match="retain_multiple_player_hypotheses"):
        TemporalStateEstimator(
            retain_detection_hypotheses=True, retain_multiple_player_hypotheses=invalid
        )


def test_multiple_player_flag_requires_read_only_detection_hypotheses():
    with pytest.raises(ValueError):
        TemporalStateEstimator(retain_multiple_player_hypotheses=True)


def test_all_current_players_and_terrain_are_tracked_without_a_singular_player_claim():
    tracker = _tracker()
    observation = _observation(0)
    result = tracker.update(observation, 0)

    assert result.reason == "fresh_detection_hypotheses"
    assert result.observation.detections == observation.detections
    assert len(_current_players(result)) == 2
    assert {track.class_id for track in result.tracks} == {"player", "rope", "platform"}
    assert all(track.observed_this_frame for track in result.tracks)
    assert not result.actionable
    assert result.observation.status is Status.UNKNOWN
    assert result.last_accepted_sequence is result.last_accepted_timestamp_ms is None
    for field_name in ("player_bbox", "map_id", "hp_ratio", "mp_ratio"):
        field = result.field(field_name)
        assert field.status is Status.UNKNOWN
        assert field.value is None
        assert not field.observed_this_frame


def test_other_people_entering_and_leaving_do_not_reset_existing_distinct_tracks():
    tracker = _tracker()
    initial = tracker.update(_observation(0), 0)
    ids = {track.bbox[0]: track.track_id for track in _current_players(initial)}
    entered = tracker.update(
        _observation(1, players=(_player(103), _player(263), _player(440))), 50
    )
    current = {track.bbox[0]: track for track in _current_players(entered)}
    assert current[103].track_id == ids[100]
    assert current[263].track_id == ids[260]
    assert current[103].association == current[263].association == "matched"
    assert current[440].association == "new"
    assert entered.epoch == initial.epoch

    left = tracker.update(_observation(2, players=(_player(106), _player(266))), 100)
    assert {track.track_id for track in _current_players(left)} == set(ids.values())
    retained_other = next(track for track in left.tracks if track.track_id == current[440].track_id)
    assert not retained_other.observed_this_frame
    assert retained_other.association == "unobserved"
    assert not left.actionable


@pytest.mark.parametrize("competing_confidence", [0.95, 0.05])
def test_any_second_raw_person_clears_previous_singular_player_field(competing_confidence):
    tracker = _tracker()
    first = tracker.update(_observation(0, players=(_player(100),)), 0)
    assert first.field("player_bbox").value == (100, 100, 116, 132)
    second = tracker.update(
        _observation(1, players=(_player(102), _player(300, confidence=competing_confidence))),
        50,
    )
    field = second.field("player_bbox")
    assert field.status is Status.UNKNOWN
    assert field.value is field.source_sequence is field.observed_at_ms is None
    assert not field.observed_this_frame
    assert len(_current_players(second)) == (2 if competing_confidence >= 0.5 else 1)
    assert not second.actionable


def test_no_current_detections_clear_singular_field_and_leave_only_unobserved_history():
    tracker = _tracker()
    first = tracker.update(_observation(0, players=(_player(100),)), 0)
    empty = replace(_observation(1, players=()), detections=())
    second = tracker.update(empty, 50)
    assert second.field("player_bbox").value is None
    assert second.field("player_bbox").status is Status.UNKNOWN
    assert not second.field("player_bbox").observed_this_frame
    assert {track.track_id for track in second.tracks} == {track.track_id for track in first.tracks}
    assert not any(track.observed_this_frame for track in second.tracks)
    assert not second.actionable


def test_crossing_ambiguity_is_local_to_the_competing_people():
    tracker = _tracker()
    first = tracker.update(_observation(0, players=(_player(100), _player(140), _player(350))), 0)
    distant_id = next(track.track_id for track in _current_players(first) if track.bbox[0] == 350)
    crossing = tracker.update(
        _observation(1, players=(_player(116), _player(120), _player(352))), 50
    )
    overlapping = [track for track in _current_players(crossing) if track.bbox[0] < 200]
    assert len(overlapping) == 2
    assert all(
        track.identity_uncertain and track.association == "ambiguous" for track in overlapping
    )
    distant = next(track for track in _current_players(crossing) if track.bbox[0] == 352)
    assert distant.track_id == distant_id
    assert distant.association == "matched"
    assert not distant.identity_uncertain
    assert crossing.field("player_bbox").value is None


@pytest.mark.parametrize(
    "sequence,timestamp,now,reason",
    [
        (0, 150, 150, "duplicate_or_reordered_observation"),
        (1, 151, 150, "future_observation"),
        (1, 101, 400, "stale_observation"),
        (1, 99, 150, "observation_timestamp_rollback"),
    ],
)
def test_multiple_player_mode_keeps_existing_frame_rejection_guards(
    sequence, timestamp, now, reason
):
    tracker = _tracker()
    first = tracker.update(_observation(0, timestamp_ms=100), 100)
    rejected = tracker.update(_observation(sequence, timestamp_ms=timestamp), now)
    assert rejected.reason == reason
    assert not rejected.actionable
    assert not any(track.observed_this_frame for track in rejected.tracks)
    assert {track.track_id for track in rejected.tracks} <= {
        track.track_id for track in first.tracks
    }
    assert rejected.field("player_bbox").value is None


def test_multiple_player_mode_preserves_terminal_clock_rollback():
    tracker = _tracker()
    tracker.update(_observation(0, timestamp_ms=100), 100)
    failed = tracker.update(_observation(1, timestamp_ms=90), 90)
    assert failed.reason == "clock_rollback"
    assert not failed.tracks
    later = tracker.update(_observation(2, timestamp_ms=200), 200)
    assert later.reason == "clock_rollback"
    assert not later.tracks
    assert not later.actionable


def test_source_change_rebuilds_the_multi_player_context():
    tracker = _tracker()
    first = tracker.update(_observation(0), 0)
    second = tracker.update(_observation(1, source="another-window"), 50)
    assert second.epoch == first.epoch + 1
    assert not (
        {track.track_id for track in first.tracks} & {track.track_id for track in second.tracks}
    )
    assert all(track.source_id == "another-window" for track in second.tracks)
    assert all(track.association == "new" for track in second.tracks)
    assert second.field("player_bbox").value is None


def test_default_mode_remains_byte_compatible_and_does_not_enable_multi_player_tracking():
    implicit = TemporalStateEstimator(retain_detection_hypotheses=True)
    explicit = TemporalStateEstimator(
        retain_detection_hypotheses=True, retain_multiple_player_hypotheses=False
    )
    for frame in (_observation(0), _observation(1, players=(_player(100),)), _observation(2)):
        first = implicit.update(frame, frame.timestamp_ms)
        second = explicit.update(frame, frame.timestamp_ms)
        assert first.to_dict() == second.to_dict()
        if frame.sequence == 0:
            assert first.reason == "unknown_observation"
            assert not first.tracks
            assert first.field("player_bbox").value is None


def test_known_but_ambiguous_game_state_is_not_upgraded_by_read_only_multi_player_option():
    tracker = _tracker()
    ambiguous = replace(
        _observation(0), status=Status.KNOWN, map_id="map", hp_ratio=0.8, mp_ratio=0.7
    )
    result = tracker.update(ambiguous, 0)
    assert result.reason == "ambiguous_player"
    assert not result.actionable
    assert not result.tracks
    assert result.field("player_bbox").value is None
