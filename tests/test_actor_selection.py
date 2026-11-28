"""A reviewed anchor selects one geometric actor among normal visible players.

Independent trajectories deliberately include other people, camera translation,
and identity conflicts. Passing these engineering cases is not real-game identity
accuracy or proof that the chosen actor belongs to a particular user.
"""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from maple_vision.actor_selection import ActorAnchor, ActorSelectionConfig, ActorSelector
from maple_vision.contracts import Detection, Status
from maple_vision.movement_hypotheses import MovementHypothesisEstimator
from maple_vision.state_estimation import StateConfig, TemporalStateEstimator
from maple_vision.vision import VisualObservation


def _rgb(sequence):
    return f"{sequence + 1:064x}"


def _box(x=150, feet=244):
    return x - 8, feet - 32, x + 8, feet


def _frame(
    sequence,
    *,
    x=150,
    feet=244,
    selected_present=True,
    confidence=0.95,
    other=(330, 244),
    extras=(),
    camera_y=0,
    timestamp_ms=None,
    source="independent-selected-actor",
):
    detections = [
        Detection("rope", (146, 170, 154, 360), 0.95),
        Detection("platform", (80, 180, 240, 192), 0.95),
        Detection("platform", (400, 100, 580, 112), 0.95),
    ]
    if selected_present:
        detections.append(Detection("player", _box(x, feet), confidence))
    if other is not None:
        detections.append(Detection("player", _box(*other), 0.95))
    detections.extend(extras)
    detections = [
        replace(
            item,
            bbox=(item.bbox[0], item.bbox[1] + camera_y, item.bbox[2], item.bbox[3] + camera_y),
        )
        for item in detections
    ]
    return VisualObservation(
        sequence=sequence,
        timestamp_ms=sequence * 50 if timestamp_ms is None else timestamp_ms,
        status=Status.UNKNOWN,
        confidence=0.95,
        detections=tuple(detections),
        hp_ratio=None,
        mp_ratio=None,
        map_id=None,
        source_id=source,
    )


def _anchor(sequence=0, *, bbox=None, timestamp_ms=None, **changes):
    value = {
        "source_sequence": sequence,
        "timestamp_ms": sequence * 50 if timestamp_ms is None else timestamp_ms,
        "rgb_sha256": _rgb(sequence),
        "bbox": _box() if bbox is None else bbox,
        "reviewer": "independent geometric review",
        "subject": "reviewed actor A; ownership unverified",
    }
    return ActorAnchor(**(value | changes))


def _tracker(config=None):
    return TemporalStateEstimator(
        config,
        retain_detection_hypotheses=True,
        retain_multiple_player_hypotheses=True,
    )


def _step(tracker, selector, frame, *, now_ms=None, rgb=None):
    belief = tracker.update(frame, frame.timestamp_ms if now_ms is None else now_ms)
    selected = selector.update(belief, rgb_sha256=_rgb(frame.sequence) if rgb is None else rgb)
    return belief, selected


def test_anchor_and_configuration_round_trip_with_explicit_schema_and_immutable_values():
    anchor = _anchor()
    document = json.loads(json.dumps(anchor.to_dict()))
    assert document == {
        "schema_version": 1,
        "source_sequence": 0,
        "timestamp_ms": 0,
        "rgb_sha256": _rgb(0),
        "bbox": list(_box()),
        "reviewer": "independent geometric review",
        "subject": "reviewed actor A; ownership unverified",
    }
    assert ActorAnchor.from_dict(document) == anchor
    config = ActorSelectionConfig()
    assert ActorSelectionConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.anchor_iou_threshold == 0.5
    assert config.min_confidence == 0.7
    assert config.max_age_ms == config.max_gap_ms == 250
    with pytest.raises(FrozenInstanceError):
        anchor.source_sequence = 5
    with pytest.raises(FrozenInstanceError):
        config.max_gap_ms = 999


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"source_sequence": True},
        {"source_sequence": -1},
        {"timestamp_ms": -1},
        {"rgb_sha256": "not-a-pixel-digest"},
        {"bbox": [142, 212, 142, 244]},
        {"bbox": [-1, 212, 158, 244]},
        {"bbox": [142, 212, float("nan"), 244]},
        {"reviewer": ""},
        {"subject": "\nunclear"},
        {"owner_verified": True},
    ],
)
def test_anchor_rejects_invalid_or_unbound_provenance(change):
    with pytest.raises(ValueError):
        ActorAnchor.from_dict(_anchor().to_dict() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"anchor_iou_threshold": 0},
        {"anchor_iou_threshold": True},
        {"anchor_iou_threshold": 1.1},
        {"min_confidence": float("nan")},
        {"min_confidence": -0.1},
        {"max_gap_ms": 0},
        {"max_gap_ms": True},
        {"max_age_ms": -1},
        {"schema_version": 2},
        {"arbitrary": 1},
    ],
)
def test_selection_configuration_rejects_ambiguous_thresholds(change):
    with pytest.raises(ValueError):
        ActorSelectionConfig.from_dict(ActorSelectionConfig().to_dict() | change)


def test_anchor_waits_for_its_exact_source_frame_without_claiming_a_person_early():
    tracker = _tracker()
    selector = ActorSelector(_anchor(2))
    for sequence in (0, 1):
        _, selection = _step(tracker, selector, _frame(sequence))
        assert selection.status == "waiting"
        assert selection.actor_track_id is None
    belief, selection = _step(tracker, selector, _frame(2))
    assert selection.status == "selected"
    selected = next(track for track in belief.tracks if track.track_id == selection.actor_track_id)
    assert selected.bbox == _box()
    assert len([track for track in belief.tracks if track.class_id == "player"]) == 2
    assert belief.field("player_bbox").value is None
    assert not belief.actionable
    payload = selection.to_dict()
    assert payload["read_only"] is True
    assert payload["actionable"] is False
    assert payload["owner_identity_verified"] is False
    assert payload["action_execution"] == "none"
    with pytest.raises(FrozenInstanceError):
        selection.actor_track_id = "other"


@pytest.mark.parametrize("camera_follows", [False, True])
def test_selected_actor_climbs_and_arrives_with_another_player_still_visible(camera_follows):
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    movement = MovementHypothesisEstimator()
    positions = (244, 236, 228, 220, 212, 204, 196, 188, 180, 180, 180, 180)
    results = []
    selected_ids = set()
    for sequence, feet in enumerate(positions):
        frame = _frame(sequence, feet=feet, camera_y=244 - feet if camera_follows else 0)
        belief, selection = _step(tracker, selector, frame)
        before = belief.to_dict()
        result = movement.update(belief, actor_selection=selection)
        assert belief.to_dict() == before
        assert belief.observation.detections == frame.detections
        assert (
            len(
                [
                    track
                    for track in belief.tracks
                    if track.class_id == "player" and track.observed_this_frame
                ]
            )
            == 2
        )
        assert selection.status == "selected"
        selected_ids.add(selection.actor_track_id)
        assert not result.to_dict()["actionable"]
        assert not belief.actionable
        results.append(result)
    assert len(selected_ids) == 1
    assert any(result.state == "climbing_candidate" for result in results)
    assert results[-1].state == "platform_arrival_candidate"
    assert all(
        result.to_dict()["actor_track_id"] in selected_ids
        for result in results
        if result.to_dict()["actor_track_id"] is not None
    )


def test_other_actor_climbing_the_same_rope_does_not_move_the_selected_stationary_actor():
    tracker = _tracker()
    selector = ActorSelector(_anchor(bbox=_box(feet=180)))
    movement = MovementHypothesisEstimator()
    for sequence in range(12):
        # Both people are on the rope, separated vertically. Only the lower,
        # unselected person climbs; the selected upper figure stays planted.
        belief, selection = _step(
            tracker, selector, _frame(sequence, feet=180, other=(150, 340 - 4 * sequence))
        )
        assert selection.status == "selected"
        result = movement.update(belief, actor_selection=selection)
        assert result.state not in {"climbing_candidate", "platform_arrival_candidate"}


def test_stationary_selected_world_position_does_not_climb_when_camera_and_other_players_move():
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    movement = MovementHypothesisEstimator()
    for sequence in range(10):
        belief, selection = _step(
            tracker,
            selector,
            _frame(sequence, camera_y=-4 * sequence, other=(330 + sequence, 244)),
        )
        assert selection.status == "selected"
        result = movement.update(belief, actor_selection=selection)
        assert result.state not in {"climbing_candidate", "platform_arrival_candidate"}


@pytest.mark.parametrize("at_anchor", [False, True])
def test_distant_low_confidence_raw_person_does_not_invalidate_selected_actor(at_anchor):
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    low = Detection("player", _box(x=650), 0.05)
    _, first = _step(tracker, selector, _frame(0, extras=(low,) if at_anchor else ()))
    assert first.status == "selected"
    _, later = _step(tracker, selector, _frame(1, feet=240, extras=(low,)))
    assert later.status == "selected"
    assert later.actor_track_id == first.actor_track_id


@pytest.mark.parametrize("at_anchor", [False, True])
def test_overlapping_low_confidence_raw_person_invalidates_even_without_a_track(at_anchor):
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    duplicate = Detection("player", _box(), 0.05)
    if not at_anchor:
        _, first = _step(tracker, selector, _frame(0))
        assert first.status == "selected"
    sequence = 0 if at_anchor else 1
    belief, result = _step(tracker, selector, _frame(sequence, extras=(duplicate,)))
    assert (
        len(
            [
                track
                for track in belief.tracks
                if track.class_id == "player" and track.observed_this_frame
            ]
        )
        == 2
    )
    assert result.status == "invalidated"
    assert result.actor_track_id is None
    _, later = _step(tracker, selector, _frame(sequence + 1))
    assert later.status == "invalidated"
    assert later.actor_track_id is None


def test_low_confidence_candidate_near_previous_position_prevents_a_silent_identity_jump():
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    _, first = _step(tracker, selector, _frame(0))
    low_at_previous = Detection("player", _box(), 0.05)
    belief, result = _step(tracker, selector, _frame(1, x=190, extras=(low_at_previous,)))
    # The current selected-size box is disjoint from the competing raw box.
    # The old geometric matcher alone would associate the 40 px jump.
    current = next(track for track in belief.tracks if track.track_id == first.actor_track_id)
    assert current.association == "matched"
    assert current.bbox[0] >= low_at_previous.bbox[2]
    assert result.status == "invalidated"
    assert result.actor_track_id is None


def test_anchor_matching_two_distinct_people_never_picks_the_first_candidate():
    tracker = _tracker()
    selector = ActorSelector(
        _anchor(bbox=(140, 210, 180, 244)), ActorSelectionConfig(anchor_iou_threshold=0.25)
    )
    belief, result = _step(tracker, selector, _frame(0, other=(170, 244)))
    assert len([track for track in belief.tracks if track.class_id == "player"]) == 2
    assert result.status == "invalidated"
    assert result.actor_track_id is None


@pytest.mark.parametrize("failure", ["rgb", "timestamp", "past_anchor", "missing_person"])
def test_failed_anchor_is_terminal_instead_of_selecting_a_later_lookalike(failure):
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    if failure == "rgb":
        _, failed = _step(tracker, selector, _frame(0), rgb="f" * 64)
    elif failure == "timestamp":
        _, failed = _step(tracker, selector, _frame(0, timestamp_ms=1))
    elif failure == "past_anchor":
        _, failed = _step(tracker, selector, _frame(1))
    else:
        _, failed = _step(tracker, selector, _frame(0, selected_present=False))
    assert failed.status == "invalidated"
    assert failed.actor_track_id is None
    _, later = _step(tracker, selector, _frame(2))
    assert later.status == "invalidated"
    assert later.actor_track_id is None


@pytest.mark.parametrize(
    "failure",
    ["missing", "overlap", "source", "sequence_gap", "time_gap", "time_stall", "confidence"],
)
def test_selected_identity_failure_latches_and_never_reacquires_or_switches(failure):
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    _, initial = _step(tracker, selector, _frame(0))
    assert initial.status == "selected"
    frame = _frame(1)
    if failure == "missing":
        frame = _frame(1, selected_present=False)
    elif failure == "overlap":
        frame = _frame(1, other=(152, 244))
    elif failure == "source":
        frame = _frame(1, source="another-recording")
    elif failure == "sequence_gap":
        frame = _frame(2)
    elif failure == "time_gap":
        frame = _frame(1, timestamp_ms=251)
    elif failure == "time_stall":
        frame = _frame(1, timestamp_ms=0)
    elif failure == "confidence":
        frame = _frame(1, confidence=0.6)
    _, failed = _step(tracker, selector, frame)
    assert failed.status == "invalidated"
    assert failed.actor_track_id is None
    # Original actor and all geometry return, but loss of continuity requires
    # an explicit new selector, never opportunistic continuation or fallback.
    _, returned = _step(tracker, selector, _frame(3, timestamp_ms=350))
    assert returned.status == "invalidated"
    assert returned.actor_track_id is None


def test_source_switch_before_anchor_invalidates_the_pending_selection():
    tracker = _tracker()
    selector = ActorSelector(_anchor(2))
    _, waiting = _step(tracker, selector, _frame(0))
    assert waiting.status == "waiting"
    _, failed = _step(tracker, selector, _frame(1, source="different-window"))
    assert failed.status == "invalidated"
    _, returned = _step(tracker, selector, _frame(2))
    assert returned.status == "invalidated"


def test_epoch_change_invalidates_selection_even_with_same_source_and_track_object():
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    _step(tracker, selector, _frame(0))
    frame = _frame(1)
    belief = tracker.update(frame, frame.timestamp_ms)
    shifted_epoch = replace(belief, epoch=belief.epoch + 1)
    failed = selector.update(shifted_epoch, rgb_sha256=_rgb(1))
    assert failed.status == "invalidated"
    _, returned = _step(tracker, selector, _frame(2))
    assert returned.status == "invalidated"


def test_selector_enforces_its_own_freshness_limit_even_when_tracker_accepts_older_pixels():
    state_config = StateConfig(max_age_ms=500, track_ttl_ms=1000)
    tracker = _tracker(state_config)
    selector = ActorSelector(_anchor(), state_config=state_config)
    _step(tracker, selector, _frame(0))
    belief, failed = _step(tracker, selector, _frame(1), now_ms=350)
    assert belief.reason == "fresh_detection_hypotheses"
    assert any(track.observed_this_frame for track in belief.tracks)
    assert failed.status == "invalidated"


def test_movement_rejects_a_selection_from_another_frame_or_modified_belief():
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    first, first_selection = _step(tracker, selector, _frame(0))
    second, _ = _step(tracker, selector, _frame(1, feet=236))
    for belief in (second, replace(first, reason="different-belief-evidence")):
        with pytest.raises(ValueError):
            MovementHypothesisEstimator().update(belief, actor_selection=first_selection)


def test_selected_movement_cannot_fall_back_when_selection_argument_is_omitted():
    tracker = _tracker()
    selector = ActorSelector(_anchor())
    movement = MovementHypothesisEstimator()
    first, selection = _step(tracker, selector, _frame(0))
    movement.update(first, actor_selection=selection)
    # Only a different person remains; making it the singular raw player must
    # not silently switch an estimator that was configured for the anchor.
    frame = _frame(1, selected_present=False, other=(150, 236))
    later = tracker.update(frame, 50)
    result = movement.update(later)
    assert result.state == "unknown"
    assert "actor_selection_required" in result.to_dict()["reasons"]


def test_new_explicit_selection_requires_a_new_movement_estimator():
    tracker = _tracker()
    frame = _frame(0)
    belief = tracker.update(frame, 0)
    first = ActorSelector(_anchor()).update(belief, rgb_sha256=_rgb(0))
    other = ActorSelector(_anchor(subject="new explicit reviewed actor selection")).update(
        belief, rgb_sha256=_rgb(0)
    )
    assert first.selection_id != other.selection_id
    movement = MovementHypothesisEstimator()
    movement.update(belief, actor_selection=first)
    with pytest.raises(ValueError):
        movement.update(belief, actor_selection=other)


def test_waiting_and_invalidated_selection_results_keep_movement_unknown():
    tracker = _tracker()
    selector = ActorSelector(_anchor(2))
    movement = MovementHypothesisEstimator()
    first, waiting = _step(tracker, selector, _frame(0, other=None))
    result = movement.update(first, actor_selection=waiting)
    assert result.state == "unknown"
    assert any(reason.startswith("actor_selection_") for reason in result.to_dict()["reasons"])
    later, invalidated = _step(tracker, selector, _frame(3, other=None))
    assert invalidated.status == "invalidated"
    result = movement.update(later, actor_selection=invalidated)
    assert result.state == "unknown"
    assert result.to_dict()["actor_track_id"] is None


def test_default_movement_still_requires_single_actor_and_has_no_selection_payload():
    tracker = _tracker()
    movement = MovementHypothesisEstimator()
    result = movement.update(tracker.update(_frame(0), 0))
    assert result.state == "unknown"
    assert "conflicting_players" in result.to_dict()["reasons"]
    assert "actor_selection" not in result.to_dict()
