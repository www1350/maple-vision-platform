"""Temporal evidence is distinct from current measurements and verified identities."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from maple_vision.contracts import Detection, Status
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.state_estimation import (
    BeliefState,
    FieldBelief,
    StateConfig,
    TemporalStateEstimator,
)
from maple_vision.vision import PixelPerception, VisualObservation


def observation(sequence=0, timestamp_ms=None, *, monsters=(80,), **changes):
    result = VisualObservation(
        sequence,
        sequence * 50 if timestamp_ms is None else timestamp_ms,
        Status.KNOWN,
        0.95,
        (Detection("player", (20, 146, 34, 168), 0.95),)
        + tuple(Detection("monster", (x, 152, x + 10, 168), 0.9) for x in monsters),
        0.7,
        0.8,
        "meadow",
    )
    return replace(result, **changes)


def monsters(belief, *, observed_only=False):
    return tuple(
        t
        for t in belief.tracks
        if t.class_id == "monster" and (not observed_only or t.observed_this_frame)
    )


def test_default_config_round_trip_has_explicit_complete_versioned_schema():
    config = StateConfig()
    payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
    assert payload == {
        "schema_version": 1,
        "max_age_ms": 250,
        "field_ttl_ms": 1000,
        "track_ttl_ms": 500,
        "max_tracks": 128,
        "max_detections": 128,
        "min_confidence": 0.5,
        "max_match_distance": 48.0,
        "ambiguity_margin": 0.15,
    }
    assert StateConfig.from_dict(payload) == config
    with pytest.raises(FrozenInstanceError):
        config.max_tracks = 10


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"max_age_ms": True},
        {"max_age_ms": -1},
        {"field_ttl_ms": 250},
        {"track_ttl_ms": 250},
        {"field_ttl_ms": 120001},
        {"max_tracks": 513},
        {"max_tracks": 1},
        {"max_detections": 0},
        {"max_detections": 257},
        {"max_match_distance": 0},
        {"max_match_distance": float("inf")},
        {"min_confidence": float("nan")},
        {"min_confidence": True},
        {"ambiguity_margin": -0.1},
        {"unexpected": 1},
    ],
    ids=[
        "boolean-schema",
        "float-schema",
        "unknown-schema",
        "boolean-age",
        "negative-age",
        "field-ttl-not-longer-than-age",
        "track-ttl-not-longer-than-age",
        "unbounded-ttl",
        "too-many-tracks",
        "fewer-tracks-than-detections",
        "empty-detection-budget",
        "too-many-detections",
        "zero-match-distance",
        "infinite-distance",
        "nan-confidence",
        "boolean-confidence",
        "negative-ambiguity",
        "extra-key",
    ],
)
def test_configuration_rejects_ambiguous_or_unbounded_values(change):
    payload = StateConfig().to_dict() | change
    with pytest.raises(ValueError):
        StateConfig.from_dict(payload)


def test_configuration_rejects_missing_keys_and_nonobjects():
    payload = StateConfig().to_dict()
    payload.pop("field_ttl_ms")
    for invalid in (payload, [], None, "config"):
        with pytest.raises(ValueError):
            StateConfig.from_dict(invalid)
    with pytest.raises(ValueError):
        TemporalStateEstimator(config={})


def test_actual_pixel_observation_is_preserved_without_reordering_or_synthetic_fields():
    frame = SceneRenderer().render(Simulator().snapshot())
    original = PixelPerception().observe(frame)
    result = TemporalStateEstimator().update(original, 0)
    assert result.actionable
    assert result.observation is original
    assert result.observation.detections is original.detections
    assert result.field("player_bbox").value == next(
        d.bbox for d in original.detections if d.class_id == "player"
    )
    assert result.last_accepted_sequence == result.last_accepted_timestamp_ms == 0
    for domain in ("inventory", "skills", "buffs"):
        field = result.field(domain)
        assert field.value is field.source_id is field.observed_at_ms is None
        assert field.status is Status.UNKNOWN
    with pytest.raises(KeyError):
        result.field("hidden_monster_hp")


def test_export_does_not_share_mutable_containers_and_is_deterministic():
    inputs = [observation(0), observation(1, monsters=(85,)), observation(2, monsters=())]
    first, second = TemporalStateEstimator(), TemporalStateEstimator()
    for item in inputs:
        left, right = first.update(item, item.timestamp_ms), second.update(item, item.timestamp_ms)
        assert left.to_dict() == right.to_dict()
        encoded = json.dumps(left.to_dict(), sort_keys=True, allow_nan=False)
        assert json.loads(encoded)["schema_version"] == 1
    exported = left.to_dict()
    exported["observation"]["detections"][0]["bbox"][0] = 999
    exported["fields"][0]["value"][0] = 999
    exported["tracks"][0]["bbox"][0] = 999
    assert left.to_dict() == right.to_dict()
    with pytest.raises(FrozenInstanceError):
        left.actionable = False
    with pytest.raises(FrozenInstanceError):
        left.tracks[0].bbox = (1, 1, 2, 2)


@pytest.mark.parametrize("invalid", [True, -1, 0.5, float("nan"), 2**53], ids=str)
def test_clock_rejects_wrong_types_and_out_of_budget_without_mutation(invalid):
    state = TemporalStateEstimator()
    good = state.update(observation(), 0)
    with pytest.raises(ValueError):
        state.update(observation(1), invalid)
    recovered = state.update(observation(1), 50)
    assert recovered.actionable and recovered.epoch == good.epoch


def test_detection_budget_is_checked_before_per_object_work(monkeypatch):
    state = TemporalStateEstimator(StateConfig(max_detections=2, max_tracks=2))
    oversized = observation(monsters=tuple(range(100, 103)))

    def forbidden(*args):
        raise AssertionError("must bound incoming count before per-object/association work")

    monkeypatch.setattr("maple_vision.state_estimation._box", forbidden)
    with pytest.raises(ValueError, match="detection budget"):
        state.update(oversized, 0)


@pytest.mark.parametrize(
    "invalid",
    [
        observation(source_id="x" * 257),
        observation(source_id="bad\nsource"),
        observation(map_id="x" * 81),
        observation(map_id="bad\x00map"),
        observation(sequence=2**53),
        observation(timestamp_ms=2**53),
        observation(detections=(Detection("x" * 65, (1, 1, 2, 2), 1),)),
        observation(detections=(Detection("player", (1, 1, 1000001, 2), 1),)),
    ],
    ids=[
        "long-source",
        "source-control",
        "long-map",
        "map-control",
        "sequence",
        "time",
        "class",
        "bbox",
    ],
)
def test_input_identity_and_geometry_budgets_are_validated_before_state_changes(invalid):
    state = TemporalStateEstimator()
    before = state.update(observation(), 0)
    with pytest.raises(ValueError):
        state.update(invalid, 50)
    after = state.update(observation(1), 50)
    assert after.actionable and after.epoch == before.epoch
    with pytest.raises(ValueError):
        state.update({}, 50)


@pytest.mark.parametrize(
    "bad,now,reason",
    [
        (observation(100000, 10000, map_id="quarry"), 50, "future_observation"),
        (observation(0), 50, "duplicate_or_reordered_observation"),
        (observation(100000, 0), 300, "stale_observation"),
        (observation(100000, 50, status=Status.UNKNOWN), 50, "unknown_observation"),
        (observation(100000, 50, confidence=0.1), 50, "low_observation_confidence"),
        (observation(100000, 50, hp_ratio=None), 50, "unknown_resources"),
        (observation(100000, 50, map_id=None), 50, "unknown_map"),
    ],
    ids=["future", "duplicate", "stale", "unknown", "low-confidence", "missing-hp", "missing-map"],
)
def test_rejected_frames_keep_their_identity_without_poisoning_next_valid_frame(bad, now, reason):
    state = TemporalStateEstimator()
    first = state.update(observation(), 0)
    rejected = state.update(bad, now)
    assert not rejected.actionable and rejected.reason == reason
    assert rejected.observation.sequence == bad.sequence
    assert rejected.observation.timestamp_ms == bad.timestamp_ms
    assert rejected.observation.detections == bad.detections
    assert rejected.observation.status is not Status.KNOWN
    assert rejected.last_accepted_sequence == rejected.last_accepted_timestamp_ms == 0
    assert rejected.epoch == first.epoch
    assert all(not t.observed_this_frame for t in rejected.tracks)
    recovered = state.update(observation(1, now + 50), now + 50)
    assert recovered.actionable and recovered.last_accepted_sequence == 1


def test_reordered_capture_time_does_not_poison_accepted_watermark():
    state = TemporalStateEstimator()
    state.update(observation(1, 100), 100)
    rejected = state.update(observation(100, 99), 150)
    assert rejected.reason == "observation_timestamp_rollback"
    assert rejected.last_accepted_sequence == 1
    assert state.update(observation(2, 200), 200).actionable


def test_clock_rollback_latches_nonactionable_and_clears_hypotheses():
    state = TemporalStateEstimator()
    state.update(observation(0, 100), 100)
    rejected = state.update(observation(1, 50), 50)
    assert not rejected.actionable and rejected.reason == "clock_rollback"
    assert not rejected.tracks
    assert all(field.value is None for field in rejected.fields)
    assert rejected.last_accepted_timestamp_ms == 100
    later = state.update(observation(2, 200), 200)
    assert not later.actionable and later.reason == "clock_rollback"
    assert TemporalStateEstimator().update(observation(2, 200), 200).actionable


def test_retained_measurements_expire_without_becoming_current_observations():
    state = TemporalStateEstimator()
    first = state.update(observation(), 0)
    lost = observation(1, 200, monsters=(), hp_ratio=None, mp_ratio=None, status=Status.UNKNOWN)
    interim = state.update(lost, 200)
    assert interim.observation.hp_ratio is None
    assert interim.field("hp_ratio").value == first.field("hp_ratio").value == 0.7
    assert not interim.field("hp_ratio").observed_this_frame
    assert interim.field("hp_ratio").observed_at_ms == 0
    assert interim.field("hp_ratio").expires_at_ms == 1000
    assert not interim.actionable
    assert monsters(interim)[0].association == "unobserved"
    assert not any(d.class_id == "monster" for d in interim.observation.detections)
    at_track_expiry = state.update(replace(lost, sequence=2, timestamp_ms=500), 500)
    assert not at_track_expiry.tracks
    assert at_track_expiry.field("hp_ratio").value == 0.7
    expired = state.update(replace(lost, sequence=3, timestamp_ms=1000), 1000)
    assert expired.field("hp_ratio").value is None
    assert expired.field("hp_ratio").status is Status.STALE
    assert expired.field("hp_ratio").confidence == 0
    assert not expired.actionable


def test_single_moving_object_keeps_id_and_velocity_is_only_a_hypothesis():
    state = TemporalStateEstimator()
    first = state.update(observation(), 0)
    second = state.update(observation(1, monsters=(85,)), 50)
    third = state.update(observation(2, monsters=(90,)), 100)
    assert (
        monsters(first)[0].track_id == monsters(second)[0].track_id == monsters(third)[0].track_id
    )
    assert monsters(second)[0].association == "matched"
    assert monsters(third)[0].velocity_px_per_second == (100.0, 0.0)
    missing = state.update(observation(3, monsters=()), 150)
    track = monsters(missing)[0]
    assert track.bbox == (90, 152, 100, 168)
    assert not track.observed_this_frame
    assert not any(d.class_id == "monster" for d in missing.observation.detections)


def test_reacquisition_before_expiry_and_new_identity_at_expiry():
    state = TemporalStateEstimator()
    initial = state.update(observation(), 0)
    old_id = monsters(initial)[0].track_id
    assert not monsters(state.update(observation(1, monsters=()), 50))[0].observed_this_frame
    reacquired = state.update(observation(2, monsters=(90,)), 100)
    assert monsters(reacquired, observed_only=True)[0].track_id == old_id
    assert monsters(reacquired, observed_only=True)[0].association == "reacquired"
    expired = state.update(observation(3, 600, monsters=(90,)), 600)
    assert monsters(expired)[0].track_id != old_id
    assert monsters(expired)[0].association == "new"


def test_crossing_indistinguishable_objects_produces_uncertain_identity_candidates():
    state = TemporalStateEstimator()
    before = state.update(observation(0, monsters=(70, 110)), 0)
    prior_ids = {track.track_id for track in monsters(before)}
    moving = state.update(observation(1, monsters=(80, 100)), 50)
    assert {t.track_id for t in monsters(moving, observed_only=True)} == prior_ids
    crossing = state.update(observation(2, monsters=(90, 90)), 100)
    current = monsters(crossing, observed_only=True)
    assert len(current) == 2
    assert all(track.identity_uncertain and track.association == "ambiguous" for track in current)
    assert all(set(track.candidate_ids) == prior_ids for track in current)
    assert not prior_ids & {track.track_id for track in current}
    assert all(track.identity_uncertain for track in monsters(crossing))
    # The current frame still has two measured boxes; no invented disappearance/death.
    assert crossing.observation.detections == observation(2, monsters=(90, 90)).detections


def test_detection_order_does_not_change_track_identity_assignment():
    original = observation(monsters=(180, 80))
    reversed_order = replace(original, detections=tuple(reversed(original.detections)))
    left = TemporalStateEstimator().update(original, 0)
    right = TemporalStateEstimator().update(reversed_order, 0)
    assert left.tracks == right.tracks
    assert right.observation.detections == reversed_order.detections


def test_identical_geometry_cannot_link_different_object_classes():
    state = TemporalStateEstimator()
    before = state.update(observation(), 0)
    old_id = monsters(before)[0].track_id
    following = observation(1, monsters=())
    following = replace(
        following,
        detections=following.detections + (Detection("loot", (80, 152, 90, 168), 0.9),),
    )
    after = state.update(following, 50)
    loot = next(t for t in after.tracks if t.class_id == "loot")
    assert loot.track_id != old_id and loot.association == "new"
    assert monsters(after)[0].association == "unobserved"


@pytest.mark.parametrize("change", [{"map_id": "cavern"}, {"source_id": "other-detector"}])
def test_map_or_source_transition_resets_all_identity_and_field_provenance(change):
    state = TemporalStateEstimator()
    before = state.update(observation(), 0)
    after = state.update(observation(1, **change), 50)
    assert after.epoch == before.epoch + 1
    assert not {t.track_id for t in before.tracks} & {t.track_id for t in after.tracks}
    assert all(t.association == "new" for t in after.tracks)
    assert all(
        field.observed_at_ms == 50
        for field in after.fields
        if field.name not in {"inventory", "skills", "buffs"}
    )


def test_track_storage_remains_bounded_under_endless_unmatched_detections():
    state = TemporalStateEstimator(
        StateConfig(max_tracks=4, max_detections=2, max_match_distance=10, track_ttl_ms=2000)
    )
    for index in range(30):
        current = state.update(observation(index, monsters=(100 + index * 100,)), index * 50)
        assert len(current.tracks) <= 4
        observed = [track for track in current.tracks if track.observed_this_frame]
        assert len(observed) == 2
        assert {track.class_id for track in observed} == {"player", "monster"}
        assert all(track.expires_at_ms > current.now_ms for track in current.tracks)


@pytest.mark.parametrize(
    "detection_tuple,reason",
    [
        ((), "ambiguous_player"),
        ((Detection("player", (1, 1, 10, 20), 0.1),), "low_player_confidence"),
        (
            (
                Detection("player", (1, 1, 10, 20), 0.9),
                Detection("player", (20, 1, 30, 20), 0.9),
            ),
            "ambiguous_player",
        ),
    ],
    ids=["absent-player", "low-confidence-player", "two-players"],
)
def test_a_usable_world_needs_one_current_measured_player(detection_tuple, reason):
    state = TemporalStateEstimator()
    state.update(observation(), 0)
    rejected = state.update(observation(1, detections=detection_tuple), 50)
    assert not rejected.actionable and rejected.reason == reason
    assert rejected.field("player_bbox").observed_at_ms == 0
    assert rejected.observation.detections == detection_tuple


def test_unknown_gameplay_domains_cannot_be_fabricated_as_measured_fields():
    for name in ("inventory", "skills", "buffs"):
        with pytest.raises(ValueError):
            FieldBelief(name, value=1, status=Status.KNOWN)
        with pytest.raises(ValueError, match="cannot be invented"):
            FieldBelief(name, 1, Status.KNOWN, 0.9, "detector", 0, 0, 500, True)


def test_current_zero_hp_is_nonactionable_despite_retained_positive_health():
    state = TemporalStateEstimator()
    state.update(observation(), 0)
    dead = state.update(observation(1, hp_ratio=0.0), 50)
    assert not dead.actionable and dead.reason == "player_dead"
    assert dead.observation.hp_ratio == 0.0
    assert dead.field("hp_ratio").value == 0.7
    assert not dead.field("hp_ratio").observed_this_frame
    assert dead.last_accepted_sequence == 0


def test_actionable_belief_cannot_claim_rejected_frame_was_accepted():
    valid = TemporalStateEstimator().update(observation(), 0)
    with pytest.raises(ValueError, match="accepted usable"):
        replace(valid, last_accepted_sequence=1)
    with pytest.raises(ValueError, match="accepted usable"):
        replace(valid, observation=replace(valid.observation, status=Status.UNKNOWN))
    assert isinstance(valid, BeliefState)
