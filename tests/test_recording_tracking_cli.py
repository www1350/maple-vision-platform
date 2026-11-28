"""Installed command routing for read-only recording tracking and verification."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from maple_vision.movement_hypotheses import MovementConfig
from maple_vision.recording_tracking import create_recording_actor_selection, track_recording
from maple_vision.recordings import import_recording
from maple_vision.scene import SceneRenderer
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator
from maple_vision.state_estimation import StateConfig


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    simulator = Simulator(seed=9)
    renderer = SceneRenderer()
    for index in range(5):
        (source / f"{index:04d}.png").write_bytes(renderer.render(simulator.snapshot()).png_bytes())
        simulator.step(SimAction("move", direction=1))
    destination = tmp_path / "recording"
    import_recording(
        source,
        destination,
        source_uri="urn:self:tracking-cli",
        permission="self_authored",
        license="CC0-1.0",
        fps=10,
    )
    return destination


def run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "maple_vision", "recording", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_cli_tracks_and_independently_verifies_one_window(recording: Path, tmp_path: Path) -> None:
    config = tmp_path / "state.json"
    config.write_text(json.dumps(StateConfig(track_ttl_ms=900).to_dict()), encoding="utf-8")
    output = tmp_path / "tracked"
    result = run_cli(
        tmp_path,
        "track",
        str(recording),
        "--output",
        str(output),
        "--state-config",
        str(config),
        "--preview-stride",
        "2",
        "--preview-max-frames",
        "3",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["frames_processed"] == 5
    assert report["preview_frames"] == 3
    assert report["state_config"]["track_ttl_ms"] == 900
    assert report["action_execution"] == "none"
    assert report["accuracy_metrics_available"] is False
    assert "movement_analysis" not in report
    assert (output / "preview.gif").is_file()
    assert len((output / "tracks.jsonl").read_text().splitlines()) == 5
    check = run_cli(tmp_path, "verify-tracking", str(recording), str(output))
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["verified"] is True


@pytest.mark.parametrize("custom_configuration", [False, True])
def test_cli_opt_in_movement_is_recomputed_from_saved_configuration(
    recording: Path, tmp_path: Path, custom_configuration: bool
) -> None:
    output = tmp_path / "movement"
    config = MovementConfig(max_climb_to_arrival_ms=900)
    options = ["--movement"]
    if custom_configuration:
        path = tmp_path / "movement-config.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        options = ["--movement-config", str(path)]
    result = run_cli(tmp_path, "track", str(recording), "--output", str(output), *options)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    analysis = report["movement_analysis"]
    assert analysis["frames_processed"] == 5
    assert sum(analysis["state_counts"].values()) == 5
    assert analysis["owner_identity_verified"] is False
    assert analysis["real_domain_accuracy_available"] is False
    if custom_configuration:
        assert analysis["config"] == config.to_dict()
    checked = run_cli(tmp_path, "verify-tracking", str(recording), str(output))
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["movement_recomputed"] is True


def test_cli_rejects_incomplete_movement_configuration_before_publication(
    recording: Path, tmp_path: Path
) -> None:
    config = tmp_path / "movement.json"
    config.write_text('{"schema_version":1,"camera_min_landmarks":1}', encoding="utf-8")
    output = tmp_path / "invalid-movement"
    result = run_cli(
        tmp_path,
        "track",
        str(recording),
        "--output",
        str(output),
        "--movement-config",
        str(config),
    )
    assert result.returncode == 2
    assert not output.exists()


def test_cli_verification_does_not_fall_back_to_recording_check(
    recording: Path, tmp_path: Path
) -> None:
    output = tmp_path / "tracked"
    created = run_cli(tmp_path, "track", str(recording), "--output", str(output))
    assert created.returncode == 0, created.stderr
    (output / "tracks.jsonl").write_text("{}\n", encoding="utf-8")
    checked = run_cli(tmp_path, "verify-tracking", str(recording), str(output))
    assert checked.returncode == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ("--model", "missing.onnx"),
        ("--manifest", "missing.json"),
        ("--preview-stride", "0"),
        ("--preview-max-frames", "121"),
    ],
)
def test_cli_rejects_incomplete_model_or_invalid_preview_without_output(
    recording: Path, tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    output = tmp_path / "invalid"
    result = run_cli(tmp_path, "track", str(recording), "--output", str(output), *arguments)
    assert result.returncode == 2
    assert not output.exists()


def test_cli_rejects_invalid_state_config_before_publication(
    recording: Path, tmp_path: Path
) -> None:
    config = tmp_path / "bad-state.json"
    config.write_text('{"schema_version":1,"unknown":true}', encoding="utf-8")
    output = tmp_path / "invalid"
    result = run_cli(
        tmp_path,
        "track",
        str(recording),
        "--output",
        str(output),
        "--state-config",
        str(config),
    )
    assert result.returncode == 2
    assert not output.exists()


def test_existing_observe_command_keeps_its_original_output(
    recording: Path, tmp_path: Path
) -> None:
    output = tmp_path / "observed"
    result = run_cli(tmp_path, "observe", str(recording), "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "perception_only_replay"
    assert report["frames_processed"] == 5
    assert (output / "observations.jsonl").is_file()
    assert not (output / "tracks.jsonl").exists()


def test_cli_movement_review_keeps_unreviewed_and_uncertain_frames_unscored(
    recording: Path, tmp_path: Path
) -> None:
    labels_path = tmp_path / "movement-labels.json"
    created = run_cli(tmp_path, "movement-labels", str(recording), "--output", str(labels_path))
    assert created.returncode == 0, created.stderr
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert all(frame["review"] is None for frame in labels["frames"])
    labels.update(domain="authored_fixture", reviewer="CLI independent review", subject="actor A")
    labels["frames"][0]["review"] = {
        "actor_bbox": None,
        "rope_climb_visible": None,
        "arrival_after_climb_visible": None,
        "note": "State cannot be determined from this frame.",
    }
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    replay = tmp_path / "movement-replay"
    tracked = run_cli(tmp_path, "track", str(recording), "--movement", "--output", str(replay))
    assert tracked.returncode == 0, tracked.stderr
    output = tmp_path / "movement-benchmark.json"
    scored = run_cli(
        tmp_path,
        "benchmark-movement",
        str(recording),
        str(replay),
        "--labels",
        str(labels_path),
        "--output",
        str(output),
        "--actor-iou-threshold",
        "0.75",
    )
    assert scored.returncode == 0, scored.stderr
    report = json.loads(scored.stdout)
    assert report["verification"]["movement_recomputed"] is True
    assert report["frames"] == report["verification"]["frames_verified"] == 5
    assert report["coverage"]["unreviewed"] == 4
    assert report["coverage"]["reviewed_uncertain"] == 1
    for metric in report["predicates"].values():
        assert metric["positives"] == metric["negatives"] == 0
        assert metric["precision"] is metric["recall"] is None
        assert metric["decision_coverage"] is None
    assert report["action_execution"] == "none"
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("option", ["--model", "--manifest"])
@pytest.mark.parametrize("operation", ["benchmark-movement", "benchmark-actor"])
def test_cli_recording_benchmark_requires_complete_model_identity(
    recording: Path, tmp_path: Path, option: str, operation: str
) -> None:
    output = tmp_path / "invalid-benchmark.json"
    checked = run_cli(
        tmp_path,
        operation,
        str(recording),
        str(tmp_path / "replay"),
        "--labels",
        str(tmp_path / "labels.json"),
        "--output",
        str(output),
        option,
        "missing-model-file",
    )
    assert checked.returncode == 2
    assert "--model and --manifest must be supplied together" in checked.stderr
    assert not output.exists()


def test_cli_actor_review_scores_selected_target_without_movement(
    recording: Path, tmp_path: Path
) -> None:
    labels_path = tmp_path / "actor-labels.json"
    created = run_cli(tmp_path, "actor-labels", str(recording), "--output", str(labels_path))
    assert created.returncode == 0, created.stderr
    labels = json.loads(created.stdout)
    assert labels == json.loads(labels_path.read_text(encoding="utf-8"))
    assert all(row["review"] is None for row in labels["frames"])
    labels.update(domain="authored_fixture", reviewer="independent CLI review", subject="actor A")
    simulator = Simulator(seed=9)
    for row in labels["frames"]:
        player = simulator.snapshot().player
        row["review"] = {
            "visibility": "visible",
            "actor_bbox": [
                int(player.x),
                int(player.y),
                int(player.x + player.w),
                int(player.y + player.h),
            ],
            "other_actors": [],
            "note": "Independent source fixture geometry, not a prediction.",
        }
        simulator.step(SimAction("move", direction=1))
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    selection = create_recording_actor_selection(
        recording,
        tmp_path / "selection.json",
        source_sequence=0,
        bbox=tuple(labels["frames"][0]["review"]["actor_bbox"]),
        reviewer="CLI fixture anchor",
        subject="actor A",
    )
    replay = tmp_path / "actor-replay"
    tracked = track_recording(recording, replay, actor_selection=selection)
    assert "movement_analysis" not in tracked
    output = tmp_path / "actor-benchmark.json"
    arguments = (
        "benchmark-actor",
        str(recording),
        str(replay),
        "--labels",
        str(labels_path),
        "--output",
        str(output),
        "--actor-iou-threshold",
        "0.75",
    )
    scored = run_cli(tmp_path, *arguments)
    assert scored.returncode == 0, scored.stderr
    report = json.loads(scored.stdout)
    assert report["kind"] == "recording_actor_benchmark"
    assert report["verification"]["actor_selection_recomputed"] is True
    assert report["frames"] == 5
    assert [row["outcome"] for row in report["diagnostics"]] == ["correct_target"] * 5
    assert report["action_execution"] == "none"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    previous = output.read_bytes()
    assert run_cli(tmp_path, *arguments).returncode == 2
    assert output.read_bytes() == previous


def test_cli_selects_a_source_bound_actor_and_reverifies_without_original_anchor_file(
    recording: Path, tmp_path: Path
) -> None:
    # Bounds come from the independently rendered fixture, not a prediction row.
    player = Simulator(seed=9).snapshot().player
    bbox = (player.x, player.y, player.x + player.w, player.y + player.h)
    selection = tmp_path / "actor-selection.json"
    created = run_cli(
        tmp_path,
        "select-actor",
        str(recording),
        "--source-frame",
        "0",
        "--bbox",
        *(str(value) for value in bbox),
        "--reviewer",
        "independent CLI fixture review",
        "--subject",
        "visible cyan actor",
        "--output",
        str(selection),
    )
    assert created.returncode == 0, created.stderr
    specification = json.loads(created.stdout)
    assert specification["kind"] == "recording_actor_selection"
    assert specification["anchor"]["source_sequence"] == 0
    replay = tmp_path / "selected-replay"
    tracked = run_cli(
        tmp_path,
        "track",
        str(recording),
        "--actor-selection",
        str(selection),
        "--movement",
        "--output",
        str(replay),
    )
    assert tracked.returncode == 0, tracked.stderr
    report = json.loads(tracked.stdout)
    assert report["actor_selection_analysis"]["specification"] == specification
    assert report["actor_selection_analysis"]["status_counts"] == {"selected": 5}
    assert report["retain_multiple_player_hypotheses"] is True
    assert report["perception_mode"] == "detector_only"
    assert report["detector_metadata"]["observation_adapter"] == "DetectorOnlyPerception.observe"
    selection.unlink()
    verified = run_cli(tmp_path, "verify-tracking", str(recording), str(replay))
    assert verified.returncode == 0, verified.stderr
    result = json.loads(verified.stdout)
    assert result["actor_selection_recomputed"] is True
    assert result["actor_selection_status_counts"] == {"selected": 5}


@pytest.mark.parametrize("source_frame", [-1, 5])
def test_cli_actor_anchor_cannot_select_a_frame_outside_the_source_window(
    recording: Path, tmp_path: Path, source_frame: int
) -> None:
    output = tmp_path / "invalid-selection.json"
    result = run_cli(
        tmp_path,
        "select-actor",
        str(recording),
        "--source-frame",
        str(source_frame),
        "--bbox",
        "20",
        "146",
        "34",
        "168",
        "--reviewer",
        "fixture reviewer",
        "--subject",
        "fixture player",
        "--output",
        str(output),
    )
    assert result.returncode == 2
    assert not output.exists()
