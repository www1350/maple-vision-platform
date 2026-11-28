"""Installed command routing for read-only recording tracking and verification."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert (output / "preview.gif").is_file()
    assert len((output / "tracks.jsonl").read_text().splitlines()) == 5
    check = run_cli(tmp_path, "verify-tracking", str(recording), str(output))
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["verified"] is True


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
