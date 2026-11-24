"""The shipped trained checkpoint must really run the simulator mission."""

from pathlib import Path

import pytest

from maple_vision.lab import RunConfig, run_episode
from maple_vision.onnx_detector import OnnxDetector


@pytest.mark.parametrize("job", ["assassin", "dragon_knight", "cleric"])
def test_shipped_yolo_checkpoint_closed_loop(job):
    pytest.importorskip("onnxruntime")
    directory = Path(__file__).parents[1] / "examples" / "fixture-model"
    detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")
    result = run_episode(config=RunConfig(job=job), detector=detector)
    assert result["status"] == "completed", result
    assert result["independent_goal_check"]
    assert result["drops_collected"] == 6
    assert result["events"]["monster_defeated"] == 6
    if job == "assassin":
        assert result["events"]["jump"] == 2
