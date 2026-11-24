"""CLI contracts for explicit learned terrain and bounded architecture audits."""

import hashlib
import json
from pathlib import Path

import pytest

from maple_vision.cli import main
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator


@pytest.mark.parametrize("operation", ["run", "suite", "soak", "reinfer"])
def test_model_terrain_requires_model_before_creating_lab_output(tmp_path, capsys, operation):
    output = tmp_path / "missing"
    target = [str(output)] if operation == "reinfer" else ["--output", str(output)]
    assert main(["lab", operation, *target, "--terrain-source", "model"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requires --model and --manifest" in captured.err
    assert not output.exists()


def test_observe_model_terrain_requires_model_before_opening_image(tmp_path, capsys):
    assert (
        main(["vision", "observe", str(tmp_path / "missing.png"), "--terrain-source", "model"]) == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requires --model and --manifest" in captured.err


def test_audit_cli_rejects_test_split_before_native_execution(tmp_path, capsys):
    output = tmp_path / "audit.json"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "model",
                "audit-terrain",
                str(tmp_path / "data"),
                "--output",
                str(output),
                "--split",
                "test",
            ]
        )
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not output.exists()


def test_audit_cli_passes_declared_parameters_and_packaged_default(tmp_path, capsys, monkeypatch):
    from maple_vision import terrain_audit

    received = {}

    def audit(dataset, config, output, **kwargs):
        received.update(dataset=dataset, config=config, output=output, **kwargs)
        return {"schema_version": 1, "scope": "structural_audit_only"}

    monkeypatch.setattr(terrain_audit, "audit_terrain_architecture", audit)
    data, output = tmp_path / "data", tmp_path / "audit.json"
    assert (
        main(
            [
                "model",
                "audit-terrain",
                str(data),
                "--output",
                str(output),
                "--split",
                "train",
                "--image-size",
                "384",
                "--warmup",
                "2",
                "--iterations",
                "7",
                "--seed",
                "91",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["scope"] == "structural_audit_only"
    assert received == {
        "dataset": data,
        "config": Path(__file__).parents[1]
        / "src"
        / "maple_vision"
        / "resources"
        / "terrain-yolo.yaml",
        "output": output,
        "split": "train",
        "image_size": 384,
        "warmup": 2,
        "iterations": 7,
        "seed": 91,
    }
    assert received["config"].is_file()


@pytest.mark.parametrize("operation", ["suite", "soak"])
def test_suite_and_soak_forward_model_terrain(tmp_path, capsys, monkeypatch, operation):
    from maple_vision import lab, onnx_detector

    detector = object()
    received = {}

    def run(output, count, **kwargs):
        received.update(output=output, count=count, **kwargs)
        return {"passed": True}

    monkeypatch.setattr(onnx_detector, "OnnxDetector", lambda *args: detector)
    monkeypatch.setattr(lab, f"run_{operation}", run)
    output = tmp_path / "out"
    assert (
        main(
            [
                "lab",
                operation,
                "--output",
                str(output),
                "--model",
                "local.onnx",
                "--manifest",
                "local.json",
                "--terrain-source",
                "model",
            ]
        )
        == 0
    )
    assert capsys.readouterr().err == ""
    assert received["terrain_source"] == "model"
    assert received["detector"] is detector
    assert received["output"] == output
    assert received["count"] == (3 if operation == "suite" else 60)


def test_real_model_cli_mode_is_bound_and_reinference_is_read_only(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    model_dir = Path(__file__).parents[1] / "examples" / "lifecycle-model"
    model = [
        "--model",
        str(model_dir / "model.onnx"),
        "--manifest",
        str(model_dir / "manifest.json"),
    ]
    image = tmp_path / "frame.png"
    image.write_bytes(SceneRenderer().render(Simulator().snapshot()).png_bytes())
    assert main(["vision", "observe", str(image), *model, "--terrain-source", "model"]) == 0
    observation = json.loads(capsys.readouterr().out)
    assert observation["source_id"] == "onnx-all-objects+fixture-hud-v1"
    output = tmp_path / "run"
    assert (
        main(
            [
                "lab",
                "run",
                "--output",
                str(output),
                "--max-steps",
                "3",
                "--record-stride",
                "1",
                *model,
                "--terrain-source",
                "model",
            ]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["perception_evidence"]["settings"]["terrain_source"] == "model"
    before = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    assert main(["lab", "reinfer", str(output), *model]) == 2
    assert "match" in capsys.readouterr().err
    assert main(["lab", "reinfer", str(output), *model, "--terrain-source", "model"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["perception_reproduced"] and result["policy_reproduced"]
    assert result["frames_reinferred"] == summary["steps"] + 1
    assert result["files_unchanged"]
    after = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    assert after == before
