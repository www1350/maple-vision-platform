from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.desktop_observation import verify_desktop_observation_evidence
from maple_vision.desktop_observation_check import run
from maple_vision.desktop_window import WindowIdentity


def test_observation_command_requires_explicit_fixture_mode_and_verifies(
    monkeypatch, tmp_path: Path
) -> None:
    identity = WindowIdentity(9, 10, 11, 12, "Owned", "fixture", (0, 0, 320, 192), 96)

    class Backend:
        def list_windows(self) -> tuple[WindowIdentity, ...]:
            return (identity,)

        def describe(self, handle: int) -> WindowIdentity | None:
            return identity if handle == identity.handle else None

        def foreground_handle(self) -> int | None:
            return identity.handle

        def capture(self, selected: WindowIdentity) -> Image.Image:
            assert selected == identity
            return Image.new("RGB", (320, 192), "#123456")

    monkeypatch.setattr("maple_vision.desktop_observation_check.NativeWindowsBackend", Backend)
    monkeypatch.setattr("maple_vision.desktop_observation_check.time.sleep", lambda _: None)
    output = tmp_path / "result"
    report = run(
        [
            "--title-exact",
            "fixture",
            "--output",
            str(output),
            "--perception",
            "synthetic-fixture",
            "--delay-seconds",
            "0",
        ]
    )
    assert report["verification"]["verified"] is True
    assert report["verification"]["status"] == "unknown"
    assert (output / "recognized-overlay.png").is_file()


def test_observation_command_loads_hash_bound_onnx_without_claiming_hud(
    monkeypatch, tmp_path: Path
) -> None:
    identity = WindowIdentity(19, 20, 21, 22, "Owned", "model", (0, 0, 320, 192), 96)

    class Backend:
        def list_windows(self) -> tuple[WindowIdentity, ...]:
            return (identity,)

        def describe(self, handle: int) -> WindowIdentity | None:
            return identity if handle == identity.handle else None

        def foreground_handle(self) -> int | None:
            return identity.handle

        def capture(self, selected: WindowIdentity) -> Image.Image:
            return Image.new("RGB", (320, 192), "#123456")

    monkeypatch.setattr("maple_vision.desktop_observation_check.NativeWindowsBackend", Backend)
    monkeypatch.setattr("maple_vision.desktop_observation_check.time.sleep", lambda _: None)
    model_root = Path(__file__).resolve().parents[1] / "examples" / "lifecycle-model"
    output = tmp_path / "model-result"
    report = run(
        [
            "--title-exact",
            "model",
            "--output",
            str(output),
            "--perception",
            "onnx-detector",
            "--model",
            str(model_root / "model.onnx"),
            "--manifest",
            str(model_root / "manifest.json"),
            "--delay-seconds",
            "0",
        ]
    )
    assert report["verification"]["perception_mode"] == "onnx_detector_only_v1"
    assert report["verification"]["status"] == "unknown"
    identity_report = report["verification"]["perception_identity"]
    assert (
        identity_report["loaded_model_sha256"] == identity_report["model_manifest"]["model_sha256"]
    )
    evidence = output / "evidence.json"
    changed = json.loads(evidence.read_text(encoding="utf-8"))
    changed["belief"]["actionable"] = True
    evidence.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="invented an actionable HUD state"):
        verify_desktop_observation_evidence(output)
