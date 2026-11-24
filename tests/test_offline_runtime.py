"""Native inference must opt out before import, not after uploader initialization."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from maple_vision.onnx_detector import OnnxDetector


def test_package_sets_native_opt_out_before_optional_import():
    environment = dict(os.environ)
    environment["ORT_DISABLE_TELEMETRY"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import maple_vision,os,sys; "
            "assert os.environ['ORT_DISABLE_TELEMETRY']=='1'; "
            "assert 'onnxruntime' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_detector_sets_opt_out_before_loading_native_module(tmp_path, monkeypatch):
    payload = b"local fixture bytes"
    model = tmp_path / "fixture.onnx"
    model.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": hashlib.sha256(payload).hexdigest(),
                "class_names": ["player"],
                "input_width": 32,
                "input_height": 32,
                "output_format": "raw_xywh",
                "scope": "synthetic_fixture",
            }
        )
    )
    calls = []

    def import_runtime(name):
        assert name == "onnxruntime"
        assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
        calls.append("import")

        def session(*args, **kwargs):
            assert calls == ["import", "disable"]
            return SimpleNamespace(
                get_inputs=lambda: [
                    SimpleNamespace(type="tensor(float)", shape=[1, 3, 32, 32], name="x")
                ],
                get_outputs=lambda: [SimpleNamespace()],
            )

        return SimpleNamespace(
            disable_telemetry_events=lambda: calls.append("disable"),
            SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=0),
            InferenceSession=session,
        )

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr("maple_vision.onnx_detector.importlib.import_module", import_runtime)
    OnnxDetector(model, manifest)
    assert calls == ["import", "disable"]
