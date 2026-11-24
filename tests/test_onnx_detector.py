import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from maple_vision.onnx_detector import (
    Letterbox,
    ModelManifest,
    OnnxDetector,
    decode,
    preprocess,
)
from maple_vision.scene import ImageFrame


def manifest(output_format="raw_xywh"):
    return ModelManifest(
        "0" * 64, ("player", "monster"), 64, 64, output_format, "synthetic_fixture"
    )


def test_letterbox_rgb_normalization_and_inverse_geometry():
    frame = ImageFrame(0, 0, 40, 20, bytes([255, 0, 0]) * 800)
    tensor, transform = preprocess(frame, 64, 64)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert transform == Letterbox(1.6, 1.6, 0, 16, 40, 20)
    assert tensor[0, :, 20, 10].tolist() == [1.0, 0.0, 0.0]
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)
    output = np.array([[[32, 32, 32, 16, 0.9, 0.1]]], dtype=np.float32).transpose(0, 2, 1)
    result = decode(output, manifest(), transform)
    assert len(result) == 1
    assert result[0].bbox == pytest.approx((10, 5, 30, 15))


def test_raw_nms_is_class_aware_and_rejects_invalid_predictions():
    rows = np.array(
        [
            [20, 20, 10, 10, 0.9, 0.1],
            [20, 20, 10, 10, 0.8, 0.1],
            [20, 20, 10, 10, 0.1, 0.85],
            [20, 20, -10, 10, 0.99, 0.0],
            [20, 20, 10, 10, float("nan"), 0.0],
            [20, 20, 10, 10, 2.0, 0.0],
        ]
    )
    result = decode(rows.T[None], manifest(), Letterbox(1, 1, 0, 0, 64, 64))
    assert [item.class_id for item in result] == ["player", "monster"]


def test_end_to_end_does_not_repeat_nms_and_clips_padding():
    rows = np.array([[[-4, -5, 30, 30, 0.8, 0], [0, 0, 30, 30, 0.7, 0]]])
    result = decode(rows, manifest("end2end_xyxy"), Letterbox(1, 1, 0, 0, 64, 64))
    assert len(result) == 2
    assert result[0].bbox == (0.0, 0.0, 30.0, 30.0)


@pytest.mark.parametrize("output", [np.zeros((6, 10)), np.zeros((2, 6, 10)), np.zeros((1, 5, 10))])
def test_wrong_output_contract_fails(output):
    with pytest.raises(ValueError):
        decode(output, manifest(), Letterbox(1, 1, 0, 0, 64, 64))


def test_real_onnx_runtime_contract_and_checksum(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    # An explicit constant test graph exercises the native runtime, not model accuracy.
    values = np.array([[[4, 5, 20, 25, 0.9, 1]]], dtype=np.float32)
    node = onnx.helper.make_node(
        "Constant", [], ["detections"], value=onnx.numpy_helper.from_array(values)
    )
    graph = onnx.helper.make_graph(
        [node],
        "adapter-contract-test",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 64, 64])],
        [onnx.helper.make_tensor_value_info("detections", onnx.TensorProto.FLOAT, [1, 1, 6])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )
    model_path, manifest_path = tmp_path / "fixture.onnx", tmp_path / "manifest.json"
    onnx.save(model, model_path)
    data = asdict(manifest("end2end_xyxy"))
    data["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    detector = OnnxDetector(model_path, manifest_path)
    assert detector.loaded_model_sha256 == data["model_sha256"]
    detections = detector.detect(ImageFrame(0, 0, 64, 64, bytes(64 * 64 * 3)))
    assert detections[0].class_id == "monster"
    assert detections[0].bbox == (4, 5, 20, 25)
    model_path.write_bytes(b"corrupt")
    assert detector.loaded_model_sha256 == data["model_sha256"]
    with pytest.raises(AttributeError):
        detector.loaded_model_sha256 = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        OnnxDetector(model_path, manifest_path)


def test_manifest_rejects_unknown_fields_and_class_order_duplicates(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**asdict(manifest()), "guessed_layout": True}))
    with pytest.raises(ValueError, match="fields"):
        ModelManifest.load(path)
    with pytest.raises(ValueError, match="distinct"):
        ModelManifest("0" * 64, ("player", "player"), 64, 64, "raw_xywh", "synthetic_fixture")
