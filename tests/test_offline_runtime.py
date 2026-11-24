"""Native inference must opt out before import, not after uploader initialization."""

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import maple_vision.training as training
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


@pytest.fixture
def training_runtime(monkeypatch):
    """Exercise the real context manager without loading native dependencies."""
    state = SimpleNamespace(threads=7)
    torch = SimpleNamespace(
        get_num_threads=lambda: state.threads,
        set_num_threads=lambda value: setattr(state, "threads", value),
    )
    logger = logging.Logger("offline-stream-test", level=logging.ERROR)
    settings = {"sync": True, "wandb": True}
    events = SimpleNamespace(enabled=True)
    utils = SimpleNamespace(SETTINGS=settings, LOGGER=logger, ONLINE=True)
    modules = {
        "torch": torch,
        "ultralytics": SimpleNamespace(),
        "ultralytics.utils": utils,
        "ultralytics.utils.torch_utils": SimpleNamespace(),
        "ultralytics.nn.tasks": SimpleNamespace(),
        "ultralytics.utils.checks": SimpleNamespace(),
        "ultralytics.utils.events": SimpleNamespace(events=events),
    }
    original_import = training.importlib.import_module

    def import_runtime(name):
        return modules[name] if name in modules else original_import(name)

    monkeypatch.setattr(training.importlib, "import_module", import_runtime)
    return SimpleNamespace(
        logger=logger, state=state, settings=settings, events=events, utils=utils
    )


class ObservedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_calls = 0
        self.flush_error = None

    def flush(self):
        self.flush_calls += 1
        if self.flush_error is not None:
            raise self.flush_error
        super().flush()


def assert_training_state_restored(runtime):
    assert runtime.state.threads == 7
    assert runtime.settings == {"sync": True, "wandb": True}
    assert runtime.events.enabled is True
    assert runtime.utils.ONLINE is True


@pytest.mark.parametrize("previous_closed", [False, True])
@pytest.mark.parametrize("operation_fails", [False, True])
def test_training_logging_routes_and_restores_existing_handler(
    tmp_path, monkeypatch, training_runtime, previous_closed, operation_fails
):
    previous, current = ObservedStream(), ObservedStream()
    handler = logging.StreamHandler(previous)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("worker: %(message)s")
    handler.setFormatter(formatter)
    training_runtime.logger.addHandler(handler)
    if previous_closed:
        previous.close()
    before_flushes = previous.flush_calls
    original_stdout = sys.stdout
    original_offline = os.environ.get("YOLO_OFFLINE")
    failure = RuntimeError("training operation failed")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "stderr", current)
            try:
                with training._offline_runtime(tmp_path):
                    assert handler.stream is current
                    assert training_runtime.logger.handlers == [handler]
                    assert handler.level == logging.ERROR and handler.formatter is formatter
                    assert training_runtime.state.threads == 2
                    training_runtime.logger.error("still visible")
                    assert current.getvalue() == "worker: still visible\n"
                    if operation_fails:
                        raise failure
            except RuntimeError as exc:
                assert operation_fails and exc is failure
            else:
                assert not operation_fails
        assert handler.stream is previous
        assert previous.flush_calls == before_flushes + (not previous_closed)
        # One flush from logging the record, one when restoring the old stream.
        assert current.flush_calls == 2
        assert sys.stdout is original_stdout
        assert os.environ.get("YOLO_OFFLINE") == original_offline
        assert_training_state_restored(training_runtime)
    finally:
        training_runtime.logger.removeHandler(handler)
        handler.close()
        previous.close()
        current.close()


@pytest.mark.parametrize("operation_fails", [False, True])
def test_training_logging_restores_when_temporary_stderr_is_closed(
    tmp_path, monkeypatch, training_runtime, operation_fails
):
    previous, current = ObservedStream(), ObservedStream()
    handler = logging.StreamHandler(previous)
    training_runtime.logger.addHandler(handler)
    failure = RuntimeError("operation closed its capture stream")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "stderr", current)
            try:
                with training._offline_runtime(tmp_path):
                    training_runtime.logger.error("before close")
                    current.close()
                    if operation_fails:
                        raise failure
            except RuntimeError as exc:
                assert operation_fails and exc is failure
            else:
                assert not operation_fails
        assert handler.stream is previous
        assert current.flush_calls == 1  # Only the log emission, never the closed stream.
        training_runtime.logger.error("after restore")
        assert previous.getvalue() == "after restore\n"
        assert_training_state_restored(training_runtime)
    finally:
        training_runtime.logger.removeHandler(handler)
        handler.close()
        previous.close()
        current.close()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("I/O operation on closed file."),
        ValueError("invalid flush"),
        OSError("I/O fault"),
    ],
)
def test_training_logging_does_not_hide_open_stream_flush_errors(
    tmp_path, monkeypatch, training_runtime, error
):
    first, broken, current = ObservedStream(), ObservedStream(), ObservedStream()
    first_handler, broken_handler = logging.StreamHandler(first), logging.StreamHandler(broken)
    training_runtime.logger.addHandler(first_handler)
    training_runtime.logger.addHandler(broken_handler)
    broken.flush_error = error
    original_offline = os.environ.get("YOLO_OFFLINE")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "stderr", current)
            with pytest.raises(type(error)) as caught:
                with training._offline_runtime(tmp_path):
                    pytest.fail("an unrelated stream error must prevent entering the runtime")
        assert caught.value is error
        assert not broken.closed
        assert first_handler.stream is first  # Undo the earlier successful switch too.
        assert broken_handler.stream is broken
        assert os.environ.get("YOLO_OFFLINE") == original_offline
        assert_training_state_restored(training_runtime)
    finally:
        broken.flush_error = None
        for handler in (first_handler, broken_handler):
            training_runtime.logger.removeHandler(handler)
            handler.close()
        for stream in (first, broken, current):
            stream.close()


def test_training_logging_propagates_open_stream_error_during_restore(
    tmp_path, monkeypatch, training_runtime
):
    previous, current = ObservedStream(), ObservedStream()
    handler = logging.StreamHandler(previous)
    training_runtime.logger.addHandler(handler)
    failure = OSError("temporary stderr could not flush")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "stderr", current)
            with pytest.raises(OSError) as caught:
                with training._offline_runtime(tmp_path):
                    current.flush_error = failure
        assert caught.value is failure
        assert not current.closed
        assert handler.stream is current  # A failed flush must not silently discard a buffer.
        assert_training_state_restored(training_runtime)
    finally:
        current.flush_error = None
        handler.setStream(previous)
        training_runtime.logger.removeHandler(handler)
        handler.close()
        previous.close()
        current.close()


def test_training_logging_keeps_an_unchanged_stream_without_flushing(
    tmp_path, monkeypatch, training_runtime
):
    stream = ObservedStream()
    handler = logging.StreamHandler(stream)
    training_runtime.logger.addHandler(handler)
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "stderr", stream)
            with training._offline_runtime(tmp_path):
                assert handler.stream is stream
                assert stream.flush_calls == 0
        assert handler.stream is stream and stream.flush_calls == 0
        assert_training_state_restored(training_runtime)
    finally:
        training_runtime.logger.removeHandler(handler)
        handler.close()
        stream.close()
