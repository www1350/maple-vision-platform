"""Local-only, CPU-bounded YOLO training and verified ONNX export.

Heavy dependencies are imported only inside these commands. Run training in a
separate CLI process: the offline guard and library settings are process-wide
for the duration of the operation, and are restored afterwards.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
import math
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from typing import Any, NoReturn
from unittest.mock import patch

from PIL import Image

from .catalog import canonical_json, strict_json
from .dataset import _contained, validate_dataset
from .onnx_detector import ModelManifest
from .sprite_pack import verify_dataset_sprite_pack

_RUNTIME_LOCK = Lock()
_SAFE_MODULES = {"Conv", "C2f", "SPPF", "Concat", "Detect", "nn.Upsample"}
_SAFE_MODEL_KEYS = {
    "nc",
    "depth_multiple",
    "width_multiple",
    "reg_max",
    "end2end",
    "backbone",
    "head",
}


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _image_size(value: object) -> int:
    result = _integer(value, "image_size", 64, 1280)
    if result % 32:
        raise ValueError("image_size must be a multiple of 32")
    return result


def _local_file(path: Path, suffixes: set[str], field: str) -> Path:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError(f"{field} must be an existing local {'/'.join(sorted(suffixes))} file")
    return path.resolve()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _validate_model_config(path: Path) -> None:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise ValueError("install the training extra for local YOLO training") from exc
    if path.stat().st_size > 64 * 1024:
        raise ValueError("model config exceeds 64 KiB")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or not {"nc", "backbone", "head"} <= set(config)
        or not set(config) <= _SAFE_MODEL_KEYS
    ):
        raise ValueError("model config needs nc/backbone/head and only supported safe keys")
    _integer(config["nc"], "model nc", 1, 1000)
    for key in ("depth_multiple", "width_multiple"):
        value = config.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 2:
            raise ValueError(f"model {key} must be finite in (0, 2]")
    _integer(config.get("reg_max", 16), "model reg_max", 1, 32)
    if type(config.get("end2end", False)) is not bool:
        raise ValueError("model end2end must be boolean")
    layers: list[Any] = []
    for section in ("backbone", "head"):
        value = config[section]
        if not isinstance(value, list) or not value:
            raise ValueError(f"model {section} must contain layers")
        layers.extend(value)
    if len(layers) > 64:
        raise ValueError("model config exceeds 64 layers")
    for index, layer in enumerate(layers):
        if not isinstance(layer, list) or len(layer) != 4:
            raise ValueError("each model layer needs from/repeats/module/args")
        reference, repeats, module, args = layer
        refs = reference if isinstance(reference, list) else [reference]
        if not refs or any(type(r) is not int or not -max(1, index) <= r < index for r in refs):
            raise ValueError("invalid model layer reference")
        _integer(repeats, "layer repeats", 1, 8)
        if not isinstance(module, str) or module not in _SAFE_MODULES:
            raise ValueError("unsupported model layer; only local detection primitives allowed")
        if not isinstance(args, list) or len(args) > 8:
            raise ValueError("model arguments must be a short primitive list")
        for value in args:
            if isinstance(value, str):
                if value not in {"nc", "nearest"}:
                    raise ValueError("unsupported model string argument")
            elif value is not None and not isinstance(value, (bool, int, float)):
                raise ValueError("model arguments must be primitive values")
            elif isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("model arguments must be finite")
            elif isinstance(value, (int, float)) and abs(value) > 2048:
                raise ValueError("model argument exceeds the bounded fixture limit")
    if layers[-1][2] != "Detect":
        raise ValueError("model must end in a detection head")


def _blocked_network(*args: Any, **kwargs: Any) -> NoReturn:
    raise OSError("network access is disabled during offline training/export")


def _set_logging_stream(handler: logging.StreamHandler[Any], stream: Any) -> None:
    """Preserve normal flushing, but do not flush an already-closed capture stream."""
    handler.acquire()
    try:
        if getattr(handler.stream, "closed", False) is True:
            # A previous caller may have closed its redirected stdout/stderr.
            # There is no remaining buffer to flush; retain the handler itself.
            handler.stream = stream
        else:
            # Do not suppress unrelated flush failures on a usable stream.
            handler.setStream(stream)
    finally:
        handler.release()


@contextmanager
def _offline_runtime(directory: Path) -> Iterator[tuple[Any, Any]]:
    """Suppress downloads/telemetry and cap CPU threads without persisting settings.

    The socket guard is a cooperative process guard, not a security sandbox. It
    prevents accidental library network calls, including optional font downloads.
    """
    with _RUNTIME_LOCK, ExitStack() as stack:
        # Ultralytics installs a global Pillow loader with a lazy HEIF dependency
        # installer. Our verified PNG input needs no such fallback. Preserve the
        # caller's loader even if a dependency import or the operation itself fails.
        original_image_open = Image.open
        stack.callback(setattr, Image, "open", original_image_open)
        stack.enter_context(redirect_stdout(sys.stderr))
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "YOLO_OFFLINE": "true",
                    "ORT_DISABLE_TELEMETRY": "1",
                    "YOLO_AUTOINSTALL": "false",
                    "YOLO_VERBOSE": "false",
                    "ULTRALYTICS_SAFE_LOAD": "true",
                    "YOLO_CONFIG_DIR": str(directory / "ultralytics"),
                    "WANDB_MODE": "disabled",
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                },
            )
        )
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            stack.enter_context(patch(target, _blocked_network))
        try:
            torch = importlib.import_module("torch")
            ultralytics = importlib.import_module("ultralytics")
            utils = importlib.import_module("ultralytics.utils")
            torch_utils = importlib.import_module("ultralytics.utils.torch_utils")
            tasks = importlib.import_module("ultralytics.nn.tasks")
            checks = importlib.import_module("ultralytics.utils.checks")
        except ImportError as exc:
            raise ValueError(
                "install maple-vision-platform[training] for YOLO training/export"
            ) from exc
        finally:
            # Undo the import side effect before any training/export image reads,
            # not only when leaving the context manager.
            Image.open = original_image_open
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        stack.callback(torch.set_num_threads, old_threads)
        for module, key, value in (
            (utils, "ONLINE", False),
            (utils, "AUTOINSTALL", False),
            (utils, "NUM_THREADS", 2),
            (checks, "ONLINE", False),
            (checks, "AUTOINSTALL", False),
            (torch_utils, "NUM_THREADS", 2),
            (tasks, "SAFE_LOAD", True),
        ):
            if hasattr(module, key):
                stack.enter_context(patch.object(module, key, value))
        settings = utils.SETTINGS
        toggles = {
            key: False
            for key in (
                "sync",
                "clearml",
                "comet",
                "dvc",
                "mlflow",
                "raytune",
                "tensorboard",
                "wandb",
            )
            if key in settings
        }
        previous = {key: settings[key] for key in toggles}
        # Bypass SettingsManager's persistence: this run must not alter user preferences.
        dict.update(settings, toggles)
        stack.callback(dict.update, settings, previous)
        for handler in utils.LOGGER.handlers:
            if isinstance(handler, logging.StreamHandler):
                previous_stream = handler.stream
                _set_logging_stream(handler, sys.stderr)
                stack.callback(_set_logging_stream, handler, previous_stream)
        # Imported analytics modules may have cached settings from another caller.
        events = importlib.import_module("ultralytics.utils.events")
        event_emitter = getattr(events, "events", None)
        if event_emitter is not None and hasattr(event_emitter, "enabled"):
            stack.enter_context(patch.object(event_emitter, "enabled", False))
        yield torch, ultralytics


@contextmanager
def _new_output(output: Path) -> Iterator[tuple[Path, Path]]:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield staging, destination
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("ultralytics", "torch", "numpy", "Pillow")
    }


def _training_yaml(dataset: Path, staging: Path) -> tuple[Path, list[str]]:
    """Copy verified inputs so third-party label caches never mutate source data."""
    snapshot = staging / "input_snapshot"
    snapshot.mkdir()
    for name in ("manifest.json", "data.yaml"):
        shutil.copyfile(dataset / name, snapshot / name)
    manifest = strict_json((snapshot / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["classes"]
    for sample in manifest["samples"]:
        for kind in ("image", "label"):
            source = _contained(dataset, sample[kind])
            target = snapshot / sample[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if "sprite_pack" in manifest:
        # Preserve the independently verified source bytes needed to validate
        # and reproduce a custom-sprite dataset after the source is unavailable.
        sprites = verify_dataset_sprite_pack(dataset, manifest["sprite_pack"])
        sprites.write_sources(snapshot)
    if "animation" in manifest:
        from .scene_animation import load_scene_animation

        animation = load_scene_animation(dataset)
        if animation.manifest() != manifest["animation"]:
            raise ValueError("training input animation binding changed")
        animation.write_sources(snapshot)
    # This checks the actual copied bytes, including concurrent source modifications.
    validate_dataset(snapshot)
    for split in ("train", "val", "test"):
        image_paths = [
            str((snapshot / sample["image"]).resolve())
            for sample in manifest["samples"]
            if sample["split"] == split
        ]
        if any("\n" in path or "\r" in path for path in image_paths):
            raise ValueError("dataset image paths cannot contain line breaks")
        (staging / f"{split}.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    data = {
        "path": str(staging),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": dict(enumerate(names)),
    }
    # JSON is a strict YAML subset and avoids custom YAML tags or code hooks.
    path = staging / "dataset.yaml"
    _write_json(path, data)
    return path, list(names)


def _metrics(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float | None] = {}
    for key, item in value.items():
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        result[str(key)] = number if math.isfinite(number) else None
    return result


def train_yolo(
    dataset: Path,
    output: Path,
    model_config: Path,
    epochs: int = 3,
    image_size: int = 320,
    batch: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    """Train a local model from scratch against an integrity-checked fixture dataset."""
    epochs = _integer(epochs, "epochs", 1, 200)
    image_size = _image_size(image_size)
    batch = _integer(batch, "batch", 1, 64)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    if batch * image_size * image_size > 16 * 1024 * 1024:
        raise ValueError("batch times image area exceeds the CPU fixture memory budget")
    config = _local_file(model_config, {".yaml", ".yml"}, "model_config")
    _validate_model_config(config)
    dataset = Path(dataset).resolve()
    validate_dataset(dataset)
    started = time.perf_counter()
    with _new_output(output) as (staging, destination):
        data, names = _training_yaml(dataset, staging)
        shutil.copyfile(config, staging / "model.yaml")
        # Validate the copied file actually passed to Ultralytics. The source
        # may have changed while the dataset snapshot was being assembled.
        _validate_model_config(staging / "model.yaml")
        model_config_hash = _hash(staging / "model.yaml")
        snapshot = staging / "input_snapshot"
        dataset_manifest_hash = _hash(snapshot / "manifest.json")
        with _offline_runtime(staging) as (torch, ultralytics):
            torch.manual_seed(seed)
            model = ultralytics.YOLO(str(staging / "model.yaml"), task="detect", verbose=False)
            result = model.train(
                data=str(data),
                epochs=epochs,
                imgsz=image_size,
                batch=batch,
                device="cpu",
                workers=0,
                pretrained=False,
                amp=False,
                seed=seed,
                deterministic=True,
                project=str(staging),
                name="run",
                exist_ok=False,
                save=True,
                plots=False,
                val=True,
                verbose=False,
                cache=False,
                optimizer="AdamW",
                lr0=0.001,
                warmup_epochs=0,
                patience=epochs + 1,
                compile=False,
                degrees=0,
                translate=0,
                scale=0,
                shear=0,
                perspective=0,
                flipud=0,
                fliplr=0,
                mosaic=0,
                mixup=0,
                copy_paste=0,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                erasing=0,
            )
            metrics = _metrics(getattr(result, "results_dict", {}))
        if (
            _hash(staging / "model.yaml") != model_config_hash
            or _hash(snapshot / "manifest.json") != dataset_manifest_hash
        ):
            raise ValueError("copied training inputs changed during training")
        # Ultralytics cache=False controls images, not label caches. Its caches
        # belong only to our disposable training workspace, never the source dataset.
        for cached in (snapshot / "labels").glob("*.cache"):
            cached.unlink()
        dataset_summary = validate_dataset(snapshot)
        best = staging / "run" / "weights" / "best.pt"
        if not best.is_file():
            raise ValueError("training finished without a best.pt checkpoint")
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "trained",
            "scope": "synthetic_fixture",
            "dataset_manifest_sha256": dataset_manifest_hash,
            "model_config_sha256": model_config_hash,
            "weights_sha256": _hash(best),
            "epochs": epochs,
            "image_size": image_size,
            "batch": batch,
            "seed": seed,
            "device": "cpu",
            "cpu_threads": 2,
            "pretrained": False,
            "class_names": names,
            "metrics_validation_split": metrics,
            "dataset": dataset_summary,
            "dependencies": _versions(),
            "wall_seconds": round(time.perf_counter() - started, 6),
            "weights": "run/weights/best.pt",
            "limitations": [
                "synthetic fixture only",
                "not held-out real-game validation",
                "training completion does not establish deployment accuracy",
            ],
        }
        _write_json(staging / "training.json", report)
    return {
        **report,
        "output": str(destination),
        "weights": str(destination / "run/weights/best.pt"),
    }


def export_yolo(
    weights: Path,
    output: Path,
    image_size: int = 320,
    scope: str = "synthetic_fixture",
) -> dict[str, object]:
    """Export local trusted weights, inspect real ONNX output, and write its manifest."""
    image_size = _image_size(image_size)
    if scope not in {"synthetic_fixture", "authorized_recordings"}:
        raise ValueError("scope must be synthetic_fixture or authorized_recordings")
    weights = _local_file(weights, {".pt"}, "weights")
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    try:
        onnx = importlib.import_module("onnx")
        runtime = importlib.import_module("onnxruntime")
        runtime.disable_telemetry_events()
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise ValueError(
            "install maple-vision-platform[training,inference] for ONNX export"
        ) from exc
    with _new_output(output) as (staging, destination):
        shutil.copyfile(weights, staging / "model.pt")
        with _offline_runtime(staging) as (_, ultralytics):
            model = ultralytics.YOLO(str(staging / "model.pt"), task="detect", verbose=False)
            if model.task != "detect":
                raise ValueError("only detection checkpoints can be exported")
            exported = Path(
                model.export(
                    format="onnx",
                    imgsz=image_size,
                    device="cpu",
                    opset=17,
                    simplify=False,
                    nms=False,
                    dynamic=False,
                    batch=1,
                    verbose=False,
                )
            )
            names_dict = model.names
        model_path = staging / "model.onnx"
        if exported.resolve() != model_path.resolve() or not model_path.is_file():
            raise ValueError("export did not produce the expected local model.onnx")
        graph = onnx.load_model(str(model_path), load_external_data=False)
        onnx.checker.check_model(graph)
        if any(
            tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer
        ):
            raise ValueError("external ONNX tensor data is not supported")
        options = runtime.SessionOptions()
        options.intra_op_num_threads = 1
        session = runtime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, image_size, image_size]:
            raise ValueError("ONNX export must have the requested static NCHW input")
        names = tuple(names_dict[index] for index in range(len(names_dict)))
        outputs = session.run(
            None,
            {
                inputs[0].name: numpy.zeros(
                    (1, 3, image_size, image_size),
                    dtype=numpy.float32,
                )
            },
        )
        if len(outputs) != 1 or outputs[0].ndim != 3 or outputs[0].shape[0] != 1:
            raise ValueError("ONNX export must produce one rank-3 detection tensor")
        if not numpy.isfinite(outputs[0]).all():
            raise ValueError("ONNX smoke inference produced non-finite predictions")
        shape = list(outputs[0].shape)
        if shape[1] == 4 + len(names):
            output_format = "raw_xywh"
        elif shape[2] == 6:
            output_format = "end2end_xyxy"
        else:
            raise ValueError(f"unsupported ONNX detection output shape: {shape}")
        manifest = ModelManifest(
            _hash(model_path), names, image_size, image_size, output_format, scope
        )
        _write_json(staging / "manifest.json", asdict(manifest))
        # Do not duplicate the trusted checkpoint in the runtime deployment artifact.
        (staging / "model.pt").unlink()
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "exported",
            "scope": scope,
            "weights_sha256": _hash(weights),
            "model_sha256": manifest.model_sha256,
            "class_names": list(names),
            "input_shape": inputs[0].shape,
            "output_shape": shape,
            "output_format": output_format,
            "opset": 17,
            "nms": False,
            "simplify": False,
            "dependencies": {
                **_versions(),
                "onnx": onnx.__version__,
                "onnxruntime": runtime.__version__,
            },
            "smoke_inference": "passed",
            "model": "model.onnx",
            "manifest": "manifest.json",
        }
        _write_json(staging / "export.json", report)
    return {
        **report,
        "output": str(destination),
        "model": str(destination / "model.onnx"),
        "manifest": str(destination / "manifest.json"),
    }
