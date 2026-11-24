import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def invoke(tmp_path: Path, *arguments: str, expected=0):
    result = subprocess.run(
        [sys.executable, "-m", "maple_vision", *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == expected, (result.stdout, result.stderr)
    return json.loads(result.stdout) if result.stdout else result.stderr


def test_catalog_import_query_snapshot_cli(tmp_path):
    database = str(tmp_path / "catalog.db")
    fixture = Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json"
    assert invoke(tmp_path, "knowledge", "init", database)["schema_version"] == 2
    imported = invoke(tmp_path, "knowledge", "import", database, str(fixture))
    assert imported["imported"] == 10
    records = invoke(tmp_path, "knowledge", "query", database, "classes", "assassin", "fixture-v1")[
        "records"
    ]
    assert len(records) == 1 and records[0]["scope"] == "synthetic_fixture"
    snapshot = str(tmp_path / "snapshot")
    invoke(tmp_path, "knowledge", "snapshot", database, snapshot)
    assert invoke(tmp_path, "knowledge", "verify-snapshot", snapshot)["verified"] is True


def test_asset_import_verify_cli(tmp_path):
    source = tmp_path / "sprite.png"
    Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(source)
    registry = str(tmp_path / "registry")
    item = invoke(
        tmp_path,
        "assets",
        "import",
        registry,
        str(source),
        "--entity-id",
        "sample",
        "--game-version",
        "fixture-v1",
        "--source-uri",
        "urn:test:drawn",
        "--license",
        "self-authored",
        "--permission",
        "self_authored",
    )
    assert (
        invoke(tmp_path, "assets", "verify", registry, item["asset_id"])["sha256"] == item["sha256"]
    )
    assert len(invoke(tmp_path, "assets", "list", registry)["assets"]) == 1


def test_dataset_benchmark_and_observe_cli(tmp_path):
    dataset = tmp_path / "dataset"
    summary = invoke(
        tmp_path, "dataset", "create", str(dataset), "--sessions", "6", "--frames", "2"
    )
    assert summary["frames"] == 12
    assert invoke(tmp_path, "dataset", "check", str(dataset))["validated"]
    image = next((dataset / "images" / "test").glob("*.png"))
    observation = invoke(tmp_path, "vision", "observe", str(image))
    assert observation["source_id"] == "synthetic-palette-v2"
    report = invoke(
        tmp_path, "vision", "benchmark", str(dataset), "--output", str(tmp_path / "benchmark.json")
    )
    assert report["split"] == "test"


def test_lab_budget_exit_and_replay_cli(tmp_path):
    output = tmp_path / "run"
    result = invoke(tmp_path, "lab", "run", "--output", str(output), "--max-steps", "2", expected=1)
    assert result["status"] == "failed"
    assert invoke(tmp_path, "lab", "replay", str(output))["verified"]
    invoke(
        tmp_path,
        "lab",
        "run",
        "--output",
        str(tmp_path / "bad"),
        "--model",
        "missing.onnx",
        expected=2,
    )
    assert not (tmp_path / "bad").exists()


def test_chat_draft_cli_does_not_send(tmp_path):
    result = invoke(tmp_path, "chat-draft", "--incoming", "hello", "--reply", "hello back")
    assert result["sent"] is False


def test_packaged_training_config_exists():
    import maple_vision

    assert (Path(maple_vision.__file__).parent / "resources" / "tiny-yolo.yaml").is_file()
