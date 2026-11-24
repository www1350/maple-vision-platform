import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "maple_vision", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_emits_full_trace_and_json_summary(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    process = run_cli(tmp_path, "demo", "--frames", "3", "--output", str(output))
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["frames_processed"] == summary["verifications_not_applicable"] == 3
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert len({row["run_id"] for row in rows}) == 1
    for index, row in enumerate(rows):
        assert row["schema_version"] == 1
        tick = row["tick"]
        assert tick["observation"]["sequence"] == index
        assert tick["world"]["status"] == "unknown"
        assert tick["intent"]["source_sequence"] == index
        assert tick["execution"]["status"] == "skipped"
        assert tick["verification"]["status"] == "not_applicable"
        assert "pixels" not in json.dumps(row)


def test_demo_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("original", encoding="utf-8")
    process = run_cli(tmp_path, "demo", "--output", str(output))
    assert process.returncode == 2
    assert output.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("count", ["0", "-1", "10001", "2.5"])
def test_invalid_frame_count_fails_without_creating_trace(tmp_path: Path, count: str) -> None:
    process = run_cli(tmp_path, "demo", "--frames", count)
    assert process.returncode == 2
    assert not (tmp_path / "p0-trace.jsonl").exists()


def test_config_and_cli_override(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'schema_version = 1\n[runtime]\nmode = "demo"\nframes = 2\nframe_interval_ms = 80\n',
        encoding="utf-8",
    )
    process = run_cli(tmp_path, "demo", "--config", str(path), "--frames", "4")
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["frames_processed"] == 4
    rows = (tmp_path / "p0-trace.jsonl").read_text().splitlines()
    assert json.loads(rows[1])["tick"]["observation"]["timestamp_ms"] == 80


@pytest.mark.parametrize(
    "config",
    [
        "schema_version = 2\n",
        "schema_version = true\n",
        'schema_version = 1\n[runtime]\nmode = "live"\n',
        "schema_version = 1\n[runtime]\nframes = true\n",
        "schema_version = 1\n[runtime]\nframez = 2\n",
        "schema_version = 1\n[runtim]\nframes = 2\n",
        "schema_version = 1\n[runtime\n",
    ],
)
def test_invalid_config_fails_before_output_creation(tmp_path: Path, config: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(config, encoding="utf-8")
    process = run_cli(tmp_path, "demo", "--config", str(path))
    assert process.returncode == 2
    assert not (tmp_path / "p0-trace.jsonl").exists()


def test_cli_initializes_and_checks_readonly_database(tmp_path: Path) -> None:
    path = tmp_path / "知识 #库.db"
    for operation in ["init", "init", "check"]:
        process = run_cli(tmp_path, "db", operation, str(path))
        assert process.returncode == 0, process.stderr
        assert json.loads(process.stdout)["schema_version"] == 1
        assert json.loads(process.stdout)["source_records"] == 0


def test_db_check_missing_file_does_not_create_one(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    process = run_cli(tmp_path, "db", "check", str(path))
    assert process.returncode == 2
    assert not path.exists()
