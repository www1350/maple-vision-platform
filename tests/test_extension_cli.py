import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from maple_vision.asset_registry import import_asset
from maple_vision.catalog import import_catalog, initialize_catalog


def invoke(directory, *arguments, expected=0):
    result = subprocess.run(
        [sys.executable, "-m", "maple_vision", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == expected, (result.stdout, result.stderr)
    return json.loads(result.stdout) if result.stdout else result.stderr


def test_catalog_rule_changes_drive_real_mission_and_replay_without_database(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures" / "knowledge-demo.json"
    document = json.loads(fixture.read_text())
    document["records"][0]["payload"]["attack_damage"] = 36
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(document))
    database = tmp_path / "catalog.sqlite"
    initialize_catalog(database)
    import_catalog(database, source)
    output = tmp_path / "mission"
    result = invoke(
        tmp_path,
        "lab",
        "run",
        "--output",
        str(output),
        "--knowledge-db",
        str(database),
        "--game-version",
        "fixture-v1",
    )
    assert result["status"] == "completed"
    assert result["events"]["attack_hit"] == 6  # one hit per enemy under the loaded rule
    assert result["class_profile"]["class_spec"]["attack_damage"] == 36
    assert json.loads((output / "initial-state.json").read_text())["schema_version"] == 2
    database.unlink()
    assert invoke(tmp_path, "lab", "replay", str(output))["verified"]


def test_registered_sprite_is_actually_used_in_cli_dataset(tmp_path):
    source = tmp_path / "sprite.png"
    Image.new("RGBA", (8, 8), (240, 210, 20, 255)).save(source)
    registry = tmp_path / "registry"
    item = import_asset(
        registry,
        source,
        entity_id="player",
        game_version="fixture-v1",
        source_uri="urn:test:self-authored-player",
        license="CC0-1.0",
        permission="self_authored",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic_fixture",
                "classes": {"player": item.asset_id},
            }
        )
    )
    output = tmp_path / "dataset"
    result = invoke(
        tmp_path,
        "dataset",
        "create",
        str(output),
        "--sessions",
        "6",
        "--frames",
        "1",
        "--asset-registry",
        str(registry),
        "--sprite-map",
        str(mapping),
    )
    assert result["validated"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["visual_origin"] == "synthetic_geometry_with_authorized_assets"
    image = Image.open(output / manifest["samples"][0]["image"]).convert("RGB")
    assert np.any(np.all(np.asarray(image) == (240, 210, 20), axis=2))


def test_local_recording_import_and_perception_cli(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 48), (30, 50 + i, 70)).save(source / f"{i:04d}.png")
    recording = tmp_path / "recording"
    invoke(
        tmp_path,
        "recording",
        "import",
        str(source),
        str(recording),
        "--source-uri",
        "urn:test:frames",
        "--license",
        "CC0-1.0",
        "--permission",
        "self_authored",
        "--fps",
        "20",
    )
    invoke(tmp_path, "recording", "check", str(recording))
    output = tmp_path / "observations"
    invoke(tmp_path, "recording", "observe", str(recording), "--output", str(output))
    assert (output / "observations.jsonl").is_file()
    lines = (output / "observations.jsonl").read_text().splitlines()
    assert len(lines) == 2
