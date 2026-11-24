"""The public map input is explicit, validated before output, and usable for observation."""

import json
from pathlib import Path

from maple_vision.cli import main
from maple_vision.map_pack import load_map_pack
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/maps/three-map-fixture.json"


def test_map_check_reports_structure_not_task_acceptance(capsys):
    assert main(["maps", "check", str(EXAMPLE)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"]
    assert report["map_ids"] == ["grove", "quarry", "ruins"]
    assert report["validation_scope"] == (
        "bounded_structure_and_identity; task_solvability_not_proven"
    )
    assert report["map_pack"]["content_sha256"] == load_map_pack(EXAMPLE).content_sha256


def test_cli_refuses_legacy_profile_with_external_map_before_creating_output(tmp_path, capsys):
    output = tmp_path / "data"
    assert main(["dataset", "create", str(output), "--map-pack", str(EXAMPLE)]) == 2
    assert "lifecycle" in capsys.readouterr().err
    assert not output.exists()


def test_cli_invalid_pack_does_not_create_a_run_directory(tmp_path, capsys):
    source = tmp_path / "invalid.json"
    content = load_map_pack(EXAMPLE).to_dict()
    content["maps"][0]["portals"][0]["destination"] = "missing"
    source.write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "run"
    assert main(["lab", "run", "--output", str(output), "--map-pack", str(source)]) == 2
    assert "portal" in capsys.readouterr().err
    assert not output.exists()


def test_external_image_observation_requires_matching_static_map_markers(tmp_path, capsys):
    pack = load_map_pack(EXAMPLE)
    path = tmp_path / "frame.png"
    frame = SceneRenderer(map_pack=pack).render(
        Simulator(map_id=pack.entry_map, map_pack=pack).snapshot()
    )
    path.write_bytes(frame.png_bytes())
    assert main(["vision", "observe", str(path)]) == 0
    missing = json.loads(capsys.readouterr().out)
    assert missing["map_id"] is None and missing["status"] == "unknown"
    assert main(["vision", "observe", str(path), "--map-pack", str(EXAMPLE)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["map_id"] == "grove" and parsed["status"] == "known"
