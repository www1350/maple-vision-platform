import json
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.owned_evidence import seal_owned_window_evidence, verify_owned_window_evidence


def make_bundle(root: Path) -> dict:
    root.mkdir()
    captures = []
    replay_frames = []
    for index in range(5):
        image = Image.new("RGB", (20, 12), (index * 30, 40, 80))
        name = f"{index:02d}.png"
        image.save(root / name)
        captures.append({"label": f"step {index}", "file": name})
        replay_frames.append(Image.new("RGB", (20, 18), (index * 30, 40, 80)))
    Image.new("RGB", (10, 10), "white").save(root / "confirmed-key.png")
    replay_frames[0].save(
        root / "replay.gif",
        save_all=True,
        append_images=replay_frames[1:],
        duration=120,
        loop=0,
    )
    report = seal_owned_window_evidence(
        {
            "schema_version": 1,
            "verified": True,
            "scope": "project_owned_native_tk_window",
            "captures": captures,
            "replay": "replay.gif",
        },
        root,
    )
    (root / "evidence.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_owned_window_bundle_reopens_every_declared_media(tmp_path):
    report = make_bundle(tmp_path / "bundle")
    assert verify_owned_window_evidence(tmp_path / "bundle") == {
        "verified": True,
        "scope": "project_owned_native_tk_window",
        "manifest_sha256": report["integrity"]["manifest_sha256"],
        "media_files": 7,
        "replay_frames": 5,
    }


@pytest.mark.parametrize("target", ["00.png", "confirmed-key.png", "replay.gif"])
def test_owned_window_bundle_rejects_substituted_media(tmp_path, target):
    root = tmp_path / "bundle"
    make_bundle(root)
    (root / target).write_bytes(b"substituted")
    with pytest.raises((ValueError, OSError)):
        verify_owned_window_evidence(root)


def test_owned_window_bundle_rejects_manifest_edits_and_undeclared_files(tmp_path):
    root = tmp_path / "bundle"
    report = make_bundle(root)
    report["captures"][0]["label"] = "rewritten"
    (root / "evidence.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest was modified"):
        verify_owned_window_evidence(root)

    make_bundle(tmp_path / "extra")
    (tmp_path / "extra" / "notes.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        verify_owned_window_evidence(tmp_path / "extra")


def test_owned_window_bundle_rejects_path_escape_and_symlink(tmp_path):
    root = tmp_path / "bundle"
    report = make_bundle(root)
    report["captures"][0]["file"] = "../00.png"
    report.pop("integrity")
    with pytest.raises(ValueError, match="invalid owned-window media name"):
        seal_owned_window_evidence(report, root)

    linked = tmp_path / "linked"
    make_bundle(linked)
    (linked / "00.png").unlink()
    try:
        (linked / "00.png").symlink_to(linked / "01.png")
    except OSError:
        pytest.skip("this runner does not permit symlink creation")
    with pytest.raises(ValueError, match="regular file"):
        verify_owned_window_evidence(linked)
