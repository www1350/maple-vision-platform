"""Exercise the actual toolkit on the Windows job and retain its first-use replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from maple_vision.owned_evidence import verify_owned_window_evidence


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows desktop acceptance")
def test_windows_first_use_replay(tmp_path):
    root = Path(__file__).resolve().parents[1]
    parent = root / "artifacts" if os.environ.get("GITHUB_ACTIONS") == "true" else tmp_path
    output = parent / "owned-window-first-use"
    subprocess.run(
        [sys.executable, "scripts/verify_owned_window.py", "--output", str(output)],
        cwd=root,
        check=True,
        timeout=60,
    )
    assert (output / "evidence.json").is_file()
    assert (output / "replay.gif").stat().st_size > 0
    assert verify_owned_window_evidence(output) == {
        "verified": True,
        "scope": "project_owned_native_tk_window",
        "manifest_sha256": json.loads((output / "evidence.json").read_text(encoding="utf-8"))[
            "integrity"
        ]["manifest_sha256"],
        "media_files": 7,
        "replay_frames": 5,
    }


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows desktop acceptance")
def test_player_can_export_their_own_trial(tmp_path, monkeypatch):
    import tkinter as tk

    from PIL import Image

    from maple_vision.owned_route_playback import verify_route_playback
    from maple_vision.owned_window import OwnedWindow
    from scripts.verify_owned_window import wait_for

    output = Path("artifacts/player-replays") if os.environ.get("GITHUB_ACTIONS") else tmp_path
    output.mkdir(parents=True, exist_ok=True)
    path = output / "completed.gif"
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **kwargs: str(path))
    monkeypatch.setattr("tkinter.filedialog.askdirectory", lambda **kwargs: str(output))
    root = tk.Tk()
    try:
        app = OwnedWindow(root)
        root.update()
        root.focus_force()
        app.key_var.set("J")
        app.confirm_button.invoke()
        app.arm_button.invoke()
        assert app.save_button.instate(["disabled"])
        wait_for(root, lambda: app.session.state == "completed", "user replay cast")
        app.save_button.invoke()
        with Image.open(path) as replay:
            assert replay.n_frames == 3
        path = output / "takeover.gif"
        app.confirm_button.invoke()
        app.arm_button.invoke()
        app.takeover_button.invoke()
        app.save_button.invoke()
        with Image.open(path) as replay:
            assert replay.n_frames == 2
        app.confirm_button.invoke()
        app.route_playback_interval_ms = 50
        app.route_button.invoke()
        routes = list(output.glob("maple-layered-route-*"))
        assert len(routes) == 1
        wait_for(root, lambda: (app._route_sequence or 0) >= 72, "route cast playback")
        assert app._route_active
        app.takeover_button.invoke()
        review = json.loads((routes[0] / "player-review.json").read_text())
        assert review["confirmed_key"] == "J"
        assert len(review["navigation"]) == 2
        assert len(review["casts"]) == 4
        assert [item["kind"] for item in review["timeline"]] == [
            "layer_reached",
            "skill_dispatched",
            "takeover",
        ]
        assert review["playback"]["status"] == "taken_over"
        assert (routes[0] / "route-takeover.gif").is_file()
        assert app.session.state == "paused"
        assert app.route_button.instate(["disabled"])
        assert "请重新确认" in app.feedback_label.cget("text")
        assert verify_route_playback(routes[0])["status"] == "taken_over"
    finally:
        root.destroy()
