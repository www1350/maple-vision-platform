"""Exercise the actual toolkit on the Windows job and retain its first-use replay."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from maple_vision.owned_evidence import verify_owned_window_evidence


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows desktop acceptance")
def test_native_external_window_capture_focus_and_resume(tmp_path):
    import tkinter as tk

    from maple_vision.desktop_window import DesktopWindowSession, NativeWindowsBackend

    title = f"Maple Vision Capture Target {os.getpid()}"
    root = tk.Tk()
    handoff = None
    released: list[str] = []
    try:
        root.title(title)
        root.geometry("360x240+80+80")
        canvas = tk.Canvas(root, width=320, height=192, bg="#123456", highlightthickness=0)
        canvas.pack()
        canvas.create_rectangle(40, 50, 280, 140, fill="#e7b44b", outline="#ffffff")
        canvas.create_text(160, 96, text="PROJECT OWNED TARGET", fill="#102030")
        root.update_idletasks()
        root.deiconify()
        root.lift()
        root.focus_force()
        root.update()
        time.sleep(0.1)

        backend = NativeWindowsBackend()
        identity = next(item for item in backend.list_windows() if item.title == title)
        session = DesktopWindowSession(backend, release_key=released.append)
        session.select(identity.handle)
        receipt = session.confirm()
        frame = session.observe()
        assert frame.image.size == (
            identity.client_bounds[2] - identity.client_bounds[0],
            identity.client_bounds[3] - identity.client_bounds[1],
        )
        session.record_key_down("UP", frame)

        handoff = tk.Toplevel(root)
        handoff.title("Maple Vision Human Handoff")
        handoff.geometry("240x100+500+80")
        handoff.update_idletasks()
        handoff.lift()
        handoff.focus_force()
        handoff.update()
        time.sleep(0.1)
        with pytest.raises(ValueError, match="lost focus"):
            session.observe()
        assert session.state == "paused"
        assert released == ["UP"]

        root.lift()
        root.focus_force()
        root.update()
        time.sleep(0.1)
        resumed = session.resume()
        resumed_frame = session.observe()
        assert resumed_frame.sequence == 1

        output = (
            Path("artifacts/windows-capture-session")
            if os.environ.get("GITHUB_ACTIONS")
            else tmp_path / "windows-capture-session"
        )
        output.mkdir(parents=True, exist_ok=True)
        frame.image.save(output / "before-focus-loss.png")
        resumed_frame.image.save(output / "after-explicit-resume.png")
        (output / "evidence.json").write_text(
            json.dumps(
                {
                    "scope": "project_owned_external_window_capture",
                    "receipt": receipt,
                    "paused_reason": "focus_lost",
                    "released_keys": released,
                    "resume": resumed,
                    "frames": [frame.manifest(), resumed_frame.manifest()],
                    "limitations": [
                        "screen pixels from a visible project-owned Tk target",
                        "no official client and no external input dispatch",
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # This native integration report has two frames and a focus timeline; the
        # one-frame command bundle has its stricter media verifier in separate tests.
        assert frame.rgb_sha256 != ""
    finally:
        if handoff is not None:
            handoff.destroy()
        root.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows desktop acceptance")
def test_native_external_window_pixels_feed_player_visible_observation(tmp_path):
    import tkinter as tk

    from PIL import Image, ImageDraw, ImageTk

    from maple_vision.desktop_observation import (
        fixture_desktop_observation_session,
        onnx_desktop_observation_session,
        verify_desktop_observation_evidence,
        write_desktop_observation_evidence,
    )
    from maple_vision.desktop_window import DesktopWindowSession, NativeWindowsBackend
    from maple_vision.onnx_detector import OnnxDetector
    from maple_vision.scene import (
        BAR_BG,
        BAR_WIDTH,
        BAR_X,
        HP_COLOR,
        HP_Y,
        HUD_BG,
        MAP_MARKERS,
        MP_COLOR,
        MP_Y,
        PALETTE,
    )

    image = Image.new("RGB", (320, 192), "#1e394c")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 24), fill=HUD_BG)
    draw.rectangle((BAR_X, HP_Y + 2, BAR_X + BAR_WIDTH - 1, HP_Y + 2), fill=BAR_BG)
    draw.rectangle((BAR_X, MP_Y + 2, BAR_X + BAR_WIDTH - 1, MP_Y + 2), fill=BAR_BG)
    draw.rectangle((BAR_X, HP_Y + 2, BAR_X + 79, HP_Y + 2), fill=HP_COLOR)
    draw.rectangle((BAR_X, MP_Y + 2, BAR_X + 69, MP_Y + 2), fill=MP_COLOR)
    draw.rectangle((300, 6, 311, 17), fill=MAP_MARKERS["meadow"])
    draw.rectangle((100, 90, 111, 113), fill=PALETTE["player"])
    draw.rectangle((170, 98, 185, 117), fill=PALETTE["monster"])

    title = f"Maple Vision Observation Target {os.getpid()}"
    root = tk.Tk()
    try:
        root.title(title)
        root.geometry("320x192+80+80")
        canvas = tk.Canvas(root, width=320, height=192, borderwidth=0, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        root.update_idletasks()
        root.deiconify()
        root.lift()
        root.focus_force()
        root.update()
        time.sleep(0.15)

        backend = NativeWindowsBackend()
        identity = next(item for item in backend.list_windows() if item.title == title)
        assert (
            identity.client_bounds[2] - identity.client_bounds[0],
            identity.client_bounds[3] - identity.client_bounds[1],
        ) == (320, 192)
        desktop = DesktopWindowSession(backend, release_key=lambda _: None)
        desktop.select(identity.handle)
        receipt = desktop.confirm()
        result = fixture_desktop_observation_session(desktop).observe()
        assert result.observation.map_id == "meadow"
        assert result.belief.actionable
        fixture_events = list(desktop.events)
        model_root = Path(__file__).resolve().parents[1] / "examples" / "lifecycle-model"
        detector = OnnxDetector(model_root / "model.onnx", model_root / "manifest.json")
        model_result = onnx_desktop_observation_session(desktop, detector).observe()
        assert {item.class_id for item in model_result.observation.detections} >= {
            "player",
            "monster",
        }
        assert not model_result.belief.actionable
        desktop.close()

        output = (
            Path("artifacts/windows-observation-session")
            if os.environ.get("GITHUB_ACTIONS")
            else tmp_path / "windows-observation-session"
        )
        write_desktop_observation_evidence(output, result, receipt=receipt, events=fixture_events)
        verified = verify_desktop_observation_evidence(output)
        assert verified["status"] == "known"
        assert verified["actionable"] is True
        model_output = (
            Path("artifacts/windows-onnx-observation-session")
            if os.environ.get("GITHUB_ACTIONS")
            else tmp_path / "windows-onnx-observation-session"
        )
        write_desktop_observation_evidence(
            model_output, model_result, receipt=receipt, events=desktop.events
        )
        model_verified = verify_desktop_observation_evidence(model_output)
        assert model_verified["status"] == "unknown"
        assert model_verified["actionable"] is False
    finally:
        root.destroy()


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
        app.route_playback_interval_ms = 20
        app.route_button.invoke()
        routes = list(output.glob("maple-layered-route-*"))
        assert len(routes) == 1
        wait_for(root, lambda: (app._route_sequence or 0) >= 72, "route cast playback")
        assert app._route_active
        app.takeover_button.invoke()
        review = json.loads((routes[0] / "player-review.json").read_text())
        assert review["confirmed_key"] == "J"
        assert len(review["navigation"]) == 1
        assert len(review["casts"]) == 1
        assert [item["kind"] for item in review["timeline"]] == [
            "layer_reached",
            "skill_dispatched",
            "takeover",
        ]
        assert review["playback"]["status"] == "taken_over"
        assert review["execution"]["status"] == "taken_over"
        assert review["execution"]["executed_steps"] == 73
        assert review["execution"]["decision_cycles"] == 73
        assert review["execution"]["remaining_steps"] == "not_precomputed"
        assert (routes[0] / "trace.jsonl").is_file()
        assert (routes[0] / "summary.json").is_file()
        assert (routes[0] / "route-takeover.gif").is_file()
        assert app.session.state == "paused"
        assert app.route_button.instate(["disabled"])
        assert "请重新确认" in app.feedback_label.cget("text")
        verified = verify_route_playback(routes[0])
        assert verified["status"] == "taken_over"
        assert verified["execution"]["remaining_steps"] == "not_precomputed"
    finally:
        root.destroy()
