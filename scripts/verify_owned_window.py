"""Exercise the mapped practice window and export its displayed Tk scene buffers."""

import argparse
import hashlib
import json
import platform
import tempfile
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from maple_vision.owned_window import OwnedWindow
from maple_vision.scene import ImageFrame
from maple_vision.vision import PixelPerception


def wait_for(root: tk.Tk, condition: Callable[[], bool], label: str) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(0.005)
    raise RuntimeError(f"window did not reach {label}")


def displayed_scene(app: OwnedWindow) -> Image.Image:
    items = [i for i in app.surface.find_all() if app.surface.type(i) == "image"]
    if (
        not app.surface.winfo_viewable()
        or len(items) != 1
        or app.surface.itemcget(items[0], "image") != str(app.photo)
        or app.surface.itemcget(items[0], "state") == "hidden"
    ):
        raise RuntimeError("scene image is not displayed on the mapped canvas")
    return ImageTk.getimage(app.photo).convert("RGB")


def visible_mp(image: Image.Image) -> float:
    logical = image.resize((320, 192), Image.Resampling.NEAREST)
    hud = PixelPerception().observe_hud(ImageFrame(0, 0, 320, 192, logical.tobytes()))
    if hud.mp_ratio is None:
        raise RuntimeError("could not read the visible MP bar")
    return hud.mp_ratio


def verify(root: tk.Tk, examples: Path, output: Path) -> dict:
    app = OwnedWindow(root, examples)
    root.update()
    root.focus_force()
    frames: list[Image.Image] = []
    captures: list[dict] = []
    key = "J"

    def capture(label: str) -> Image.Image:
        scene = displayed_scene(app)
        filename = f"{len(captures):02d}.png"
        scene.save(output / filename)
        captures.append(
            {
                "label": label,
                "file": filename,
                "rgb_sha256": hashlib.sha256(scene.tobytes()).hexdigest(),
            }
        )
        frame = Image.new("RGB", (scene.width, scene.height + 58), "#172337")
        frame.paste(scene, (0, 58))
        draw = ImageDraw.Draw(frame)
        draw.text((12, 7), label, fill="white", font_size=17)
        draw.text(
            (12, 34), "Owned practice window | Lucky Seven | Tk scene readback", fill="#becada"
        )
        frames.append(frame)
        return scene

    def accepted_casts() -> list[dict]:
        return [e for e in app.session.events if e["event"] == "press" and e["accepted"]]

    def confirm() -> None:
        app.confirm_button.invoke()
        wait_for(root, lambda: app.session.state == "confirmed", "confirmed binding")
        if app.session.confirmed_key != key:
            raise RuntimeError("window confirmed a different role binding")

    app.key_entry.delete(0, "end")
    app.key_entry.insert(0, key)
    root.update()
    if app.read_display_key() != key:
        raise RuntimeError("visible rebound key was not recognized")
    ImageTk.getimage(app.key_photo).save(output / "confirmed-key.png")
    confirm()
    before = capture("1. Role binding confirmed: J")
    app.arm_button.invoke()
    wait_for(root, lambda: app.session.state == "completed", "cast and key release")
    after = capture("2. Lucky Seven dispatched: two hits, two stars consumed")
    casts = accepted_casts()
    if len(casts) != 1 or app.session.held_key is not None:
        raise RuntimeError("single dispatch or key release failed")
    cast = casts[0]
    if (
        cast["key"],
        cast["skill"],
        cast["hit_segments"],
        cast["ammo_before"],
        cast["ammo_after"],
    ) != (key, "lucky_seven", 2, 200, 198):
        raise RuntimeError("unexpected simulator cast receipt")
    mp_before, mp_after = visible_mp(before), visible_mp(after)
    if mp_after >= mp_before:
        raise RuntimeError("visible scene did not show MP consumption")

    # Transfer focus to a second real toplevel before the queued press.
    confirm()
    peer = tk.Toplevel(root)
    peer.title("Practice focus target")
    peer.geometry("170x60+5+5")
    root.update()
    root.focus_force()
    app.arm_button.invoke()
    wait_for(root, lambda: root.focus_get() == app.surface, "focused armed canvas")
    peer.focus_force()
    wait_for(root, lambda: app.session.state == "paused", "focus-loss cancellation")
    settled = tk.BooleanVar(root, False)
    root.after(900, lambda: settled.set(True))
    root.wait_variable(settled)
    if len(accepted_casts()) != 1 or app.session.held_key is not None:
        raise RuntimeError("pending cast survived focus transfer")
    root.focus_force()
    capture("3. Focus left the arena: pending cast cancelled")

    # A takeover during a pressed key must release that key and fence later callbacks.
    for case, interrupt, caption in [
        ("focus_lost", peer.focus_force, "4. Focus lost while holding J: released and paused"),
        ("manual_takeover", app.takeover_button.invoke, "5. Manual takeover: released and paused"),
    ]:
        root.focus_force()
        confirm()
        count = len(accepted_casts())
        app.arm_button.invoke()
        wait_for(root, lambda: app.session.held_key == key, "held key before interruption")
        interrupt()
        wait_for(root, lambda: app.session.state == "paused", case)
        settled.set(False)
        root.after(900, lambda: settled.set(True))
        root.wait_variable(settled)
        last = app.session.events[-1]
        if (
            app.session.held_key is not None
            or app.session.state != "paused"
            or len(accepted_casts()) != count + 1
            or (last["event"], last.get("reason"), last.get("released_key"))
            != ("takeover", case, key)
        ):
            raise RuntimeError(f"{case} failed to release and cancel")
        capture(caption)
    peer.destroy()
    frames[0].save(
        output / "replay.gif", save_all=True, append_images=frames[1:], duration=1500, loop=0
    )
    with Image.open(output / "replay.gif") as replay:
        if replay.n_frames != len(captures):
            raise RuntimeError("incomplete replay")
    return {
        "schema_version": 1,
        "verified": True,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "tk_patchlevel": str(root.tk.call("info", "patchlevel")),
        "scope": "project_owned_native_tk_window",
        "input": "widget_scoped_tk_key_events",
        "capture": "mapped_canvas_photoimage_rgb",
        "visible_key": key,
        "normal_cast_receipt": cast,
        "visible_mp_before": mp_before,
        "visible_mp_after": mp_after,
        "focus_loss_cancelled_pending_press": True,
        "focus_loss_released_held_key": True,
        "takeover_released_held_key": True,
        "events": app.session.events,
        "captures": captures,
        "replay": "replay.gif",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path(__file__).resolve().parents[1] / "examples"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        parser.error("use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "verified"
        staged.mkdir()
        root = tk.Tk()
        try:
            report = verify(root, args.examples, staged)
        finally:
            root.destroy()
        (staged / "evidence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    print(f"Verified {report['scope']} on {report['platform']}: {output}")


if __name__ == "__main__":
    main()
