"""Interactive first-use trial for the project's own Tk window."""

from __future__ import annotations

import argparse
import string
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageTk

from .input_bindings import load_input_profile_confirmation
from .owned_key_image import key_tile, read_key_tile
from .owned_session import OwnedSession
from .scene import SceneRenderer


class OwnedWindow:
    """Generate one key pair inside one focused canvas per armed trial."""

    def __init__(
        self,
        root: tk.Tk,
        examples: Path | None = None,
        input_confirmation: dict[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.session = OwnedSession(examples)
        confirmed_key = (
            "S"
            if input_confirmation is None
            else self.session.load_confirmation(input_confirmation)
        )
        self._timer: str | None = None
        self._emitting: tuple[str, str, int] | None = None
        self._receipt: dict[str, Any] | None = None
        self._replay: list[Image.Image] = []
        self.root.title("Maple Vision · 标飞单次施放")
        frame = ttk.Frame(root, padding=12)
        frame.pack()
        ttk.Label(frame, text="无影人 · 双飞斩", font=("TkDefaultFont", 16)).pack(anchor="w")
        ttk.Label(frame, text="自有练习窗口　｜　死亡之林合成场景").pack(anchor="w")

        settings = ttk.Frame(frame, padding=(0, 8))
        settings.pack(fill="x")
        ttk.Label(settings, text="角色技能键（A–Z）").pack(side="left")
        self.key_var = tk.StringVar(root, value=confirmed_key)
        self.key_entry = ttk.Entry(settings, textvariable=self.key_var, width=5)
        self.key_entry.pack(side="left", padx=8)
        self.display_key = tk.Label(settings, borderwidth=0)
        self.display_key.pack(side="left", padx=4)
        self.key_photo: ImageTk.PhotoImage | None = None
        self.confirm_button = ttk.Button(settings, text="识别并确认键位", command=self.confirm)
        self.confirm_button.pack(side="left", padx=4)
        if input_confirmation is not None:
            ttk.Label(settings, text="已载入玩家确认凭证").pack(side="left", padx=4)

        self.surface = tk.Canvas(frame, width=640, height=384, takefocus=True, highlightthickness=1)
        self.surface.pack()
        self._scene_item = self.surface.create_image(1, 1, anchor="nw")
        actions = ttk.Frame(frame, padding=(0, 8))
        actions.pack(fill="x")
        self.arm_button = ttk.Button(actions, text="施放一次", command=self.arm)
        self.arm_button.pack(side="left")
        self.takeover_button = ttk.Button(actions, text="人工接管（Esc）", command=self.takeover)
        self.takeover_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(actions, text="保存本次回放", command=self.save_replay)
        self.save_button.pack(side="left")
        self.feedback_label = ttk.Label(frame)
        self.feedback_label.pack(anchor="w")
        self.key_var.trace_add("write", self._edited)
        self.surface.bind("<KeyPress>", self._press)
        self.surface.bind("<KeyRelease>", self._release)
        self.surface.bind("<FocusOut>", self._lost_focus)
        self.surface.bind("<ButtonPress>", lambda _: self.takeover())
        self.root.bind("<Escape>", lambda _: self.takeover())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._edited()

    def _render(self) -> None:
        simulator = self.session.simulator
        image = SceneRenderer(map_pack=simulator.map_pack).render(simulator.snapshot()).to_image()
        if self._receipt is not None:
            facts = self._receipt
            draw = ImageDraw.Draw(image)
            draw.rectangle((4, 28, 188, 48), fill=(23, 31, 47))
            draw.text(
                (8, 32),
                f"HITS {facts['hit_segments']}   "
                f"AMMO {facts['ammo_before']} > {facts['ammo_after']}",
                fill=(255, 219, 124),
            )
        self.photo = ImageTk.PhotoImage(
            image.resize((640, 384), Image.Resampling.NEAREST), master=self.root
        )
        self.surface.itemconfigure(self._scene_item, image=self.photo)

    def _capture(self, label: str) -> None:
        scene = ImageTk.getimage(self.photo).convert("RGB")
        frame = Image.new("RGB", (scene.width, scene.height + 32), "#172337")
        frame.paste(scene, (0, 32))
        ImageDraw.Draw(frame).text(
            (8, 7), f"Owned practice | Lucky Seven | {label}", fill="white", font_size=16
        )
        self._replay.append(frame)

    def save_replay(self) -> None:
        if self.session.state not in {"completed", "paused"} or not self._replay:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存本次练习回放",
            defaultextension=".gif",
            filetypes=[("GIF 回放", "*.gif")],
            initialfile="maple-practice.gif",
        )
        if path:
            try:
                self._replay[0].save(
                    path,
                    format="GIF",
                    save_all=True,
                    append_images=self._replay[1:],
                    duration=1200,
                    loop=0,
                )
            except OSError as exc:
                self._feedback(f"回放保存失败：{exc}")
                return
            self._feedback(f"已保存本次练习回放：{path}")

    def _feedback(self, text: str) -> None:
        self.feedback_label.configure(text=text)
        can_save = self.session.state in {"completed", "paused"} and bool(self._replay)
        self.save_button.state(["!disabled" if can_save else "disabled"])
        self.arm_button.state(["!disabled" if self.session.state == "confirmed" else "disabled"])

    def _edited(self, *_: str) -> None:
        self._cancel_timer()
        if self.session.state != "unconfirmed":
            self.session.takeover("binding_changed")
        self._replay.clear()
        key = self.key_var.get().upper()
        self._receipt = None
        if len(key) == 1 and key in string.ascii_uppercase:
            self.key_photo = ImageTk.PhotoImage(key_tile(key), master=self.root)
            self.display_key.configure(image=self.key_photo, text="")
            self.confirm_button.state(["!disabled"])
            self._feedback("请核对显示的技能键，然后确认。")
        else:
            self.display_key.configure(image="", text="—")
            self.key_photo = None
            self.confirm_button.state(["disabled"])
            self._feedback("技能键须为一个英文字母。")
        self._render()

    def read_display_key(self) -> str:
        if (
            not self.display_key.winfo_viewable()
            or self.key_photo is None
            or str(self.display_key.cget("image")) != str(self.key_photo)
        ):
            raise ValueError("技能键图块缺失。")
        key = read_key_tile(ImageTk.getimage(self.key_photo))
        if key is None:
            raise ValueError("无法识别技能键图块。")
        return key

    def confirm(self) -> None:
        self._cancel_timer()
        try:
            self.session.confirm(self.read_display_key())
        except ValueError as exc:
            self.takeover("binding_unreadable")
            self._feedback(str(exc))
            return
        self._receipt = None
        self._render()
        self._replay.clear()
        self._capture(f"Confirmed {self.session.confirmed_key}")
        self._feedback(f"已确认双飞斩键 {self.session.confirmed_key}。可以开始。")

    def arm(self) -> None:
        if self.session.state != "confirmed":
            return
        try:
            key = self.read_display_key()
        except ValueError:
            self.takeover("binding_unreadable")
            return
        if key != self.session.confirmed_key:
            self.takeover("binding_changed")
            return
        token = self.session.arm()
        self.surface.focus_set()
        self._feedback(f"即将在练习窗口施放 {key}。按 Esc 接管。")
        self._timer = self.root.after(400, self._emit, "press", key, token)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self.root.after_cancel(self._timer)
            self._timer = None

    def _focused(self) -> bool:
        return self.surface.winfo_viewable() == 1 and self.root.focus_get() is self.surface

    def _emit(self, kind: str, key: str, token: int) -> None:
        self._timer = None
        if token != self.session.generation:
            return
        if not self._focused():
            self.takeover("focus_lost")
            return
        self._emitting = (kind, key, token)
        event = "<KeyPress>" if kind == "press" else "<KeyRelease>"
        try:
            self.surface.event_generate(event, keysym=key.lower(), when="now")
        finally:
            self._emitting = None
        if kind == "press" and self.session.held_key == key:
            self._timer = self.root.after(220, self._emit, "release", key, token)
        elif self.session.state in {"armed", "pressed"}:
            self.takeover("missing_key_event")

    def _press(self, event: tk.Event[tk.Canvas]) -> str:
        key = event.keysym.upper()
        if self._emitting != ("press", key, self.session.generation):
            self.takeover()
            return "break"
        receipt = self.session.press(
            key, focused=self._focused(), generation=self.session.generation
        )
        if receipt["accepted"]:
            self._receipt = receipt
            self._render()
            self._capture(f"Pressed {key}")
            self._feedback(f"双飞斩 · {key}　命中 {receipt['hit_segments']} 段")
        return "break"

    def _release(self, event: tk.Event[tk.Canvas]) -> str:
        key = event.keysym.upper()
        if self._emitting == ("release", key, self.session.generation):
            self.session.release(key)
            if self._receipt is not None:
                self._capture(f"Released {key}")
                facts = self._receipt
                self._feedback(
                    f"已释放 {key}　｜　命中 {facts['hit_segments']} 段　｜　"
                    f"模拟飞镖 {facts['ammo_before']} → {facts['ammo_after']}"
                )
        return "break"

    def _lost_focus(self, _: tk.Event[tk.Canvas]) -> None:
        if self.session.state in {"armed", "pressed"}:
            self.takeover("focus_lost")

    def takeover(self, reason: str = "manual_takeover") -> None:
        self._cancel_timer()
        interrupted = self.session.state in {"confirmed", "armed", "pressed"}
        self.session.takeover(reason)
        if interrupted and self._replay:
            self._capture(f"Stopped: {reason}")
        messages = {"focus_lost": "练习窗口已失焦", "manual_takeover": "玩家已接管"}
        self._feedback(messages.get(reason, "本次操作已取消") + "，请重新确认后再开始。")

    def close(self) -> None:
        self.takeover("window_closed")
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--input-confirmation", type=Path)
    args = parser.parse_args()
    confirmation = None
    if args.input_confirmation is not None:
        _, confirmation = load_input_profile_confirmation(args.input_confirmation)
    root = tk.Tk()
    OwnedWindow(root, args.examples, confirmation)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
