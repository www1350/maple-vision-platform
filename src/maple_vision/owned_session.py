"""Single-use input session for the project's own interactive practice window."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from .class_package import load_class_package
from .input_bindings import InputBindingProfile, normalize_key, verify_input_profile_confirmation
from .map_pack import load_map_pack
from .sim_types import SimAction
from .simulation import Simulator


class OwnedSession:
    """Route a confirmed local letter to one simulator cast per explicit arm."""

    def __init__(self, examples: Path) -> None:
        self.package = load_class_package(examples / "classes/throwing_star.json")
        self.map_pack = load_map_pack(examples / "maps/dead-forest-drill.json")
        self.simulator = self._fresh_trial()
        self.state = "unconfirmed"
        self.confirmed_key: str | None = None
        self.held_key: str | None = None
        self.generation = 0
        self.events: list[dict[str, Any]] = []
        self._profile: InputBindingProfile | None = None
        self._confirmed_source: tuple[InputBindingProfile, dict[str, Any], str] | None = None

    def load_confirmation(self, receipt: object) -> str:
        """Bind a verified player-confirmed profile to this owned-window trial."""
        if self.state in {"armed", "pressed"}:
            self.takeover("binding_changed")
        profile, normalized = verify_input_profile_confirmation(receipt)
        matches = []
        for item in profile.bindings:
            skill, action = profile.action_for_key(item.key, self.package.class_id, self.package)
            if skill == "lucky_seven" and action is not None:
                matches.append(item)
        if len(matches) != 1:
            raise ValueError("confirmed profile must bind exactly one lucky_seven skill")
        key = normalize_key(matches[0].key)
        if len(key) != 1 or key.upper() not in string.ascii_uppercase:
            raise ValueError("owned-window Lucky Seven key must be one letter A-Z")
        self._confirmed_source = (profile, normalized, key.upper())
        self.state = "unconfirmed"
        self.confirmed_key = None
        self._profile = None
        return key.upper()

    def _fresh_trial(self) -> Simulator:
        simulator = Simulator(
            seed=17,
            job=self.package.class_id,
            map_id=self.map_pack.entry_map,
            map_pack=self.map_pack,
            class_package=self.package,
        )
        # Fixed scene setup, equivalent to walking right for 800 simulated ms.
        for _ in range(16):
            simulator.step(SimAction("move", 1))
        return simulator

    def _record(self, event: str, **facts: Any) -> dict[str, Any]:
        row = {"event": event, "generation": self.generation, "state": self.state, **facts}
        self.events.append(row)
        del self.events[:-128]
        return row

    def confirm(self, key: str) -> None:
        if self.held_key is not None or self.state == "armed":
            self.takeover("binding_changed")
        if not isinstance(key, str) or len(key) != 1 or key not in string.ascii_letters:
            self.takeover("invalid_binding")
            raise ValueError("confirm one visible letter A-Z for Lucky Seven")
        self.generation += 1
        self.confirmed_key = key.upper()
        if self._confirmed_source is None:
            self._profile = InputBindingProfile.from_key_map(
                "owned-window-role", {self.confirmed_key: "lucky_seven"}
            )
            profile_id = self._profile.profile_id
            receipt_sha256 = None
            comparison_source = None
        else:
            profile, receipt, expected_key = self._confirmed_source
            if self.confirmed_key != expected_key:
                self.takeover("binding_changed")
                raise ValueError("visible key does not match the confirmed player profile")
            self._profile = profile
            profile_id = profile.profile_id
            receipt_sha256 = receipt["receipt_sha256"]
            comparison_source = receipt["comparison_source"]
        self.simulator = self._fresh_trial()
        self.state = "confirmed"
        self._record(
            "confirm",
            key=self.confirmed_key,
            skill="lucky_seven",
            profile_id=profile_id,
            confirmation_receipt_sha256=receipt_sha256,
            confirmation_comparison_source=comparison_source,
        )

    def arm(self) -> int:
        if self.state != "confirmed":
            raise ValueError("confirm the role key before arming a single cast")
        self.state = "armed"
        self._record("arm")
        return self.generation

    def press(self, key: str, *, focused: bool, generation: int | None = None) -> dict[str, Any]:
        if generation is not None and generation != self.generation:
            return self._record("press", accepted=False, reason="stale_dispatch")
        if focused is not True:
            self.takeover("focus_lost")
            return self._record("press", accepted=False, reason="focus_lost")
        if self.state != "armed" or self._profile is None:
            return self._record("press", accepted=False, reason="not_armed")
        if len(key) != 1 or key not in string.ascii_letters or key.upper() != self.confirmed_key:
            return self._record("press", accepted=False, reason="unbound_key")
        skill, action = self._profile.action_for_key(key, self.package.class_id, self.package)
        if action is None or skill != "lucky_seven":
            self.takeover("binding_changed")
            return self._record("press", accepted=False, reason="binding_changed")
        self.held_key = self.confirmed_key
        self.state = "pressed"
        before = self.simulator.snapshot()
        outcome = self.simulator.step(action)
        return self._record(
            "press",
            accepted=outcome.accepted,
            reason=outcome.reason,
            skill=skill,
            key=self.held_key,
            map_id=before.map_id,
            source_sequence=before.sequence,
            hit_segments=outcome.events.count("attack_hit"),
            ammo_before=before.player.ammo,
            ammo_after=outcome.snapshot.player.ammo,
            profile_id=self._profile.profile_id,
            confirmation_receipt_sha256=(
                None
                if self._confirmed_source is None
                else self._confirmed_source[1]["receipt_sha256"]
            ),
        )

    def release(self, key: str, *, generation: int | None = None) -> None:
        if generation is not None and generation != self.generation:
            return
        if self.held_key is not None and key.upper() == self.held_key:
            released = self.held_key
            self.held_key = None
            self.state = "completed"
            self._record("release", key=released)

    def takeover(self, reason: str = "manual_takeover") -> None:
        self.generation += 1
        released = self.held_key
        self.held_key = None
        self.confirmed_key = None
        self._profile = None
        self.state = "paused"
        self._record("takeover", reason=reason, released_key=released)
