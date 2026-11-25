"""Independent ordered objective witnesses, used by lab execution and action replay."""

from typing import Any

from .mission_program import MissionProgram
from .sim_types import SimSnapshot


class MissionOracle:
    def __init__(self, program: MissionProgram) -> None:
        self.program = program
        self.map_index = 0
        self.phase_index = 0
        self.witnesses: list[dict[str, Any]] = []
        self.invalid_transition = False

    def observe(self, before: SimSnapshot, after: SimSnapshot, events: tuple[str, ...]) -> None:
        if self.map_index == len(self.program.stages):
            return
        stage = self.program.stages[self.map_index]
        if before.map_id != stage.map_id:
            self.invalid_transition = True
            return
        if self.phase_index == len(stage.phases):
            if after.map_id != before.map_id:
                self.map_index += 1
                self.phase_index = 0
                if (
                    self.map_index >= len(self.program.stages)
                    or after.map_id != self.program.maps[self.map_index]
                ):
                    self.invalid_transition = True
            return
        if after.map_id != stage.map_id:
            self.invalid_transition = True
            return
        phase = stage.phases[self.phase_index]
        satisfied = {
            "fight": not after.monsters,
            "collect": not after.monsters and not after.drops,
            "talk": "dialogue" in events and after.dialogue is not None,
        }[phase]
        if satisfied:
            self.witnesses.append(
                {"map_id": stage.map_id, "phase": phase, "sequence": after.sequence}
            )
            self.phase_index += 1
            if self.phase_index == len(stage.phases) and self.map_index + 1 == len(
                self.program.stages
            ):
                self.map_index += 1

    def result(self, final: SimSnapshot) -> dict[str, Any]:
        return {
            "program_sha256": self.program.content_sha256,
            "completed": (
                self.map_index == len(self.program.stages)
                and final.map_id == self.program.maps[-1]
                and final.player.alive
                and not self.invalid_transition
            ),
            "invalid_transition": self.invalid_transition,
            "witnesses": list(self.witnesses),
        }
