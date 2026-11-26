"""Independent ordered objective witnesses, used by lab execution and action replay."""

from typing import Any

from .map_pack import MapPack
from .mission_program import MissionProgram
from .sim_types import SimSnapshot


def compare_branch_decisions(runtime_decisions: object, oracle_decisions: object) -> dict[str, Any]:
    """Compare visual-policy branch choices with simulator-truth choices.

    The full runtime and oracle records remain separate evidence.  This compact
    derived check compares only the semantic edge identity, never treating a
    policy-reported observation as simulator truth.
    """

    def path(value: object, label: str) -> list[dict[str, str]]:
        if not isinstance(value, list) or len(value) > 8:
            raise ValueError(f"invalid {label} mission branch evidence")
        result: list[dict[str, str]] = []
        for row in value:
            if not isinstance(row, dict):
                raise ValueError(f"invalid {label} mission branch evidence")
            source_map = row.get("source_map")
            condition = row.get("condition")
            target_map = row.get("target_map")
            if any(
                not isinstance(item, str) or not item
                for item in (source_map, condition, target_map)
            ):
                raise ValueError(f"invalid {label} mission branch evidence")
            assert isinstance(source_map, str)
            assert isinstance(condition, str)
            assert isinstance(target_map, str)
            result.append(
                {
                    "source_map": source_map,
                    "condition": condition,
                    "target_map": target_map,
                }
            )
        return result

    runtime_path = path(runtime_decisions, "runtime")
    oracle_path = path(oracle_decisions, "oracle")
    matched = runtime_path == oracle_path
    mismatch_index = None
    if not matched:
        for index in range(max(len(runtime_path), len(oracle_path))):
            if (
                index >= len(runtime_path)
                or index >= len(oracle_path)
                or runtime_path[index] != oracle_path[index]
            ):
                mismatch_index = index
                break
    return {
        "schema_version": 1,
        "scope": "runtime_visual_branch_vs_simulator_truth",
        "matched": matched,
        "mismatch_index": mismatch_index,
        "runtime_path": runtime_path,
        "oracle_path": oracle_path,
    }


class MissionOracle:
    def __init__(self, program: MissionProgram, map_pack: MapPack | None = None) -> None:
        self.program = program
        self.map_pack = map_pack
        if any(stage.navigation is not None for stage in program.stages):
            if type(map_pack) is not MapPack:
                raise ValueError("navigation mission oracle requires its map pack")
            for stage in program.stages:
                if stage.navigation is None:
                    continue
                platforms = map_pack.get(stage.map_id).platforms
                try:
                    index = int(stage.navigation.target_platform.removeprefix("platform_"))
                except ValueError as exc:
                    raise ValueError("navigation target must name a map-pack platform") from exc
                if not 0 <= index < len(platforms):
                    raise ValueError("navigation target platform is outside the map pack")
        self.map_index = 0
        self.phase_index = 0
        self.witnesses: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self.invalid_transition = False
        self.completed = False
        self.pending_map_index: int | None = None

    def observe(self, before: SimSnapshot, after: SimSnapshot, events: tuple[str, ...]) -> None:
        if self.completed:
            return
        stage = self.program.stages[self.map_index]
        if before.map_id != stage.map_id:
            self.invalid_transition = True
            return
        if self.phase_index == len(stage.phases):
            if after.map_id != before.map_id:
                if self.pending_map_index is None:
                    self.invalid_transition = True
                    return
                self.map_index = self.pending_map_index
                self.pending_map_index = None
                self.phase_index = 0
                if after.map_id != self.program.maps[self.map_index]:
                    self.invalid_transition = True
            return
        if after.map_id != stage.map_id:
            self.invalid_transition = True
            return
        phase = stage.phases[self.phase_index]
        if phase == "navigate":
            assert self.map_pack is not None and stage.navigation is not None
            index = int(stage.navigation.target_platform.removeprefix("platform_"))
            platform = self.map_pack.get(stage.map_id).platforms[index]
            player = after.player
            satisfied = (
                player.grounded
                and abs(player.y + player.h - platform.y) <= 1e-6
                and player.x + player.w > platform.x1
                and player.x < platform.x2
            )
        else:
            satisfied = {
                "fight": not after.monsters,
                "collect": not after.monsters and not after.drops,
                "talk": "dialogue" in events and after.dialogue is not None,
            }[phase]
        if satisfied:
            witness = {"map_id": stage.map_id, "phase": phase, "sequence": after.sequence}
            if phase == "navigate":
                assert stage.navigation is not None
                witness["target_platform"] = stage.navigation.target_platform
            self.witnesses.append(witness)
            self.phase_index += 1
            if self.phase_index == len(stage.phases):
                if self.program.conditional:
                    transition = self.program.transition(
                        stage.map_id, loot_visible=bool(after.drops)
                    )
                    if transition is None:
                        self.completed = True
                    else:
                        self.pending_map_index = self.program.maps.index(transition.target_map)
                        self.branches.append(
                            {
                                "source_map": stage.map_id,
                                "target_map": transition.target_map,
                                "condition": transition.condition,
                                "sequence": after.sequence,
                            }
                        )
                elif self.map_index + 1 == len(self.program.stages):
                    self.completed = True
                else:
                    self.pending_map_index = self.map_index + 1

    def result(self, final: SimSnapshot) -> dict[str, Any]:
        report = {
            "program_sha256": self.program.content_sha256,
            "completed": (
                self.completed
                and final.map_id == self.program.maps[self.map_index]
                and final.player.alive
                and not self.invalid_transition
            ),
            "invalid_transition": self.invalid_transition,
            "witnesses": list(self.witnesses),
        }
        if self.program.conditional:
            report["branches"] = list(self.branches)
        return report
