"""Strict authored sources, meaningful compilation and bounded linear DSL."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from maple_vision.class_package import (
    MAX_CLASS_PACKAGE_BYTES,
    ClassPackage,
    compile_class_package,
    load_class_package,
)
from maple_vision.sim_types import CLASS_SPECS, ClassSpec, SimAction

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/classes"


def source(name="archer"):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def skill(value, key):
    return next(item for item in value["skills"] if item["id"] == key)


@pytest.mark.parametrize("name", ["archer", "bandit", "blood_knight"])
def test_packages_roundtrip_full_sources_hash_and_static_compilation(name):
    package = load_class_package(EXAMPLES / f"{name}.json")
    restored = ClassPackage.from_manifest(json.loads(json.dumps(package.manifest())))
    assert restored == package
    compiled = compile_class_package(package)
    assert compiled.to_dict() == compile_class_package(restored).to_dict()
    assert compiled.class_id == name and compiled.package_sha256 == package.content_sha256
    assert compiled.class_spec.max_hp == package.resources.max_hp
    assert compiled.program_for("combat") is not None
    json.dumps(compiled.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        compiled.class_spec.max_hp = 1
    with pytest.raises(TypeError):
        compiled.skills["new"] = next(iter(compiled.skills.values()))
    document = package.to_dict()
    document["resources"]["max_hp"] = 1
    assert package.resources.max_hp > 1


def test_selected_levels_compile_effects_and_preserve_unselected_source():
    original = source()
    selected = compile_class_package(ClassPackage.from_dict(original))
    assert selected.skills["arrow_shot"].level == 2
    assert selected.skills["arrow_shot"].power == 22
    changed = deepcopy(original)
    skill(changed, "arrow_shot")["selected_level"] = 1
    lower = compile_class_package(ClassPackage.from_dict(changed))
    assert lower.skills["arrow_shot"].power == 16
    assert lower.class_spec.attack_range == 108
    assert lower.package_sha256 != selected.package_sha256
    assert len(skill(original, "arrow_shot")["levels"]) == 2


def test_compound_program_resolves_distinct_skills_wait_offsets_and_cooldown_groups():
    caps = compile_class_package(load_class_package(EXAMPLES / "bandit.json"))
    program = caps.program_for("combat")
    steps = program.instantiate(direction=-1)
    assert [(step.action, step.not_before_ms) for step in steps] == [
        (SimAction("cast", -1, "slash"), 0),
        (SimAction("cast", -1, "finisher"), 150),
    ]
    assert program.skill_ids == ("slash", "finisher")
    assert program.duration_ms == 150 and program.reuse_after_ms == 450
    assert program.cooldown_ready_offsets == (("weapon", 450),)
    assert (program.hp_cost, program.mp_cost, program.ammo_cost) == (3, 4, 0)
    assert program.range_px == 42
    with pytest.raises(ValueError):
        program.instantiate(direction=0)
    with pytest.raises(ValueError):
        program.instantiate(direction=True)


def test_airborne_program_is_real_jump_then_separate_air_skill():
    caps = compile_class_package(load_class_package(EXAMPLES / "archer.json"))
    program = caps.program_for("airborne")
    steps = program.instantiate(1)
    assert program.posture == "ground"
    assert [(step.action.kind, step.not_before_ms) for step in steps] == [("jump", 0), ("cast", 50)]
    assert steps[1].action.skill == "air_arrow"
    assert caps.skills["air_arrow"].posture == "air"


@pytest.mark.parametrize("value", [True, 1.0, 2, "1", None])
def test_manifest_schema_requires_exact_supported_integer(value):
    raw = load_class_package(EXAMPLES / "archer.json").manifest()
    raw["schema_version"] = value
    with pytest.raises(ValueError):
        ClassPackage.from_manifest(raw)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(scope="game_reference"),
        lambda p: p.update(permission="unverified"),
        lambda p: p.update(extra="hidden"),
        lambda p: p.update(class_id="../archer"),
        lambda p: p["resources"].update(max_hp=True),
        lambda p: p["resources"].update(max_mp=float("nan")),
        lambda p: p["resources"].update(initial_ammo=-1),
        lambda p: p["resources"].update(walk_speed=201),
        lambda p: p["resources"].pop("max_hp"),
        lambda p: skill(p, "arrow_shot").update(selected_level=3),
        lambda p: skill(p, "arrow_shot").update(selected_level=True),
        lambda p: skill(p, "arrow_shot").update(posture="unknown"),
        lambda p: skill(p, "arrow_shot").update(shape="self"),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(unknown=10),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(cooldown_ms=0),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(hp_cost=110),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(mp_cost=71),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(max_targets=9),
        lambda p: skill(p, "arrow_shot")["levels"][0].update(duration_ms=1),
        lambda p: skill(p, "focus")["levels"][0].update(duration_ms=0),
        lambda p: skill(p, "focus")["levels"][0].update(ammo_cost=1),
        lambda p: p["skills"].append(deepcopy(p["skills"][0])),
        lambda p: skill(p, "arrow_shot")["levels"].append(
            deepcopy(skill(p, "arrow_shot")["levels"][0])
        ),
        lambda p: p["bindings"].update(combat="missing"),
        lambda p: p["bindings"].update(buff="aimed_shot"),
        lambda p: p["bindings"].update(area="aimed_shot"),
        lambda p: p["programs"][0]["steps"][0].update(skill="missing"),
        lambda p: p["programs"][0]["steps"][0].update(op="eval"),
        lambda p: p["programs"][0]["steps"][0].update(direction=True),
    ],
)
def test_invalid_or_unsafe_sources_fail_before_execution(change):
    value = source()
    change(value)
    with pytest.raises(ValueError):
        ClassPackage.from_dict(value)


def test_dsl_rejects_implicit_cooldown_violations_and_impossible_total_resources():
    value = source("bandit")
    value["programs"][0]["steps"][1]["duration_ms"] = 149
    with pytest.raises(ValueError, match="cooldown"):
        ClassPackage.from_dict(value)
    value = source("bandit")
    skill(value, "slash")["levels"][1]["hp_cost"] = 118
    with pytest.raises(ValueError, match="program resource"):
        ClassPackage.from_dict(value)
    value = source("bandit")
    skill(value, "finisher")["levels"][0]["mp_cost"] = 70
    with pytest.raises(ValueError, match="program resource"):
        ClassPackage.from_dict(value)


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [{"op": "wait", "duration_ms": 5}],
        [{"op": "cast", "skill": "arrow_shot", "direction": 1}, {"op": "wait", "duration_ms": 5}],
        [{"op": "jump", "direction": 0}] * 9,
        [
            {"op": "wait", "duration_ms": 5001},
            {"op": "cast", "skill": "arrow_shot", "direction": 1},
        ],
        [{"op": "loop", "steps": []}],
        [{"op": "move", "direction": 0}],
    ],
)
def test_dsl_cannot_hide_unbounded_execution_or_no_effect_programs(steps):
    value = source()
    value["programs"][0]["steps"] = steps
    with pytest.raises(ValueError):
        ClassPackage.from_dict(value)


def test_source_and_instruction_order_have_explicit_semantics():
    package = load_class_package(EXAMPLES / "archer.json")
    reordered = replace(
        package, skills=tuple(reversed(package.skills)), programs=tuple(reversed(package.programs))
    )
    assert reordered == package and reordered.content_sha256 == package.content_sha256
    raw = source("bandit")
    raw["programs"][0]["steps"] = list(reversed(raw["programs"][0]["steps"]))
    with pytest.raises(ValueError, match="cooldown"):
        ClassPackage.from_dict(raw)


def test_bounded_local_loader_rejects_duplicates_oversize_and_symlinks(tmp_path):
    path = tmp_path / "package.json"
    path.write_bytes(b" " * (MAX_CLASS_PACKAGE_BYTES + 1))
    with pytest.raises(ValueError, match="64KiB"):
        load_class_package(path)
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        load_class_package(path)
    path.write_text('{"schema_version":NaN}')
    with pytest.raises(ValueError):
        load_class_package(path)
    link = tmp_path / "link.json"
    try:
        link.symlink_to(EXAMPLES / "archer.json")
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        load_class_package(link)


def test_hash_mismatch_wrong_compiler_and_legacy_job_contracts_remain_strict():
    package = load_class_package(EXAMPLES / "archer.json")
    for change in (
        {"content_sha256": "0" * 64},
        {"compiler_version": True},
        {"compiler_version": 2},
    ):
        with pytest.raises(ValueError):
            ClassPackage.from_manifest(package.manifest() | change)
    values = dict(CLASS_SPECS["assassin"].__dict__)
    values["id"] = "archer"
    with pytest.raises(ValueError):
        ClassSpec(**values)
    with pytest.raises(ValueError):
        SimAction("attack", skill="arrow_shot")
    with pytest.raises(ValueError):
        SimAction("cast", skill="../code")
