"""Counterexamples for independent ground truth and layered archive integrity."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

import maple_vision.failure_bundle as failure
from maple_vision.agent import AgentDecision, VisualAgent
from maple_vision.animation_pack import verify_animation_pack
from maple_vision.class_package import load_class_package
from maple_vision.failure_bundle import export_failure_bundle, verify_failure_bundle
from maple_vision.lab import CapabilityRunConfig, RunConfig, run_episode
from maple_vision.map_pack import load_map_pack
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.scene_animation import bind_scene_animation
from maple_vision.sim_types import SimAction
from maple_vision.simulation import Simulator


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_rows(path, values):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in values),
        encoding="utf-8",
    )


def hashes(root):
    return {
        item.relative_to(root).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in root.rglob("*")
        if item.is_file()
    }


def rehash_run(root):
    manifest = read_json(root / "manifest.json")
    manifest["files"] = {
        name: value for name, value in hashes(root).items() if name != "manifest.json"
    }
    write_json(root / "manifest.json", manifest)


def rehash_bundle(root, *, source_changed=False):
    if source_changed:
        rehash_run(root / "source-run")
    manifest = read_json(root / "manifest.json")
    manifest["files"] = {
        name: value for name, value in hashes(root).items() if name != "manifest.json"
    }
    manifest["source"]["manifest_sha256"] = hashlib.sha256(
        (root / "source-run" / "manifest.json").read_bytes()
    ).hexdigest()
    write_json(root / "manifest.json", manifest)


@pytest.fixture(scope="module")
def base_bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("failure-base")
    result = {}
    for schema in (1, 2):
        run = root / f"run-{schema}"
        bundle = root / f"bundle-{schema}"
        run_episode(run, RunConfig(max_steps=3), runtime=None if schema == 1 else RuntimeConfig())
        export_failure_bundle(run, bundle)
        result[schema] = (run, bundle)
    return result


@pytest.fixture
def bundle(tmp_path, base_bundles):
    target = tmp_path / "bundle"
    shutil.copytree(base_bundles[2][1], target)
    return target


@pytest.mark.parametrize("schema", [1, 2])
def test_real_terminal_n_plus_one_and_read_only_verification(base_bundles, schema):
    source, bundle = base_bundles[schema]
    before = hashes(bundle)
    report = verify_failure_bundle(bundle)
    assert report["frames_reconstructed"] == 4
    assert report["steps_replayed"] == 3
    assert report["perception_reproduced"] is False
    assert report["reinference"] is None
    assert hashes(bundle) == before
    assert hashes(source) == {
        name.removeprefix("source-run/"): value
        for name, value in before.items()
        if name.startswith("source-run/")
    }
    for name in (
        "ground_truth.jsonl",
        "observations.jsonl",
        "world_states.jsonl",
        "task_trace.jsonl",
        "supervisor.jsonl",
        "frames.jsonl",
    ):
        values = rows(bundle / name)
        assert len(values) == 4
        assert [row["timestamp_ms"] for row in values] == [0, 50, 100, 150]
        assert [row["terminal"] for row in values] == [False, False, False, True]
    assert len(rows(bundle / "action_trace.jsonl")) == 3
    assert len(rows(bundle / "verification.jsonl")) == 3
    terminal = rows(bundle / "task_trace.jsonl")[-1]
    assert terminal["policy_evaluated"] is False
    assert terminal["decision"] is None
    assert terminal["termination_cause"] == "step_budget_exhausted"
    assert rows(bundle / "supervisor.jsonl")[-1]["guard"] is None
    expected = read_json(source / "summary.json")["perception_evidence"]["terminal"]["observation"]
    assert rows(bundle / "observations.jsonl")[-1]["observation"] == expected
    if schema == 1:
        assert all(row["belief"] is None for row in rows(bundle / "world_states.jsonl"))
        assert terminal["runtime_state"] is None
    else:
        assert terminal["runtime_state"]["state"] == "stopped"
        assert terminal["interruption"]["state"] == "stopped"


def test_freeze_input_truth_is_not_requested_attack(tmp_path):
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    run_episode(source, RunConfig(max_steps=28, fault="freeze_input"), runtime=RuntimeConfig())
    export_failure_bundle(source, bundle)
    actions = rows(bundle / "action_trace.jsonl")
    blocked = [row for row in actions[20:] if row["requested_action"]["kind"] != "idle"]
    assert blocked
    assert all(row["execution"]["applied_action"]["kind"] == "idle" for row in blocked)
    state = read_json(source / "initial-state.json")
    actual, counterfactual = Simulator.restore(state), Simulator.restore(state)
    for row in actions:
        actual.step(SimAction(**row["execution"]["applied_action"]))
        counterfactual.step(SimAction(**row["requested_action"]))
    final = rows(bundle / "ground_truth.jsonl")[-1]["simulator_state"]
    assert final == actual.save()
    assert final != counterfactual.save()
    shutil.rmtree(source)
    assert verify_failure_bundle(bundle)["steps_replayed"] == 28


@pytest.mark.parametrize("fault", ["brief_dropout", "duplicate_frames", "persistent_dropout"])
def test_fault_capture_identity_and_real_supervisor_terminal(tmp_path, fault):
    source, bundle = tmp_path / "run", tmp_path / "bundle"
    summary = run_episode(source, RunConfig(max_steps=40, fault=fault), runtime=RuntimeConfig())
    export_failure_bundle(source, bundle)
    frames = rows(bundle / "frames.jsonl")
    truth = rows(bundle / "ground_truth.jsonl")
    assert len(frames) == summary["steps"] + 1
    assert frames[20]["timestamp_ms"] == 1000
    if fault == "duplicate_frames":
        assert frames[20]["capture_sequence"] == 19
        assert frames[20]["capture_timestamp_ms"] == 950
        assert frames[20]["rgb_sha256"] == frames[19]["rgb_sha256"]
    else:
        assert frames[20]["capture_sequence"] == 20
        assert frames[20]["rgb_sha256"] == hashlib.sha256(bytes(320 * 192 * 3)).hexdigest()
        assert truth[20]["rendered_rgb_sha256"] != truth[20]["capture_rgb_sha256"]
    if fault == "persistent_dropout":
        assert summary["status"] == "stopped"
        last = rows(bundle / "task_trace.jsonl")[-1]
        assert last["decision"] is None and last["policy_evaluated"] is False
        assert last["termination_cause"] == "supervisor"
        assert rows(bundle / "supervisor.jsonl")[-1]["guard"]["state"] == "stopped"
    assert verify_failure_bundle(bundle, reinfer=True)["reinference"]["frames_reinferred"] == len(
        frames
    )


def test_success_is_exportable_and_keeps_real_terminal_policy(tmp_path):
    source, bundle = tmp_path / "run", tmp_path / "bundle"
    summary = run_episode(source)
    assert summary["status"] == "completed"
    report = export_failure_bundle(source, bundle)
    assert report["recorded_status"] == "completed"
    terminal = rows(bundle / "task_trace.jsonl")[-1]
    assert terminal["policy_evaluated"] is True
    assert terminal["decision"]["state"] == "completed"
    assert terminal["termination_cause"] == "policy"
    assert verify_failure_bundle(bundle)["action_replay"]["independent_goal_check"] is True


def test_reinfer_really_executes_and_cannot_reuse_an_archived_success(base_bundles, monkeypatch):
    calls = []
    actual = failure.verify_perception

    def counting(*args, **kwargs):
        result = actual(*args, **kwargs)
        calls.append(result["frames_reinferred"])
        return result

    monkeypatch.setattr(failure, "verify_perception", counting)
    bundle = base_bundles[2][1]
    assert verify_failure_bundle(bundle, reinfer=True)["perception_reproduced"] is True
    assert verify_failure_bundle(bundle)["perception_reproduced"] is False
    assert calls == [4]


def test_default_does_not_falsely_claim_a_recorded_observation_was_reinferred(
    tmp_path, base_bundles
):
    source = tmp_path / "source"
    shutil.copytree(base_bundles[1][0], source)
    trace = rows(source / "trace.jsonl")
    trace[0]["observation"]["hp_ratio"] = 0.123
    write_rows(source / "trace.jsonl", trace)
    rehash_run(source)
    bundle = tmp_path / "bundle"
    assert export_failure_bundle(source, bundle)["perception_reproduced"] is False
    assert verify_failure_bundle(bundle)["perception_reproduced"] is False
    with pytest.raises(ValueError, match="observation|perception"):
        verify_failure_bundle(bundle, reinfer=True)


@pytest.mark.parametrize(
    "name",
    [
        "ground_truth.jsonl",
        "observations.jsonl",
        "world_states.jsonl",
        "task_trace.jsonl",
        "action_trace.jsonl",
        "verification.jsonl",
        "supervisor.jsonl",
        "frames.jsonl",
    ],
)
def test_layer_rewrite_is_rejected_after_all_outer_hashes_are_repaired(bundle, name):
    values = rows(bundle / name)
    if name == "ground_truth.jsonl":
        values[-1]["simulator_state"]["player"]["hp"] -= 1
        encoded = json.dumps(
            values[-1]["simulator_state"], sort_keys=True, separators=(",", ":")
        ).encode()
        values[-1]["simulator_state_sha256"] = hashlib.sha256(encoded).hexdigest()
    elif name == "observations.jsonl":
        values[-1]["observation"]["hp_ratio"] = 0.5
    elif name == "world_states.jsonl":
        values[-1]["belief"]["actionable"] = not values[-1]["belief"]["actionable"]
    elif name == "task_trace.jsonl":
        values[-1]["policy_evaluated"] = True
    elif name == "action_trace.jsonl":
        values[-1]["requested_action"]["kind"] = "idle"
    elif name == "verification.jsonl":
        values[-1]["perception_reproduced"] = True
    elif name == "supervisor.jsonl":
        values[-1]["guard"] = {"state": "running", "reason": "invented"}
    else:
        values[-1]["capture_sequence"] = 2
    write_rows(bundle / name, values)
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="projection mismatch"):
        verify_failure_bundle(bundle)


@pytest.mark.parametrize("name", ["versions.lock", "config.yaml", "report.md"])
def test_static_layer_rewrite_is_not_just_checksum_checked(bundle, name):
    with (bundle / name).open("ab") as stream:
        stream.write(b"\nrewritten\n")
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="projection mismatch"):
        verify_failure_bundle(bundle)


def test_rehashed_wrong_pixels_are_rejected_by_actual_rgb_reconstruction(bundle):
    Image.new("RGB", (320, 192), (11, 13, 17)).save(bundle / "frames/000003.png")
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="RGB projection mismatch"):
        verify_failure_bundle(bundle)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("include_video", 0),
        ("counts", {"steps": 3, "frames": 3}),
        ("timeline", {"step_ms": 100}),
    ],
)
def test_manifest_schema_and_timeline_cannot_be_downgraded_or_relabelled(bundle, key, value):
    manifest = read_json(bundle / "manifest.json")
    manifest[key] = value
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_failure_bundle(bundle)


def test_repaired_nested_run_hashes_do_not_admit_a_forged_terminal_identity(bundle):
    summary = read_json(bundle / "source-run/summary.json")
    summary["perception_evidence"]["terminal"]["observation"]["sequence"] += 1
    write_json(bundle / "source-run/summary.json", summary)
    rehash_bundle(bundle, source_changed=True)
    with pytest.raises(ValueError, match="capture identity"):
        verify_failure_bundle(bundle)


def test_missing_terminal_is_rejected_instead_of_inventing_a_last_frame(tmp_path, base_bundles):
    source = tmp_path / "source"
    shutil.copytree(base_bundles[1][0], source)
    summary = read_json(source / "summary.json")
    summary.pop("perception_evidence")
    write_json(source / "summary.json", summary)
    rehash_run(source)
    with pytest.raises(ValueError, match="terminal perception evidence"):
        export_failure_bundle(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("operation", ["remove", "extra"])
def test_file_inventory_is_exact_even_after_rehash(bundle, operation):
    if operation == "remove":
        (bundle / "verification.jsonl").unlink()
    else:
        (bundle / "unexpected.txt").write_text("extra", encoding="utf-8")
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="required layers|unexpected files"):
        verify_failure_bundle(bundle)


def test_paths_cannot_escape_the_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    manifest["files"]["../outside"] = "0" * 64
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_failure_bundle(bundle)


def test_symlinked_parent_is_rejected_before_reading_frames(bundle, tmp_path):
    original = tmp_path / "outside"
    (bundle / "frames").rename(original)
    try:
        (bundle / "frames").symlink_to(original, target_is_directory=True)
    except OSError:
        pytest.skip("this OS user cannot create symlinks")
    with pytest.raises(ValueError, match="symlinks"):
        verify_failure_bundle(bundle)


@pytest.mark.parametrize(
    "budget,value", [("MAX_SOURCE_BYTES", 32), ("MAX_SOURCE_FILES", 2), ("MAX_FRAMES", 2)]
)
def test_source_budgets_fail_before_action_replay(
    tmp_path, base_bundles, monkeypatch, budget, value
):
    monkeypatch.setattr(failure, budget, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded source reached replay")

    monkeypatch.setattr(failure, "verify_run", forbidden)
    with pytest.raises(ValueError):
        export_failure_bundle(base_bundles[1][0], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_source_png_dimensions_are_checked_before_the_legacy_decoder(
    tmp_path, base_bundles, monkeypatch
):
    source = tmp_path / "source"
    shutil.copytree(base_bundles[1][0], source)
    Image.new("RGB", (321, 192)).save(source / "final.png")
    rehash_run(source)
    monkeypatch.setattr(failure, "verify_run", lambda *a: pytest.fail("decoded unbounded PNG"))
    with pytest.raises(ValueError, match="320 x 192"):
        export_failure_bundle(source, tmp_path / "output")


def test_partial_write_is_removed_and_does_not_change_source(tmp_path, base_bundles, monkeypatch):
    source = base_bundles[1][0]
    before = hashes(source)
    original = failure._Writer.line

    def fail_after_initial_frame(self, stream, row):
        if str(stream.name).endswith("ground_truth.jsonl") and row["frame_index"] == 1:
            raise OSError("synthetic disk failure")
        return original(self, stream, row)

    monkeypatch.setattr(failure._Writer, "line", fail_after_initial_frame)
    with pytest.raises(OSError, match="disk failure"):
        export_failure_bundle(source, tmp_path / "output")
    assert not list(tmp_path.iterdir())
    assert hashes(source) == before


def test_output_never_overwrites_and_cannot_be_inside_source(tmp_path, base_bundles):
    source = base_bundles[1][0]
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export_failure_bundle(source, output)
    assert (output / "keep").read_bytes() == b"keep"
    with pytest.raises(ValueError, match="outside"):
        export_failure_bundle(source, source / "nested-export")


@pytest.mark.parametrize("options", [{"reinfer": 1}, {"include_video": 1}, {"detector": object()}])
def test_export_options_do_not_accept_ambiguous_claims(tmp_path, base_bundles, options):
    with pytest.raises(ValueError):
        export_failure_bundle(base_bundles[1][0], tmp_path / "output", **options)


def test_external_map_animation_and_schema4_class_are_self_contained(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copytree(examples / "animations/fixture", inputs / "animation")
    shutil.copyfile(examples / "maps/three-map-fixture.json", inputs / "map.json")
    shutil.copyfile(examples / "classes/archer.json", inputs / "class.json")
    animation = bind_scene_animation(
        verify_animation_pack(inputs / "animation"), {"player": "fixture.player.move"}
    )
    source, bundle = tmp_path / "run", tmp_path / "bundle"
    run_episode(
        source,
        CapabilityRunConfig(job="archer", max_steps=6),
        runtime=RuntimeConfig(),
        map_pack=load_map_pack(inputs / "map.json"),
        animation=animation,
        class_package=load_class_package(inputs / "class.json"),
    )
    assert read_json(source / "initial-state.json")["schema_version"] == 4
    export_failure_bundle(source, bundle)
    shutil.rmtree(inputs)
    shutil.rmtree(source)
    result = verify_failure_bundle(bundle, reinfer=True)
    assert result["frames_reconstructed"] == 7
    assert result["perception_reproduced"] is True


@pytest.fixture(scope="module")
def video_bundle(tmp_path_factory, base_bundles):
    pytest.importorskip("cv2", reason="optional local MP4 encoder/decoder")
    root = tmp_path_factory.mktemp("failure-video") / "bundle"
    export_failure_bundle(base_bundles[1][0], root, include_video=True)
    return root


def test_actual_video_container_pts_and_decoded_hashes(video_bundle):
    info = read_json(video_bundle / "video-info.json")
    assert info["container"]["frames"] == 4
    assert info["container"]["last_pts_ms"] == 150
    assert info["container"]["duration_ms"] == 200
    assert info["container"]["width"] == 320 and info["container"]["height"] == 192
    assert [row["pts_ms"] for row in rows(video_bundle / "video_frames.jsonl")] == [0, 50, 100, 150]
    assert verify_failure_bundle(video_bundle)["video"] == {
        "present": True,
        "container_verified": True,
        "decode_check": "matched",
        "semantic_reproduction": False,
    }


def test_video_different_decoder_does_not_reject_authoritative_rgb(video_bundle, monkeypatch):
    monkeypatch.setattr(
        failure, "_codec_identity", lambda _: {"opencv": "different", "build_sha256": "0" * 64}
    )
    result = verify_failure_bundle(video_bundle)
    assert result["rgb_reproduced"] is True
    assert result["video"]["decode_check"] == "not_run_decoder_mismatch"
    assert result["video"]["semantic_reproduction"] is False


def test_no_decoder_still_checks_container_without_claiming_decoded_pixels(
    video_bundle, monkeypatch
):
    monkeypatch.setattr(failure, "_cv2", lambda **_: None)
    result = verify_failure_bundle(video_bundle)
    assert result["video"]["container_verified"] is True
    assert result["video"]["decode_check"] == "not_run_decoder_unavailable"


def test_rehashed_mp4_timing_change_is_rejected(tmp_path, video_bundle):
    bundle = tmp_path / "bundle"
    shutil.copytree(video_bundle, bundle)
    data = bytearray((bundle / "replay.mp4").read_bytes())
    delta_offset = data.index(b"stts") + 16
    delta = int.from_bytes(data[delta_offset : delta_offset + 4], "big")
    data[delta_offset : delta_offset + 4] = (delta + 1).to_bytes(4, "big")
    (bundle / "replay.mp4").write_bytes(data)
    info = read_json(bundle / "video-info.json")
    info["sha256"] = hashlib.sha256(data).hexdigest()
    write_json(bundle / "video-info.json", info)
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="sample timing"):
        verify_failure_bundle(bundle)


def test_rehashed_video_semantic_claim_is_rejected(tmp_path, video_bundle):
    bundle = tmp_path / "bundle"
    shutil.copytree(video_bundle, bundle)
    info = read_json(bundle / "video-info.json")
    info["semantic_reproduction"] = True
    write_json(bundle / "video-info.json", info)
    rehash_bundle(bundle)
    with pytest.raises(ValueError, match="source RGB timeline"):
        verify_failure_bundle(bundle)


def test_missing_local_encoder_fails_without_a_placeholder(tmp_path, base_bundles, monkeypatch):
    original = failure.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("unavailable locally")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(failure.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="video extra"):
        export_failure_bundle(base_bundles[1][0], tmp_path / "output", include_video=True)
    assert not (tmp_path / "output").exists()


def test_actual_false_completion_keeps_policy_decision_and_independent_failure(
    tmp_path, monkeypatch
):
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    with monkeypatch.context() as context:
        context.setattr(
            VisualAgent,
            "decide",
            lambda self, observation: AgentDecision(
                SimAction(), "finished", "completed", "premature"
            ),
        )
        summary = run_episode(source)
    assert summary["status"] == "failed"
    assert summary["reason"] == "independent_goal_check_failed"
    assert summary["perception_evidence"]["terminal"]["decision"]["state"] == "completed"
    exported = export_failure_bundle(source, bundle)
    assert exported["steps_replayed"] == 0 and exported["frames_reconstructed"] == 1
    assert exported["action_replay"]["independent_goal_check"] is False
    terminal = rows(bundle / "task_trace.jsonl")[0]
    assert terminal["decision"]["state"] == "completed"
    assert terminal["policy_evaluated"] is True
    assert verify_failure_bundle(bundle)["recorded_status"] == "failed"


def test_video_sample_offsets_checked_without_any_decoder(tmp_path, video_bundle, monkeypatch):
    bundle = tmp_path / "bundle"
    shutil.copytree(video_bundle, bundle)
    data = bytearray((bundle / "replay.mp4").read_bytes())
    offset = data.index(b"stco") + 12
    data[offset : offset + 4] = (len(data) + 1000).to_bytes(4, "big")
    (bundle / "replay.mp4").write_bytes(data)
    info = read_json(bundle / "video-info.json")
    info["sha256"] = hashlib.sha256(data).hexdigest()
    write_json(bundle / "video-info.json", info)
    rehash_bundle(bundle)
    monkeypatch.setattr(failure, "_cv2", lambda **_: None)
    with pytest.raises(ValueError, match="outside local media"):
        verify_failure_bundle(bundle)
