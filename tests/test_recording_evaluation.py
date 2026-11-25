import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from maple_vision.cli import main
from maple_vision.contracts import Detection
from maple_vision.recording_evaluation import benchmark_recordings, create_label_template
from maple_vision.recordings import import_recording
from maple_vision.scene import PALETTE


def case(root, name, color=(1, 2, 3)):
    folder = root / name
    folder.mkdir()
    for i in range(2):
        image = Image.new("RGB", (160, 96), color)
        if i == 0:
            ImageDraw.Draw(image).rectangle((60, 50, 69, 59), fill=PALETTE["monster"])
        image.save(folder / f"{i}.png")
    recording = root / (name + "-recording")
    import_recording(
        folder,
        recording,
        source_uri="urn:self:authored:" + name,
        permission="self_authored",
        license="CC0",
        fps=12,
    )
    labels = root / (name + ".json")
    create_label_template(recording, labels)
    data = json.loads(labels.read_text())
    data.update(reviewer="fixture-author", domain="authored_fixture")
    for entry in data["frames"]:
        entry.update(map_id=name, objects=[])
    data["frames"][0]["objects"] = [
        {"class_id": "monster", "bbox": [60, 50, 70, 60], "occluded": True}
    ]
    labels.write_text(json.dumps(data))
    return recording, labels


def test_cli_scores_all_frames_and_empty_ground_truth(tmp_path, capsys):
    recording, labels = case(tmp_path, "blue")
    template = tmp_path / "blank.json"
    assert main(["recording", "label-template", str(recording), str(template)]) == 0
    assert all(x["objects"] is None for x in json.loads(template.read_text())["frames"])
    output = tmp_path / "score.json"
    assert (
        main(
            [
                "recording",
                "benchmark",
                "--case",
                str(recording),
                str(labels),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    score = json.loads(output.read_text())
    assert score["frames"] == 2 and score["ground_truth_objects"] == 1
    assert score["metrics_at_50"]["micro"]["tp"] == 1 and score["metrics_at_50"]["micro"]["fn"] == 0
    assert score["recall_at_50"]["occluded"] == {"total": 1, "matched": 1, "recall": 1}
    assert score["recall_at_50"]["other_size"]["recall"] is None
    assert len(score["ap_by_iou"]) == 10 and score["mean_ap_50_95"] == 1
    assert score["training_overlap_verified"] is False
    before = output.read_bytes()
    assert (
        main(
            [
                "recording",
                "benchmark",
                "--case",
                str(recording),
                str(labels),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == before


class ImperfectDetector:
    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame.sequence)
        if frame.sequence == 0:
            return (Detection("monster", (60, 50, 67, 60), 0.9),)
        return (Detection("monster", (10, 10, 15, 15), 0.8),)


def test_pooled_predictions_and_localization_thresholds(tmp_path):
    first = case(tmp_path, "meadow")
    second = case(tmp_path, "cave", (5, 8, 10))
    detector = ImperfectDetector()
    report = benchmark_recordings([first, second], tmp_path / "report.json", detector)
    assert detector.frames == [0, 1, 0, 1]
    assert len(report["sessions"]) == 2 and set(report["by_map"]) == {"meadow", "cave"}
    assert (
        report["metrics_at_50"]["micro"]["tp"] == 2 and report["metrics_at_50"]["micro"]["fp"] == 2
    )
    assert report["ap_by_iou"][-1]["mean_ap"] == 0 and report["mean_ap_50_95"] == 0.5
    assert report["exact_rgb_hashes_shared_across_sessions"] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["frames"].pop(),
        lambda d: d.update(reviewer=None),
        lambda d: d.update(session_id="0" * 64),
        lambda d: d["frames"][0].update(sequence=True),
        lambda d: d["frames"][0].update(rgb_sha256="0" * 64),
        lambda d: d["frames"][0].update(objects=None),
        lambda d: d["frames"][0]["objects"][0].update(occluded=1),
        lambda d: d["frames"][0]["objects"][0].update(class_id="unknown"),
        lambda d: d["frames"][0]["objects"][0].update(bbox=[60, 50, 400, 80]),
    ],
)
def test_preflight_rejects_partial_or_unbound_labels_before_any_predictions(tmp_path, change):
    first = case(tmp_path, "first")
    rec, labels = case(tmp_path, "second", (5, 8, 10))
    raw = json.loads(labels.read_text())
    change(raw)
    labels.write_text(json.dumps(raw))
    detector = ImperfectDetector()
    output = tmp_path / "score.json"
    with pytest.raises(ValueError):
        benchmark_recordings([first, (rec, labels)], output, detector)
    assert not detector.frames and not output.exists()


def test_duplicate_sources_and_mixed_domains_rejected(tmp_path):
    pair = case(tmp_path, "duplicate")
    with pytest.raises(ValueError, match="duplicate source"):
        benchmark_recordings([pair, pair], tmp_path / "score.json")
    rec, labels = case(tmp_path, "second", (5, 8, 10))
    doc = json.loads(labels.read_text())
    doc["domain"] = "game_recording"
    labels.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="separate reports"):
        benchmark_recordings([pair, (rec, labels)], tmp_path / "score.json")


def test_declared_development_data_checked_on_decoded_pixels_before_inference(tmp_path):
    from maple_vision.dataset import generate_dataset

    dataset = tmp_path / "development"
    generate_dataset(dataset, sessions=3, frames_per_session=2)
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    development = [r for r in manifest["samples"] if r["split"] in ("train", "val")]
    recording, labels = case(tmp_path, "unseen")
    result = benchmark_recordings(
        [(recording, labels)], tmp_path / "disjoint.json", development_datasets=[dataset]
    )
    check = result["declared_development_exact_rgb_checks"][0]
    assert check["train_val_frames"] == len(development) and len(check["manifest_sha256"]) == 64
    copied = tmp_path / "reencoded"
    copied.mkdir()
    with Image.open(dataset / development[0]["image"]) as original:
        original.convert("RGB").save(copied / "changed.png", compress_level=0)
    overlap = tmp_path / "overlap"
    import_recording(
        copied,
        overlap,
        source_uri="urn:self:copy",
        permission="self_authored",
        license="CC0",
        fps=12,
    )
    label_copy = tmp_path / "overlap.json"
    create_label_template(overlap, label_copy)
    doc = json.loads(label_copy.read_bytes())
    doc.update(reviewer="fixture-author", domain="authored_fixture")
    doc["frames"][0].update(map_id="copy", objects=[])
    label_copy.write_text(json.dumps(doc))
    detector = ImperfectDetector()
    with pytest.raises(ValueError, match="overlap"):
        benchmark_recordings(
            [(overlap, label_copy)],
            tmp_path / "rejected.json",
            detector,
            development_datasets=[dataset],
        )
    assert not detector.frames and not (tmp_path / "rejected.json").exists()


def test_input_bundles_are_immutable(tmp_path):
    recording, labels = case(tmp_path, "protected")
    with pytest.raises(ValueError, match="outside"):
        create_label_template(recording, recording / "labels.json")
    with pytest.raises(ValueError, match="outside"):
        benchmark_recordings([(recording, labels)], recording / "score.json")


@pytest.mark.parametrize(
    "target,listed", [("east_hill", 1), ("east_forest", 0), ("unlisted", 0), ("henesys", 0)]
)
def test_reviewed_map_changes(tmp_path, target, listed):
    from pathlib import Path

    recording, labels = case(tmp_path, "henesys")
    data = json.loads(labels.read_text())
    data["frames"][1]["map_id"] = target
    labels.write_text(json.dumps(data))
    graph = Path(__file__).resolve().parents[1] / "examples/navigation/cms-east-hill.json"
    output = tmp_path / "route.json"
    assert main(["recording", "audit-route", *map(str, [recording, labels, graph, output])]) == 0
    report = json.loads(output.read_text())
    assert report["frames"] == 2 and report["observed_changes"] == int(target != "henesys")
    assert report["listed_direct_changes"] == listed
    assert report["map_ids_outside_graph"] == (["unlisted"] if target == "unlisted" else [])
    if report["changes"]:
        change = report["changes"][0]
        assert (change["before_ms"], change["after_ms"]) == (0, 83)


def test_measure_reviewed_strike(tmp_path):
    session, labels = case(tmp_path, "forest")
    data = json.loads(labels.read_text())
    data["frames"][0]["objects"].append(
        {"class_id": "player", "bbox": [90, 45, 100, 60], "occluded": False}
    )
    labels.write_text(json.dumps(data))
    output = tmp_path / "one.json"
    args = [
        "recording",
        "measure-strike",
        str(session),
        str(labels),
        str(output),
        "--start",
        "0",
        "--end",
        "1",
        "--skill-id",
        "example",
        "--level",
        "1",
        "--weapon",
        "fixture",
        "--reviewer",
        "tester",
        "--facing",
        "left",
        "--feedback",
        "no_feedback",
        "--player-index",
        "1",
        "--monster-index",
        "0",
    ]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["window_ms"] == 83
    assert main(["recording", "verify-strike", str(session), str(labels), str(output)]) == 0
    geometry = report["start_geometry"]
    assert geometry["forward_center_delta_px"] == 30 and geometry["horizontal_edge_gap_px"] == 20
    assert geometry["monster_occluded"] and report["reviewed_feedback"] == "no_feedback"
    assert main(args) == 2
    output.unlink()
    data["frames"][1]["map_id"] = "changed"
    labels.write_text(json.dumps(data))
    assert main(args) == 2 and not output.exists()


def test_measure_reviewed_strike_preserves_distinct_feedback_frame_timing(tmp_path):
    session, labels = case(tmp_path, "forest")
    data = json.loads(labels.read_text())
    data["frames"][0]["objects"].append(
        {"class_id": "player", "bbox": [90, 45, 100, 60], "occluded": False}
    )
    labels.write_text(json.dumps(data))
    output = tmp_path / "sequence.json"
    args = [
        "recording",
        "measure-strike",
        str(session),
        str(labels),
        str(output),
        "--start",
        "0",
        "--end",
        "1",
        "--skill-id",
        "example",
        "--level",
        "1",
        "--weapon",
        "fixture",
        "--reviewer",
        "tester",
        "--facing",
        "left",
        "--feedback",
        "damage_visible",
        "--feedback-frame",
        "0",
        "--player-index",
        "1",
        "--monster-index",
        "0",
    ]
    movement_args = args + ["--movement-frame", "1"]
    args.extend(["--feedback-frame", "1"])
    assert main(args) == 0
    report = json.loads(output.read_text())
    verify = ["recording", "verify-strike", str(session), str(labels), str(output)]
    assert main(verify) == 0
    summary_path = tmp_path / "summary.json"
    summary_args = [
        "recording",
        "summarize-strikes",
        "--verified-case",
        str(session),
        str(labels),
        str(output),
        "--output",
        str(summary_path),
    ]
    assert main(summary_args) == 0
    summary = json.loads(summary_path.read_text())
    assert summary["source_fields_verified"] == summary["observations"] == 1
    assert summary["source_reports_sha256"] == [hashlib.sha256(output.read_bytes()).hexdigest()]
    assert summary["domain"] == "authored_fixture"
    assert (
        not summary["calibrated_range"] and not summary["feedback_sequences"]["calibrated_timing"]
    )
    summary_path.unlink()
    for field, key, value in (
        ("start_geometry", "horizontal_edge_gap_px", 999),
        ("end", "rgb_sha256", "0" * 64),
    ):
        changed = json.loads(json.dumps(report))
        changed[field][key] = value
        output.write_text(json.dumps(changed))
        assert main(verify) == 2
        assert main(summary_args) == 2 and not summary_path.exists()
    changed = json.loads(json.dumps(report))
    changed["feedback_frames"][0]["rgb_sha256"] = "0" * 64
    output.write_text(json.dumps(changed))
    assert main(verify) == 2
    assert report["schema_version"] == 2
    assert [frame["offset_ms"] for frame in report["feedback_frames"]] == [0, 83]
    assert [frame["gap_ms"] for frame in report["feedback_frames"]] == [None, 83]
    output.unlink()
    assert main(movement_args) == 0
    measured = json.loads(output.read_text())
    assert measured["schema_version"] == 3
    assert measured["reviewed_movement"]["after_last_feedback_ms"] == 83
    assert main(verify) == 0 and main(summary_args) == 0
    summary = json.loads(summary_path.read_text())
    assert summary["observed_movement_after_feedback_ms"] == {
        "observations": 1,
        "min": 83,
        "median": 83,
        "max": 83,
    }
    measured["reviewed_movement"]["rgb_sha256"] = "0" * 64
    output.write_text(json.dumps(measured))
    assert main(verify) == 2
    output.unlink()
    assert main(args + ["--movement-frame", "0"]) == 2 and not output.exists()
    invalid = args.copy()
    invalid[invalid.index("damage_visible")] = "no_feedback"
    assert main(invalid) == 2 and not output.exists()


def test_summarize_reviewed_strikes_without_inventing_range(tmp_path):
    reports = []
    for index, (feedback, forward, window, facing) in enumerate(
        [
            ("damage_visible", 45, 100, "right"),
            ("damage_visible", 62, 120, "left"),
            ("miss_text_visible", 70, 80, "right"),
            ("no_feedback", 90, 150, "right"),
        ]
    ):
        path = tmp_path / f"strike-{index}.json"
        document = {
            "schema_version": 1,
            "kind": "reviewed_strike_window",
            "session_id": f"session-{index}",
            "domain": "game_recording",
            "labels_sha256": "a" * 64,
            "label_reviewer": "labeler",
            "strike_reviewer": "reviewer",
            "skill_id": "1311001",
            "level": 30,
            "weapon": "spear",
            "facing": facing,
            "map_id": "dead_forest_1",
            "reviewed_feedback": feedback,
            "timebase": "source_pts",
            "start": {"sequence": index * 2, "timestamp_ms": 0, "rgb_sha256": f"{index + 1:064x}"},
            "end": {
                "sequence": index * 2 + 1,
                "timestamp_ms": window,
                "rgb_sha256": f"{index + 20:064x}",
            },
            "window_ms": window,
            "source_frame_size": [1920, 1080],
            "start_geometry": {
                "player_index": 0,
                "monster_index": 1,
                "player_bbox": [10, 10, 20, 30],
                "monster_bbox": [30, 10, 40, 30],
                "player_occluded": False,
                "monster_occluded": False,
                "forward_center_delta_px": forward,
                "vertical_center_delta_px": 0,
                "horizontal_edge_gap_px": 10,
            },
        }
        path.write_text(json.dumps(document))
        reports.append(path)
    output = tmp_path / "summary.json"
    args = ["recording", "summarize-strikes"]
    for report in reports:
        args.extend(("--report", str(report)))
    args.extend(("--output", str(output)))
    assert main(args) == 0
    summary = json.loads(output.read_text())
    assert summary["schema_version"] == 1 and "feedback_sequences" not in summary
    assert summary["observations"] == 4 and summary["sessions"] == 4
    assert summary["observed_damage_distance_lower_bound_px"] == 62
    assert summary["range_upper_bound_px"] is None and not summary["calibrated_range"]
    assert summary["by_feedback"]["damage_visible"]["window_ms"] == {
        "min": 100,
        "median": 110.0,
        "max": 120,
    }
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(reports[0].read_bytes())
    output.unlink()
    assert (
        main(
            [
                "recording",
                "summarize-strikes",
                "--report",
                str(reports[0]),
                "--report",
                str(duplicate),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()

    repeat = json.loads(reports[0].read_bytes())
    repeat["end"].update(sequence=3, timestamp_ms=180, rgb_sha256="e" * 64)
    repeat["window_ms"] = 180
    duplicate.write_text(json.dumps(repeat))
    pair = [
        "recording",
        "summarize-strikes",
        "--report",
        str(reports[0]),
        "--report",
        str(duplicate),
        "--output",
        str(output),
    ]
    assert main(pair) == 2 and not output.exists()

    # A later cast can have exactly the same start and end pixels.
    later = json.loads(reports[0].read_bytes())
    for frame in ("start", "end"):
        later[frame]["sequence"] += 10
        later[frame]["timestamp_ms"] += 1000
    duplicate.write_text(json.dumps(later))
    assert main(pair) == 0
    combined = json.loads(output.read_bytes())
    assert combined["sessions"] == 1 and combined["observations"] == 2
    assert combined["timebase"] == later["timebase"]
    output.unlink()
    later["timebase"] = "nominal_fps_user"
    duplicate.write_text(json.dumps(later))
    assert main(pair) == 2 and not output.exists()


def test_summarize_feedback_sequences_without_claiming_hit_timing_calibration(tmp_path):
    reports = []
    for index, offsets in enumerate(([40, 80, 130], [50, 100, 150])):
        frames = []
        for frame_index, offset in enumerate(offsets):
            frames.append(
                {
                    "sequence": index * 10 + frame_index + 1,
                    "timestamp_ms": offset,
                    "rgb_sha256": f"{index * 10 + frame_index + 30:064x}",
                    "offset_ms": offset,
                    "gap_ms": None if frame_index == 0 else offset - offsets[frame_index - 1],
                }
            )
        document = {
            "schema_version": 2,
            "kind": "reviewed_strike_window",
            "session_id": f"session-{index}",
            "domain": "game_recording",
            "labels_sha256": "a" * 64,
            "label_reviewer": "labeler",
            "strike_reviewer": "reviewer",
            "skill_id": "1311001",
            "level": 30,
            "weapon": "spear",
            "facing": "right",
            "map_id": "dead_forest_1",
            "reviewed_feedback": "damage_visible",
            "timebase": "source_pts",
            "start": {"sequence": index * 10, "timestamp_ms": 0, "rgb_sha256": f"{index + 1:064x}"},
            "end": {
                "sequence": index * 10 + 4,
                "timestamp_ms": 180,
                "rgb_sha256": f"{index + 20:064x}",
            },
            "window_ms": 180,
            "source_frame_size": [1920, 1080],
            "start_geometry": {
                "player_index": 0,
                "monster_index": 1,
                "player_bbox": [10, 10, 20, 30],
                "monster_bbox": [30, 10, 40, 30],
                "player_occluded": False,
                "monster_occluded": False,
                "forward_center_delta_px": 45 + index,
                "vertical_center_delta_px": 0,
                "horizontal_edge_gap_px": 10,
            },
            "feedback_frames": frames,
        }
        path = tmp_path / f"sequence-{index}.json"
        path.write_text(json.dumps(document))
        reports.append(path)
    output = tmp_path / "summary.json"
    args = ["recording", "summarize-strikes"]
    for report in reports:
        args.extend(("--report", str(report)))
    args.extend(("--output", str(output)))
    assert main(args) == 0
    summary = json.loads(output.read_text())
    assert summary["schema_version"] == 2
    assert summary["feedback_sequences"] == {
        "observations": 2,
        "event_count_distribution": {"3": 2},
        "first_feedback_offset_ms": {"min": 40, "median": 45.0, "max": 50},
        "inter_feedback_gap_ms": {"observations": 4, "min": 40, "median": 50.0, "max": 50},
        "calibrated_timing": False,
    }
    assert not summary["calibrated_range"]
    output.unlink()
    forged = json.loads(reports[0].read_text())
    forged["feedback_frames"][1]["gap_ms"] = 999
    reports[0].write_text(json.dumps(forged))
    assert main(args) == 2 and not output.exists()
