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
