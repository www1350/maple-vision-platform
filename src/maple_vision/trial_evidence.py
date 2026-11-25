"""Bound and independently recheck separated offline trial evidence lanes."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from .catalog import canonical_json, strict_json
from .input_bindings import load_input_profile
from .lab import verify_perception, verify_run
from .local_files import read_regular
from .recording_evaluation import _event_report, _recording_window_report

MAX_TRIAL_LEDGER_BYTES = 256 * 1024
MAX_TRIAL_FILE_BYTES = 32 * 1024 * 1024
MAX_TRIAL_ARCHIVE_FILES = 4096
MAX_TRIAL_ARCHIVE_BYTES = 256 * 1024 * 1024


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {label} fields")
    return value


def _integer(value: object, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"invalid {label}")
    return value


def _rooted_file(
    root: Path,
    reference: object,
    label: str,
    *,
    canonical_text: bool = False,
) -> tuple[Path, bytes]:
    item = _object(reference, {"path", "sha256"}, label)
    relative = PurePosixPath(_text(item["path"], f"{label} path"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid {label} path")
    sha256 = _text(item["sha256"], f"{label} sha256")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError(f"invalid {label} sha256")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symbolic links")
    data = read_regular(path, MAX_TRIAL_FILE_BYTES, label)
    checksum_data = data.replace(b"\r\n", b"\n") if canonical_text else data
    if hashlib.sha256(checksum_data).hexdigest() != sha256:
        raise ValueError(f"{label} checksum mismatch")
    return path, data


def _json_file(root: Path, reference: object, label: str) -> tuple[Path, dict[str, Any]]:
    path, data = _rooted_file(root, reference, label, canonical_text=True)
    try:
        value = strict_json(data.decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return path, value


def _extract_run(data: bytes, target: Path) -> Path:
    files = total = 0
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_TRIAL_ARCHIVE_FILES:
            raise ValueError("trial archive exceeds file-count budget")
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or relative.parts[0] != "run"
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not (member.isdir() or member.isreg())
            ):
                raise ValueError("trial archive contains an unsafe member")
            key = relative.as_posix().rstrip("/")
            if key in seen:
                raise ValueError("trial archive contains a duplicate member")
            seen.add(key)
            if member.isreg():
                files += 1
                total += member.size
                if (
                    member.size > MAX_TRIAL_FILE_BYTES
                    or files > MAX_TRIAL_ARCHIVE_FILES
                    or total > MAX_TRIAL_ARCHIVE_BYTES
                ):
                    raise ValueError("trial archive exceeds byte budget")
        for member in members:
            relative = PurePosixPath(member.name)
            destination = target.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("trial archive member cannot be read")
            data_value = source.read(member.size + 1)
            if len(data_value) != member.size:
                raise ValueError("trial archive member changed size")
            with destination.open("xb") as stream:
                stream.write(data_value)
    run = target / "run"
    if not run.is_dir():
        raise ValueError("trial archive omits its run root")
    return run


def _verify_run_archive(root: Path, reference: object, label: str) -> dict[str, Any]:
    _, data = _rooted_file(root, reference, label)
    with TemporaryDirectory(prefix="maple-trial-") as temporary:
        run = _extract_run(data, Path(temporary))
        replay = verify_run(run)
        reinference = verify_perception(run)
        summary = strict_json((run / "summary.json").read_text(encoding="utf-8"))
        final_state = strict_json((run / "final-state.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(final_state, dict):
        raise ValueError("trial archive run facts must be objects")
    return {
        "replay_verified": replay.get("verified") is True,
        "steps_replayed": replay.get("steps_replayed"),
        "goal_verified": replay.get("independent_goal_check") is True,
        "reinference_verified": reinference.get("verified") is True,
        "frames_reinferred": reinference.get("frames_reinferred"),
        "perception_reproduced": reinference.get("perception_reproduced") is True,
        "policy_reproduced": reinference.get("policy_reproduced") is True,
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "events": summary.get("events"),
        "input_dispatch": summary.get("input_dispatch_summary"),
        "final_ammo": final_state.get("player", {}).get("ammo"),
    }


def _verify_coverage(coverage: dict[str, Any]) -> None:
    declared = coverage.get("coverage_sha256")
    measured = dict(coverage)
    measured.pop("coverage_sha256", None)
    if (
        not isinstance(declared, str)
        or hashlib.sha256(canonical_json(measured).encode()).hexdigest() != declared
    ):
        raise ValueError("recording coverage identity mismatch")
    windows = coverage.get("windows")
    if (
        not isinstance(windows, list)
        or coverage.get("window_count") != len(windows)
        or coverage.get("contiguous") is not True
        or coverage.get("decoder_stop_reached") is not True
    ):
        raise ValueError("recording coverage is not a complete contiguous chain")
    previous = -1
    count = 0
    for index, window_value in enumerate(windows):
        window = _object(
            window_value,
            {
                "session_id",
                "schema_version",
                "source_start_frame",
                "source_end_frame",
                "frame_count",
                "truncated",
                "completion_reason",
                "first",
                "last",
            },
            "recording coverage window",
        )
        start = _integer(window["source_start_frame"], 0, 2**53, "source start frame")
        end = _integer(window["source_end_frame"], start, 2**53, "source end frame")
        frames = _integer(window["frame_count"], 1, 2**31, "window frame count")
        if start != previous + 1 or end != start + frames - 1:
            raise ValueError("recording coverage window ranges are inconsistent")
        if index < len(windows) - 1:
            if window["truncated"] is not True or window["completion_reason"] != "frame_limit":
                raise ValueError("non-final recording coverage window must end at its frame limit")
        elif window["truncated"] is not False or window["completion_reason"] != "decoder_stopped":
            raise ValueError("final recording coverage window must reach decoder stop")
        previous, count = end, count + frames
    if coverage.get("decoded_frame_count") != count or coverage.get("decoded_source_range") != [
        0,
        previous,
    ]:
        raise ValueError("recording coverage totals are inconsistent")


def _verify_event_sources(event_reports: list[dict[str, Any]], coverage: dict[str, Any]) -> int:
    source = coverage.get("source")
    if not isinstance(source, dict):
        raise ValueError("recording coverage omits source identity")
    windows = {window["session_id"]: window for window in coverage["windows"]}
    total = 0
    for report in event_reports:
        report_source = report.get("source")
        if not isinstance(report_source, dict):
            raise ValueError("reviewed event report omits source identity")
        window = windows.get(report_source.get("session_id"))
        if window is None:
            raise ValueError("reviewed event report is outside the coverage chain")
        for field in ("source_sha256", "source_name", "source_uri", "permission", "license"):
            if report_source.get(field) != source.get(field):
                raise ValueError("reviewed event report source differs from coverage")
        if report_source.get("frame_count") != window["frame_count"]:
            raise ValueError("reviewed event report frame count differs from coverage")
        if window["source_start_frame"]:
            if (
                report_source.get("source_start_frame") != window["source_start_frame"]
                or report_source.get("source_end_frame") != window["source_end_frame"]
            ):
                raise ValueError("reviewed event report source range differs from coverage")
        elif "source_start_frame" in report_source or "source_end_frame" in report_source:
            raise ValueError("first-window event report has an unexpected source offset")
        events = report.get("events")
        if not isinstance(events, list):
            raise ValueError("reviewed event report omits events")
        for event_value in events:
            event = _object(event_value, {"kind", "start", "end", "duration_ms"}, "visible event")
            start = event.get("start")
            end = event.get("end")
            if not isinstance(start, dict) or not isinstance(end, dict):
                raise ValueError("visible event omits frame boundaries")
            first = _integer(start.get("sequence"), 0, window["frame_count"] - 1, "event start")
            _integer(end.get("sequence"), first, window["frame_count"] - 1, "event end")
            if not isinstance(event.get("kind"), str) or not event["kind"]:
                raise ValueError("visible event kind is invalid")
            if type(event.get("duration_ms")) is not int or event["duration_ms"] < 0:
                raise ValueError("visible event duration is invalid")
            total += 1
    return total


def verify_trial_evidence(path: Path, root: Path) -> dict[str, Any]:
    """Verify four separated evidence lanes without promoting real-game claims."""
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("trial evidence root must be a real directory")
    manifest_path = Path(path).absolute()
    if not manifest_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("trial ledger must be inside its evidence root")
    raw = strict_json(read_regular(manifest_path, MAX_TRIAL_LEDGER_BYTES, "trial ledger").decode())
    ledger = _object(
        raw,
        {"schema_version", "kind", "trial_id", "scope", "evidence", "limitations"},
        "trial ledger",
    )
    if ledger["schema_version"] != 1 or ledger["kind"] != "offline_trial_evidence_ledger":
        raise ValueError("unsupported trial ledger")
    trial_id = _text(ledger["trial_id"], "trial ID")
    if ledger["scope"] != "synthetic_fixture_and_separate_read_only_recording":
        raise ValueError("invalid trial evidence scope")
    evidence = _object(
        ledger["evidence"],
        {"real_recording", "synthetic_mechanics", "role_binding", "ammunition"},
        "trial evidence",
    )
    real = _object(
        evidence["real_recording"],
        {
            "coverage",
            "event_reports",
            "expected_coverage_sha256",
            "expected_frames",
            "expected_events",
        },
        "real recording evidence",
    )
    coverage_path, coverage = _json_file(root, real["coverage"], "recording coverage report")
    if _recording_window_report(coverage_path) != coverage:
        raise ValueError("recording coverage report changed while verifying")
    _verify_coverage(coverage)
    event_references = real["event_reports"]
    if not isinstance(event_references, list) or not 1 <= len(event_references) <= 32:
        raise ValueError("trial ledger must reference 1..32 event reports")
    event_reports = []
    for index, reference in enumerate(event_references):
        event_path, report = _json_file(root, reference, f"event report {index}")
        if _event_report(event_path) != report:
            raise ValueError("reviewed event report changed while verifying")
        event_reports.append(report)
    event_count = _verify_event_sources(event_reports, coverage)
    if (
        coverage["coverage_sha256"] != real["expected_coverage_sha256"]
        or coverage["decoded_frame_count"] != real["expected_frames"]
        or event_count != real["expected_events"]
    ):
        raise ValueError("real recording evidence differs from trial expectations")

    synthetic = _object(
        evidence["synthetic_mechanics"],
        {
            "acceptance_report",
            "successful_run_archive",
            "wrong_binding_archive",
            "class_package",
            "map_package",
            "mission_program",
            "expected",
        },
        "synthetic mechanics evidence",
    )
    _, acceptance = _json_file(root, synthetic["acceptance_report"], "trial acceptance report")
    source_refs = {
        "class_package": synthetic["class_package"],
        "map_pack": synthetic["map_package"],
        "mission_program": synthetic["mission_program"],
    }
    for field, reference in source_refs.items():
        source_path, _ = _rooted_file(root, reference, field.replace("_", " "), canonical_text=True)
        if (
            acceptance.get("synthetic_inputs", {}).get(field) != reference["path"]
            or acceptance["synthetic_inputs"].get(f"{field}_sha256") != reference["sha256"]
            or source_path.is_symlink()
        ):
            raise ValueError("synthetic source differs from the acceptance report")
    expected = _object(
        synthetic["expected"],
        {"steps", "climbs", "casts", "hit_segments", "kills", "reinferred_frames"},
        "synthetic trial expectations",
    )
    positive = acceptance.get("positive_result")
    if not isinstance(positive, dict) or (
        positive.get("status"),
        positive.get("steps"),
        positive.get("actions", {}).get("climb"),
        positive.get("actions", {}).get("cast"),
        positive.get("events", {}).get("attack_hit"),
        positive.get("events", {}).get("monster_defeated"),
    ) != (
        "completed",
        expected["steps"],
        expected["climbs"],
        expected["casts"],
        expected["hit_segments"],
        expected["kills"],
    ):
        raise ValueError("synthetic acceptance result differs from trial expectations")
    if (
        acceptance.get("positive_archive", {}).get("reinference", {}).get("frames_reinferred")
        != expected["reinferred_frames"]
    ):
        raise ValueError("synthetic reinference denominator differs from the ledger")
    positive_archive = _verify_run_archive(
        root, synthetic["successful_run_archive"], "successful trial archive"
    )
    wrong_archive = _verify_run_archive(
        root, synthetic["wrong_binding_archive"], "wrong-binding trial archive"
    )
    positive_reference = synthetic["successful_run_archive"]
    wrong_reference = synthetic["wrong_binding_archive"]
    if (
        acceptance.get("positive_archive", {}).get("file")
        != PurePosixPath(positive_reference["path"]).name
        or acceptance.get("positive_archive", {}).get("sha256") != positive_reference["sha256"]
        or acceptance.get("wrong_binding_result", {}).get("archive", {}).get("file")
        != PurePosixPath(wrong_reference["path"]).name
        or acceptance.get("wrong_binding_result", {}).get("archive", {}).get("sha256")
        != wrong_reference["sha256"]
    ):
        raise ValueError("trial archives differ from the acceptance report")
    if (
        positive_archive["status"] != "completed"
        or positive_archive["events"].get("attack_hit") != expected["hit_segments"]
        or positive_archive["events"].get("monster_defeated") != expected["kills"]
        or wrong_archive["status"] != "failed"
        or wrong_archive["reason"] != "combat_progress_timeout"
        or wrong_archive["events"].get("attack_hit", 0) != 0
    ):
        raise ValueError("replayed trial archives differ from acceptance outcomes")

    binding = _object(
        evidence["role_binding"],
        {"profile", "profile_id", "key", "skill", "accepted_dispatches", "denied_dispatches"},
        "role binding evidence",
    )
    profile_path, _ = _rooted_file(
        root, binding["profile"], "input binding profile", canonical_text=True
    )
    profile = load_input_profile(profile_path)
    configured = profile.binding_for_key(_text(binding["key"], "binding key"))
    if (
        profile.profile_id != binding["profile_id"]
        or configured is None
        or configured.skill != binding["skill"]
        or acceptance.get("positive_result", {}).get("input_dispatch", {}).get("accepted")
        != binding["accepted_dispatches"]
        or acceptance.get("wrong_binding_result", {}).get("denied_casts")
        != binding["denied_dispatches"]
        or positive_archive["input_dispatch"].get("accepted") != binding["accepted_dispatches"]
        or wrong_archive["input_dispatch"].get("denied") != binding["denied_dispatches"]
    ):
        raise ValueError("role binding evidence differs from the trial ledger")

    ammunition = _object(
        evidence["ammunition"],
        {"scope", "initial", "per_cast", "successful_casts", "final"},
        "ammunition evidence",
    )
    initial = _integer(ammunition["initial"], 0, 10**6, "initial ammunition")
    per_cast = _integer(ammunition["per_cast"], 0, 200, "ammunition per cast")
    casts = _integer(ammunition["successful_casts"], 0, 10**6, "successful casts")
    final = _integer(ammunition["final"], 0, 10**6, "final ammunition")
    if (
        ammunition["scope"] != "synthetic_fixture_rule"
        or initial - per_cast * casts != final
        or acceptance.get("synthetic_inputs", {}).get("ammo_rule")
        != {"initial": initial, "per_cast": per_cast, "successful_casts": casts, "final": final}
        or positive_archive["final_ammo"] != final
        or wrong_archive["final_ammo"] != initial
    ):
        raise ValueError("ammunition evidence differs from the synthetic rule and run")
    limitations = ledger["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item for item in limitations)
    ):
        raise ValueError("trial ledger must retain explicit limitations")
    return {
        "verified": True,
        "trial_id": trial_id,
        "scope": ledger["scope"],
        "evidence_lanes": {
            "real_recording": {
                "persistent_reports_verified": True,
                "raw_pixels_reverified": False,
                "coverage_sha256": coverage["coverage_sha256"],
                "frames": coverage["decoded_frame_count"],
                "reviewed_visible_events": event_count,
            },
            "synthetic_mechanics": {
                "replay_verified": positive_archive["replay_verified"],
                "steps_replayed": positive_archive["steps_replayed"],
                "goal_verified": positive_archive["goal_verified"],
                "reinference_verified": positive_archive["reinference_verified"],
                "frames_reinferred": positive_archive["frames_reinferred"],
                "perception_reproduced": positive_archive["perception_reproduced"],
                "policy_reproduced": positive_archive["policy_reproduced"],
                "steps": expected["steps"],
                "casts": expected["casts"],
                "hit_segments": expected["hit_segments"],
                "kills": expected["kills"],
            },
            "role_binding": {
                "profile_id": profile.profile_id,
                "key": configured.key,
                "skill": configured.skill,
                "accepted_dispatches": binding["accepted_dispatches"],
                "wrong_profile_denials": binding["denied_dispatches"],
                "wrong_run_replay_verified": wrong_archive["replay_verified"],
                "wrong_run_reinference_verified": wrong_archive["reinference_verified"],
            },
            "ammunition": {
                "scope": ammunition["scope"],
                "initial": initial,
                "per_cast": per_cast,
                "successful_casts": casts,
                "final": final,
            },
        },
        "limitations": limitations,
    }
