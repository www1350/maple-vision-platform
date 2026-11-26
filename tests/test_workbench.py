import base64
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from maple_vision.catalog import canonical_json
from maple_vision.recording_evaluation import review_events, summarize_recording_windows
from maple_vision.recordings import import_recording
from maple_vision.scene import ImageFrame
from maple_vision.workbench import Workbench
from maple_vision.workbench_api import create_app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
REPOSITORY = EXAMPLES.parent
TRIAL_LEDGER = REPOSITORY / "reports/release-0.10.0/throwing-star-trial-evidence-ledger.json"
ORIGIN = "http://127.0.0.1"


def client(root, examples=EXAMPLES):
    return TestClient(create_app(root, examples), base_url=ORIGIN)


def submit(http, body):
    reply = http.post("/api/jobs", json=body)
    assert reply.status_code == 202, reply.text
    return reply.json()["id"]


def exact_recording_window(root, name, start, pts, *, truncated):
    session = root / name
    frames = session / "frames"
    frames.mkdir(parents=True)
    rows = []
    first_pts = pts[0]
    for sequence, timestamp in enumerate(pts):
        pixels = bytes([40 + (start + sequence) * 20, 80, 120]) * (12 * 8)
        frame = ImageFrame(sequence, timestamp - first_pts, 12, 8, pixels)
        data = frame.png_bytes()
        file = f"frames/{sequence:06d}.png"
        (session / file).write_bytes(data)
        rows.append(
            {
                "sequence": sequence,
                "timestamp_ms": timestamp - first_pts,
                "file": file,
                "sha256": hashlib.sha256(data).hexdigest(),
                "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "source_name": f"video-frame-{start + sequence:06d}",
                "source_sha256": None,
                "width": 12,
                "height": 8,
                "pts": timestamp,
            }
        )
    manifest = {
        "schema_version": 2 if start == 0 else 3,
        "kind": "offline_recording",
        "scope": "user_supplied_recording",
        "source_kind": "mp4",
        "source_uri": "urn:self:workbench-window-source",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "source_name": "fixture.mp4",
        "source_sha256": "a" * 64,
        "source_hash_basis": "mp4_file",
        "fps": None,
        "timebase": "video_presentation_ticks",
        "frame_count": len(rows),
        "width": 12,
        "height": 8,
        "max_frames": len(rows),
        "total_source_frames": None,
        "truncated": truncated,
        "completion_reason": "frame_limit" if truncated else "decoder_stopped",
        "frames": rows,
        "video_clock": {"stream_index": 0, "numerator": 1, "denominator": 1000},
    }
    if start:
        manifest["source_start_frame"] = start
    manifest["session_id"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    (session / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return session


def wait(http, identity):
    for _ in range(1200):
        response = http.get(f"/api/jobs/{identity}")
        assert response.status_code == 200
        data = response.json()
        if data["state"] == "finished":
            return data
        assert data["state"] in {"queued", "running"}, data
        time.sleep(0.05)
    pytest.fail("job timed out")


def test_live_run_replay_layers_and_restart(tmp_path):
    with client(tmp_path) as http:
        assert http.get("/").status_code == 200
        assert http.get("/static/app.js").headers["content-type"].startswith("text/javascript")
        registered = http.get("/api/content").json()
        assert {"assassin", "archer"} <= set(registered["classes"])
        assert len(registered["maps"]["three-map-fixture"]) == 3
        body = dict(
            kind="run",
            job="archer",
            map_id="three-map-fixture",
            seed=7,
            max_steps=1500,
            fault="persistent_dropout",
        )
        rid = submit(http, body)
        run = wait(http, rid)
        assert run["outcome"] == run["result"]["status"] == "stopped"
        assert run["result"]["steps"] > 0
        assert run["frames"][0] == "000000" and run["frames"][-1] == "final"
        for name in ["000000", "final"]:
            assert http.get(f"/api/jobs/{rid}/frames/{name}").content[:8] == b"\x89PNG\r\n\x1a\n"
        first = http.get(f"/api/jobs/{rid}/trace").json()
        assert len(first) == 4 and first != http.get(f"/api/jobs/{rid}/trace?offset=4").json()
        for operation in ("replay", "reinfer", "export_bundle"):
            detail = wait(http, submit(http, {"kind": operation, "source": rid}))
            assert detail["outcome"] == "verified"
        bundle = detail
        assert len(bundle["frames"]) == run["result"]["steps"] + 1
        verified = wait(http, submit(http, {"kind": "verify_bundle", "source": bundle["id"]}))
        assert verified["frames"] == bundle["frames"]
        assert len(http.get("/api/jobs").json()) == 5
    with client(tmp_path) as restarted:
        assert restarted.get("/api/jobs/" + rid).json()["outcome"] == "stopped"
        assert restarted.get(f"/api/jobs/{rid}/frames/000000").status_code == 200


def test_revision_checked_single_step_save_reset_and_restore(tmp_path):
    with client(tmp_path) as http:
        initial = http.post("/api/sessions", json={"job": "bandit", "seed": 3}).json()
        uri = "/api/sessions/" + initial["id"]
        assert initial["sequence"] == 0 and base64.b64decode(initial["image"]).startswith(
            b"\x89PNG"
        )
        assert http.post(uri, json={"operation": "restore", "revision": 0}).status_code == 409
        request = {"operation": "step", "revision": 0, "action": {"kind": "move", "direction": 1}}
        with ThreadPoolExecutor(max_workers=2) as workers:
            statuses = list(
                workers.map(lambda _: http.post(uri, json=request).status_code, range(2))
            )
        assert sorted(statuses) == [200, 409]
        moved = http.get(uri).json()
        assert moved["sequence"] == 1 and moved["ground_truth"] != initial["ground_truth"]
        assert http.post(uri, json={"operation": "save", "revision": 1}).json()["revision"] == 2
        reset = http.post(uri, json={"operation": "reset", "revision": 2}).json()
        assert reset["ground_truth"] == initial["ground_truth"]
    with client(tmp_path) as http:
        restored = http.post(uri, json={"operation": "restore", "revision": 3}).json()
        assert restored["revision"] == 4 and restored["ground_truth"] == moved["ground_truth"]
        assert (
            restored["observation"] == moved["observation"] and restored["image"] == moved["image"]
        )


def test_read_only_trial_ledger_exposes_four_separate_verified_lanes(tmp_path):
    app = create_app(
        tmp_path,
        EXAMPLES,
        trial_evidence_root=REPOSITORY,
        trial_evidence_ledger=TRIAL_LEDGER,
    )
    with TestClient(app, base_url=ORIGIN) as http:
        result = http.get("/api/trial-evidence").json()
        assert result["configured"] is True and result["verified"] is True
        assert set(result["evidence_lanes"]) == {
            "real_recording",
            "synthetic_mechanics",
            "role_binding",
            "ammunition",
        }
        assert result["evidence_lanes"]["real_recording"]["raw_pixels_reverified"] is False
        assert result["evidence_lanes"]["synthetic_mechanics"]["hit_segments"] == 8
        assert result["evidence_lanes"]["role_binding"]["wrong_profile_denials"] == 12
        assert result["evidence_lanes"]["ammunition"]["final"] == 192
        page = http.get("/").text
        assert "试用验收账本" in page and "trial-lanes" in page


def test_trial_ledger_requires_its_read_only_root(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        create_app(tmp_path, trial_evidence_ledger=TRIAL_LEDGER)


def test_local_validation_and_no_arbitrary_paths(tmp_path):
    with client(tmp_path, None) as http:
        for headers in [
            {"Host": "other.example"},
            {"Origin": "http://other.example"},
            {"Origin": "http://127.0.0.1:8989"},
        ]:
            assert http.post("/api/jobs", json={}, headers=headers).status_code == 403
        assert http.post("/api/jobs", content="{}").status_code == 415
        assert http.post("/api/jobs", json={"note": "x" * 5000}).status_code == 413
        for body in [
            dict(kind="run", job="assassin", seed=True),
            dict(kind="run", job="assassin", max_steps=2501),
            dict(kind="run", job="archer"),
            dict(kind="run", job="assassin", model="../x"),
            dict(kind="run", job="assassin", output="/tmp/x"),
            dict(kind="replay", source="../run"),
        ]:
            assert http.post("/api/jobs", json=body).status_code == 422
        assert http.get("/api/jobs/" + 32 * "a").status_code == 404
        assert http.get("/static/owner.sqlite").status_code == 404
        assert http.get("/api/jobs").json() == []


def test_read_only_recording_replay_and_reviewed_events(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (12, 8), "#244f3e").save(source / "0000.png")
    Image.new("RGB", (12, 8), "#c67242").save(source / "0001.png")
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    session = recordings / "window-zero"
    imported = import_recording(
        source,
        session,
        source_uri="urn:self:workbench-recording",
        permission="self_authored",
        license="CC0-1.0",
        fps=20,
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    review_events(
        session,
        reports / "visible.json",
        reviewer="fixture-reviewer",
        events=[("attack_motion_visible", 0, 1)],
    )
    app = create_app(
        tmp_path / "workspace",
        recordings=recordings,
        recording_events=reports,
    )
    with TestClient(app, base_url=ORIGIN) as http:
        page = http.get("/").text
        assert "真实录像复核" in page and "recording-range" in page
        assert "recording-event-prev" in page and "recording-event-next" in page
        listed = http.get("/api/recordings").json()
        assert len(listed) == 1
        assert listed[0]["session_id"] == imported["session_id"]
        assert listed[0]["validated"] is True and listed[0]["event_reports"] == 1
        assert "root" not in listed[0] and "frames" not in listed[0]
        identity = listed[0]["session_id"]
        detail = http.get(f"/api/recordings/{identity}").json()
        assert [frame["timestamp_ms"] for frame in detail["frames"]] == [0, 50]
        assert detail["reviewed_event_reports"][0]["verification"] == {
            "verified": True,
            "scope": "source_frame_identity_and_reviewed_visible_intervals",
            "session_id": identity,
            "events": 1,
        }
        assert http.get(f"/api/recordings/{identity}/frames/1").content[:8] == b"\x89PNG\r\n\x1a\n"
        assert http.get(f"/api/recordings/{identity}/frames/2").status_code == 404
        assert http.get("/api/recordings/../manifest.json").status_code in {404, 422}
        (session / "frames/000001.png").write_bytes(b"changed")
        assert http.get(f"/api/recordings/{identity}/frames/1").status_code == 422


def test_recording_replay_rejects_unfixed_or_symlink_roots(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    with pytest.raises(ValueError, match="require a recordings directory"):
        with TestClient(
            create_app(tmp_path / "workspace", recording_events=reports),
            base_url=ORIGIN,
        ):
            pass
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (recordings / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="only real session directories"):
        with TestClient(
            create_app(tmp_path / "workspace-two", recordings=recordings),
            base_url=ORIGIN,
        ):
            pass


def test_contiguous_recording_coverage_replay_crosses_window_boundary(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    first = exact_recording_window(recordings, "first", 0, [100, 140], truncated=True)
    final = exact_recording_window(recordings, "final", 2, [220, 360], truncated=False)
    reports = tmp_path / "events"
    reports.mkdir()
    review_events(
        first,
        reports / "first.json",
        reviewer="fixture-reviewer",
        events=[("rope_contact_visible", 1, 1)],
    )
    review_events(
        final,
        reports / "final.json",
        reviewer="fixture-reviewer",
        events=[("attack_motion_visible", 0, 1)],
    )
    coverage = tmp_path / "coverage.json"
    report = summarize_recording_windows([final, first], coverage)
    with TestClient(
        create_app(
            tmp_path / "workspace",
            recordings=recordings,
            recording_events=reports,
            recording_coverages=[coverage],
        ),
        base_url=ORIGIN,
    ) as http:
        listed = http.get("/api/recording-coverages").json()
        assert listed == [
            {
                "coverage_sha256": report["coverage_sha256"],
                "source_name": "fixture.mp4",
                "source_sha256": "a" * 64,
                "permission": "self_authored",
                "license": "CC0-1.0",
                "window_count": 2,
                "decoded_frame_count": 4,
                "decoded_source_range": [0, 3],
                "contiguous": True,
                "decoder_stop_reached": True,
                "timing_accuracy": "source_pts",
                "event_reports": 2,
                "reviewed_events": 2,
                "verified": True,
            }
        ]
        identity = report["coverage_sha256"]
        detail = http.get(f"/api/recording-coverages/{identity}").json()
        assert [frame["source_sequence"] for frame in detail["frames"]] == [0, 1, 2, 3]
        assert [frame["timestamp_ms"] for frame in detail["frames"]] == [0, 40, 120, 260]
        assert [event["start_source_sequence"] for event in detail["reviewed_visible_events"]] == [
            1,
            2,
        ]
        assert http.get(f"/api/recording-coverages/{identity}/frames/1").status_code == 200
        boundary = http.get(f"/api/recording-coverages/{identity}/frames/2")
        assert boundary.status_code == 200 and boundary.content.startswith(b"\x89PNG")
        assert http.get(f"/api/recording-coverages/{identity}/frames/4").status_code == 404

    final.rename(tmp_path / "unregistered-final")
    with pytest.raises(ValueError, match="unregistered session"):
        with TestClient(
            create_app(
                tmp_path / "workspace-missing",
                recordings=recordings,
                recording_coverages=[coverage],
            ),
            base_url=ORIGIN,
        ):
            pass


def test_queue_cancel_interruption_and_owner_exclusion(tmp_path, monkeypatch):
    worker_entered = threading.Event()
    release = threading.Event()
    original = Workbench.execute

    def held(self, identity):
        worker_entered.set()
        assert release.wait(10)
        original(self, identity)

    monkeypatch.setattr(Workbench, "execute", held)
    with client(tmp_path, None) as http:
        try:
            with pytest.raises(sqlite3.OperationalError):
                Workbench(tmp_path)
            body = {"kind": "run", "job": "assassin", "max_steps": 1}
            ids = [submit(http, body) for _ in range(4)]
            assert worker_entered.is_set()
            assert http.post("/api/jobs", json=body).status_code == 409
            for rid in ids:
                assert http.post("/api/jobs/" + rid + "/cancel", json={}).status_code == 200
            assert http.post("/api/jobs/" + ids[0] + "/cancel", json={}).status_code == 409
        finally:
            release.set()
    with sqlite3.connect(tmp_path / "sessions.sqlite") as db:
        db.execute("UPDATE jobs SET state='running' WHERE id=?", (ids[1],))
    with client(tmp_path, None) as http:
        assert http.get("/api/jobs/" + ids[1]).json()["state"] == "interrupted"
        assert http.get("/api/jobs/" + ids[0]).json()["state"] == "cancelled"


def test_workbench_mission_selection_and_bound_evidence(tmp_path):
    with client(tmp_path) as http:
        content = http.get("/api/content").json()
        assert "talk-then-clear" in content["missions"]
        invalid = http.post(
            "/api/jobs",
            json={
                "kind": "run",
                "job": "assassin",
                "map_id": "three-map-fixture",
                "mission": "talk-then-clear",
            },
        )
        assert invalid.status_code == 422 and "prefix" in invalid.json()["detail"]
        assert not http.get("/api/jobs").json()
        rid = submit(
            http,
            {
                "kind": "run",
                "job": "archer",
                "map_id": "three-map-fixture",
                "mission": "three-map-visits",
            },
        )
        result = wait(http, rid)["result"]
        assert result["status"] == "completed"
        assert [(r["map_id"], r["phase"]) for r in result["mission_evidence"]["witnesses"]] == [
            ("grove", "talk"),
            ("quarry", "fight"),
            ("quarry", "talk"),
            ("ruins", "talk"),
        ]
        reinfer = wait(http, submit(http, {"kind": "reinfer", "source": rid}))
        assert reinfer["outcome"] == "verified"
    with client(tmp_path) as http:
        restored = http.get(f"/api/jobs/{rid}").json()
        assert restored["config"]["mission_program"] == content["missions"]["three-map-visits"]


def test_workbench_world_routes(tmp_path):
    with client(tmp_path) as http:
        graphs = http.get("/api/content").json()["world_graphs"]
        assert graphs["cms_east_hill"]["scope"] == "game_reference"

        def lookup(graph, source, target):
            return http.get(
                f"/api/world-graphs/{graph}/route", params={"source": source, "target": target}
            )

        route = lookup("cms_east_hill", "henesys", "mushroom_forest").json()["route"]
        assert [e["target"] for e in route["portals"]] == [
            "east_hill",
            "east_grass",
            "east_forest",
            "mushroom_forest",
        ]
        assert route["total_cost"] == 4
        for source, target in (
            ("dead_forest_1", "dead_forest_2"),
            ("dead_forest_2", "dead_forest_1"),
        ):
            result = lookup("cms_dead_forest", source, target).json()
            assert result["scope"] == "game_reference"
            assert result["route"]["portals"] == [
                {"source": source, "target": target, "portal_id": "to_" + target, "cost": 1}
            ]

        assert lookup("cms_east_hill", "henesys", "henesys").json()["route"]["portals"] == []
        assert lookup("branch_fixture", "camp", "island").json()["route"] is None
        assert lookup("branch_fixture", "camp", "missing").status_code == 422
        assert lookup("missing", "camp", "island").status_code == 404
    with client(tmp_path, examples=None) as http:
        assert http.get("/api/content").json()["world_graphs"] == {}


def test_key_steps_rebind_unbind_rejection_and_audit(tmp_path):
    with client(tmp_path) as http:
        session = http.post("/api/sessions", json={"job": "dragon_knight"}).json()
        base = "/api/sessions/" + session["id"]

        def press(version, key, bindings, **extra):
            return http.post(
                base + "/key-step",
                json={"revision": version, "key": key, "bindings": bindings, **extra},
            )

        used = press(0, "a", {"A": "模拟范围攻击"}, profile_id="测试区 / 一号").json()
        assert used["receipt"]["accepted"] and "attack" in used["receipt"]["events"]
        old = press(1, "a", {"S": "模拟范围攻击"}).json()
        assert old["receipt"]["reason"] == "unbound_key" and old["sequence"] == 1
        assert http.post(base, json={"revision": 2, "operation": "reset"}).status_code == 200
        assert press(3, "s", {"S": "模拟范围攻击"}).json()["receipt"]["accepted"]
        assert press(4, "s", {}).json()["receipt"]["reason"] == "unbound_key"
        assert press(5, "s", {"S": "未实现技能"}).json()["receipt"]["reason"] == "unsupported_skill"
        assert press(5, "s", {}).status_code == 409
    with client(tmp_path) as http:
        logs = http.get(base + "/key-log").json()
        assert [r["profile_id"] for r in logs] == [None, None, None, None, "测试区 / 一号"]
        assert [r["revision"] for r in logs] == [6, 5, 4, 2, 1]


def test_key_log_upgrades_existing_workspace_without_attributing_old_steps(tmp_path):
    with client(tmp_path) as http:
        identity = http.post("/api/sessions", json={"job": "dragon_knight"}).json()["id"]
        base = "/api/sessions/" + identity
        press = {"revision": 0, "key": "A", "bindings": {"A": "模拟范围攻击"}}
        assert http.post(base + "/key-step", json=press).status_code == 200
    with sqlite3.connect(tmp_path / "sessions.sqlite") as db:
        db.execute("ALTER TABLE key_steps DROP COLUMN profile_id")
    db.close()
    with client(tmp_path) as http:
        assert http.get(base + "/key-log").json()[0]["profile_id"] is None
        press.update(revision=1, profile_id="fixture")
        assert http.post(base + "/key-step", json=press).status_code == 200
        assert [row["profile_id"] for row in http.get(base + "/key-log").json()] == [
            "fixture",
            None,
        ]


def test_task_run_binds_confirmed_profile_and_exposes_dispatch_evidence(tmp_path):
    input_profile = {
        "schema_version": 1,
        "profile_id": "小白兔 / 大美人乖乖",
        "bindings": [{"key": "A", "skill": "模拟范围攻击"}],
    }
    # This synthetic UTF-8 layout fits the editor while its JSON exceeds 4 KiB.
    input_profile["bindings"].extend(
        {"skill": f"测试角色已确认的技能编号{index}", "key": f"K{index}"} for index in range(100)
    )
    with client(tmp_path) as http:
        assert http.post("/api/input-profile", json=input_profile).json() == input_profile
        confirmation = http.post(
            "/api/input-profile/confirmation",
            json={
                "schema_version": 1,
                "profile": input_profile,
                "comparison_source": "live_window_manual_comparison",
                "source_image": None,
                "user_confirmed": True,
                "automatic_skill_recognition": False,
            },
        ).json()
        session = http.post("/api/sessions", json={"job": "dragon_knight"}).json()
        manual = http.post(
            f"/api/sessions/{session['id']}/key-step",
            json={
                "revision": 0,
                "profile_id": input_profile["profile_id"],
                "key": "A",
                "bindings": {entry["key"]: entry["skill"] for entry in input_profile["bindings"]},
            },
        )
        assert len(manual.request.content) > 4096
        assert manual.json()["receipt"]["accepted"]
        duplicate = {
            **input_profile,
            "bindings": [
                {"key": "Control", "skill": "普通攻击"},
                {"key": "ctrl", "skill": "模拟范围攻击"},
            ],
        }
        rejected = http.post(
            "/api/jobs",
            json={
                "kind": "run",
                "job": "dragon_knight",
                "map_id": "dead-forest-drill",
                "mission": "dead-forest-hunt",
                "input_profile": duplicate,
            },
        )
        assert rejected.status_code == 422 and "duplicate normalized keys" in rejected.text
        identity = submit(
            http,
            {
                "kind": "run",
                "job": "dragon_knight",
                "map_id": "dead-forest-drill",
                "mission": "dead-forest-hunt",
                "input_profile": input_profile,
                "input_confirmation": confirmation,
            },
        )
        run = wait(http, identity)
        assert run["outcome"] == "completed"
        assert run["config"]["input_profile"] == input_profile
        assert run["config"]["input_confirmation"] == confirmation
        summary = json.loads((tmp_path / identity / "run/summary.json").read_text(encoding="utf-8"))
        assert summary["input_confirmation"]["receipt_sha256"] == confirmation["receipt_sha256"]
        rows = http.get(f"/api/jobs/{identity}/trace").json()
        assert all("input_dispatch" in row["execution"] for row in rows)
        reinfer = wait(http, submit(http, {"kind": "reinfer", "source": identity}))
        assert reinfer["outcome"] == "verified"


def test_input_profile_export_uses_execution_validation(tmp_path):
    profile = {
        "schema_version": 1,
        "profile_id": "fixture",
        "bindings": [{"key": "Ctrl", "skill": "basic"}],
    }
    with client(tmp_path) as http:
        reply = http.post("/api/input-profile", json=profile)
        assert reply.status_code == 200 and reply.json() == profile
        profile["bindings"].append({"key": "Control", "skill": "area"})
        assert http.post("/api/input-profile", json=profile).status_code == 422
        assert http.get("/api/jobs").json() == []


def test_input_profile_confirmation_validates_manual_source_and_image_digest(tmp_path):
    profile = {
        "schema_version": 1,
        "profile_id": "小白兔 / 无影人",
        "bindings": [{"key": "J", "skill": "lucky_seven"}],
    }
    request = {
        "schema_version": 1,
        "profile": profile,
        "comparison_source": "browser_selected_image_digest",
        "source_image": {
            "sha256": "b" * 64,
            "bytes": 2048,
            "media_type": "image/png",
            "width": 1536,
            "height": 864,
        },
        "user_confirmed": True,
        "automatic_skill_recognition": False,
    }
    with client(tmp_path) as http:
        reply = http.post("/api/input-profile/confirmation", json=request)
        assert reply.status_code == 200
        receipt = reply.json()
        assert receipt["profile"] == profile
        assert receipt["comparison_source"] == "browser_selected_image_digest"
        assert receipt["source_image"]["image_bytes_uploaded"] is False
        assert receipt["recognition"]["automatic_skill_recognition"] is False
        assert len(receipt["receipt_sha256"]) == 64
        review = {
            "schema_version": 1,
            "row": "ASDFGHJKL",
            "anchors": [
                {"key": "A", "x": 641, "y": 430},
                {"key": "L", "x": 831, "y": 430},
            ],
            "algorithm_candidate_keys": ["A", "S", "D", "G"],
            "confirmed_content_keys": ["A", "S", "D", "F"],
            "user_reviewed": True,
        }
        reviewed_reply = http.post(
            "/api/input-profile/confirmation",
            json={**request, "schema_version": 2, "key_grid_review": review},
        )
        assert reviewed_reply.status_code == 200
        reviewed = reviewed_reply.json()
        assert reviewed["schema_version"] == 2
        assert reviewed["key_grid_review"]["player_review"]["added_keys"] == ["F"]
        assert reviewed["key_grid_review"]["player_review"]["removed_keys"] == ["G"]

        live = {
            **request,
            "comparison_source": "live_window_manual_comparison",
            "source_image": None,
        }
        live_receipt = http.post("/api/input-profile/confirmation", json=live)
        assert live_receipt.status_code == 200
        assert live_receipt.json()["source_image"] is None
        large_profile = {
            **profile,
            "bindings": [{"skill": f"示例技能名称较长{i:03}", "key": f"K{i}"} for i in range(120)],
        }
        assert http.post("/api/input-profile", json=large_profile).status_code == 200
        assert (
            http.post(
                "/api/input-profile/confirmation", json={**live, "profile": large_profile}
            ).status_code
            == 200
        )

        mismatched = {**request, "comparison_source": "live_window_manual_comparison"}
        assert http.post("/api/input-profile/confirmation", json=mismatched).status_code == 422
        claimed = {**request, "automatic_skill_recognition": True}
        assert http.post("/api/input-profile/confirmation", json=claimed).status_code == 422
        missing_review = {**request, "schema_version": 2}
        assert http.post("/api/input-profile/confirmation", json=missing_review).status_code == 422
        huge = {
            **request,
            "source_image": {**request["source_image"], "width": 8192, "height": 8192},
        }
        assert http.post("/api/input-profile/confirmation", json=huge).status_code == 422
        assert http.get("/api/jobs").json() == []
