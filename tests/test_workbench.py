import base64
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from maple_vision.workbench import Workbench
from maple_vision.workbench_api import create_app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ORIGIN = "http://127.0.0.1"


def client(root, examples=EXAMPLES):
    return TestClient(create_app(root, examples), base_url=ORIGIN)


def submit(http, body):
    reply = http.post("/api/jobs", json=body)
    assert reply.status_code == 202, reply.text
    return reply.json()["id"]


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
