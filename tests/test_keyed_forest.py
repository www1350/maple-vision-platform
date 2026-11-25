"""Exercise the workbench key path against an independently verified two-map run."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from maple_vision.lab import RunConfig, run_episode, verify_run
from maple_vision.map_pack import load_map_pack
from maple_vision.mission_program import load_mission_program
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.workbench_api import create_app


def test_keyed_attacks_reproduce_two_map_combat(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    pack = load_map_pack(examples / "maps/dead-forest-drill.json")
    mission = load_mission_program(examples / "missions/dead-forest-hunt.json")
    original = tmp_path / "original"
    report = run_episode(
        original,
        RunConfig(job="dragon_knight"),
        map_pack=pack,
        runtime=RuntimeConfig(mission=mission),
    )
    assert report["mission_evidence"]["completed"] and verify_run(original)["verified"]
    rows = [json.loads(line) for line in (original / "trace.jsonl").read_text().splitlines()]
    with TestClient(
        create_app(tmp_path / "workbench", examples), base_url="http://127.0.0.1"
    ) as http:
        state = http.post("/api/sessions", json={"job": "dragon_knight", "map_id": pack.id}).json()
        uri = "/api/sessions/" + state["id"]
        hits = []
        kills = []
        for row in rows:
            action = row["execution"]["applied_action"]
            source = state["observation"]["map_id"]
            assert source == row["observation"]["map_id"]
            if action["kind"] == "attack":
                assert action["skill"] == "area"
                key = "a" if source == mission.maps[0] else "s"
                bindings = {key: "模拟范围攻击"}
                if key == "s":
                    denied = http.post(
                        uri + "/key-step",
                        json={"key": "a", "bindings": bindings, "revision": state["revision"]},
                    ).json()
                    assert denied["receipt"]["reason"] == "unbound_key"
                    assert denied["sequence"] == state["sequence"]
                    state = denied
                response = http.post(
                    uri + "/key-step",
                    json={"key": key, "bindings": bindings, "revision": state["revision"]},
                )
            else:
                response = http.post(
                    uri, json={"operation": "step", "action": action, "revision": state["revision"]}
                )
            assert response.status_code == 200, response.text
            state = response.json()
            receipt = state["receipt"]
            assert receipt["accepted"] == row["execution"]["accepted"]
            assert receipt["events"] == row["execution"]["events"]
            if "attack_hit" in receipt["events"]:
                hits.append(source)
            if "monster_defeated" in receipt["events"]:
                kills.append(source)
        assert state["sequence"] == report["steps"] and state["ground_truth"]["player"]["alive"]
        assert hits == [mission.maps[0]] * 2 + [mission.maps[1]] * 2
        assert kills == list(mission.maps)
        log = http.get(uri + "/key-log").json()
        assert [r["key_name"] for r in reversed(log) if r["accepted"]] == ["a", "a", "s", "s"]
