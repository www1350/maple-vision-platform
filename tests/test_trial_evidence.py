import json
from pathlib import Path

import pytest

from maple_vision.cli import main
from maple_vision.trial_evidence import verify_trial_evidence

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "reports/release-0.10.0/throwing-star-trial-evidence-ledger.json"


def test_throwing_star_trial_evidence_rechecks_all_four_separate_lanes(capsys):
    assert main(["lab", "verify-trial-evidence", str(LEDGER), "--root", str(ROOT)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)

    assert report["verified"] is True
    assert report["evidence_lanes"]["real_recording"] == {
        "persistent_reports_verified": True,
        "raw_pixels_reverified": False,
        "coverage_sha256": "0b2c6165e350838cce0c00263cdabb9f01efeff57aee5fc74ceea6e2c402c21f",
        "frames": 2197,
        "reviewed_visible_events": 6,
    }
    assert report["evidence_lanes"]["synthetic_mechanics"]["steps"] == 266
    assert report["evidence_lanes"]["synthetic_mechanics"]["hit_segments"] == 8
    assert report["evidence_lanes"]["synthetic_mechanics"]["replay_verified"] is True
    assert report["evidence_lanes"]["synthetic_mechanics"]["reinference_verified"] is True
    assert report["evidence_lanes"]["role_binding"]["accepted_dispatches"] == 4
    assert report["evidence_lanes"]["role_binding"]["wrong_profile_denials"] == 12
    assert report["evidence_lanes"]["ammunition"] == {
        "scope": "synthetic_fixture_rule",
        "initial": 200,
        "per_cast": 2,
        "successful_casts": 4,
        "final": 192,
    }


def test_trial_evidence_rejects_paths_outside_its_root(tmp_path):
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ledger["evidence"]["real_recording"]["coverage"]["path"] = "../outside.json"
    target = tmp_path / "ledger.json"
    target.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid recording coverage report path"):
        verify_trial_evidence(target, tmp_path)
