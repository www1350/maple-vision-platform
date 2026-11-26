"""Exercise the actual toolkit on the Windows job and retain its first-use replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from maple_vision.owned_evidence import verify_owned_window_evidence


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows desktop acceptance")
def test_windows_first_use_replay(tmp_path):
    root = Path(__file__).resolve().parents[1]
    parent = root / "artifacts" if os.environ.get("GITHUB_ACTIONS") == "true" else tmp_path
    output = parent / "owned-window-first-use"
    subprocess.run(
        [sys.executable, "scripts/verify_owned_window.py", "--output", str(output)],
        cwd=root,
        check=True,
        timeout=60,
    )
    assert (output / "evidence.json").is_file()
    assert (output / "replay.gif").stat().st_size > 0
    assert verify_owned_window_evidence(output) == {
        "verified": True,
        "scope": "project_owned_native_tk_window",
        "manifest_sha256": json.loads((output / "evidence.json").read_text(encoding="utf-8"))[
            "integrity"
        ]["manifest_sha256"],
        "media_files": 7,
        "replay_frames": 5,
    }
