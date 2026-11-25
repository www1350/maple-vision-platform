"""Exercise the installed workbench through Chromium against its real HTTP server."""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def verify(url: str, executable: str) -> dict[str, object]:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator("#connection-label")).to_have_text("本机已连接")
        page.select_option("#class-select", "archer")
        page.select_option("#map-select", "three-map-fixture")
        page.select_option("#fault-select", "persistent_dropout")
        page.click("#submit-run")
        expect(page.locator("#detail-status")).to_have_text("已停止", timeout=30000)
        expect(page.locator("#replay-frame")).to_be_visible()
        page.click("#frame-next")
        page.click('[data-evidence="trace"]')
        expect(page.locator("#trace-page")).to_have_text("1–4")
        page.click("#trace-next")
        expect(page.locator("#trace-page")).to_have_text("5–8")
        for action in ["replay", "reinfer", "export_bundle", "verify_bundle"]:
            page.select_option("#run-kind", action)
            expect(page.locator("#submit-run")).to_be_enabled()
            page.click("#submit-run")
            expect(page.locator("#detail-heading")).to_contain_text(
                {
                    "replay": "动作重放",
                    "reinfer": "感知重推理",
                    "export_bundle": "导出分层包",
                    "verify_bundle": "验证分层包",
                }[action]
            )
            expect(page.locator("#detail-status")).to_have_text("校验完成", timeout=30000)
        page.click('[data-view="simulator"]')
        page.select_option("#sim-class", "bandit")
        page.click("#sim-create")
        expect(page.locator("#sim-clock")).to_contain_text("第 0 帧")
        page.select_option("#sim-action", "move")
        page.click("#sim-step")
        expect(page.locator("#sim-clock")).to_contain_text("第 1 帧")
        page.click("#sim-save")
        expect(page.locator("#sim-outcome")).to_have_text("已保存当前状态")
        page.click("#sim-reset")
        expect(page.locator("#sim-clock")).to_contain_text("第 0 帧")
        page.click("#sim-restore")
        expect(page.locator("#sim-clock")).to_contain_text("第 1 帧")
        page.reload()
        expect(page.locator("#sim-clock")).to_contain_text("第 1 帧")
        page.click('[data-view="content"]')
        expect(page.locator("#content-cards")).to_contain_text("arrow_shot")
        page.set_viewport_size({"width": 390, "height": 844})
        page.click('[data-view="runs"]')
        expect(page.locator("#view-runs")).to_be_visible()
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        assert not overflow and not errors, (overflow, errors)
        result = {
            "chromium": browser.version,
            "run_outcome": "stopped",
            "evidence_operations": ["replay", "reinfer", "export_bundle", "verify_bundle"],
            "single_step_restore_after_reload": True,
            "mobile_width": 390,
            "horizontal_overflow": overflow,
            "page_errors": errors,
        }
        browser.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser", default=shutil.which("google-chrome") or shutil.which("chromium")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.browser:
        parser.error("a local Chromium or Chrome executable is required")
    with tempfile.TemporaryDirectory(prefix="maple-browser-") as folder:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "maple_vision",
                "workbench",
                "--workspace",
                folder,
                "--examples",
                str(Path(__file__).resolve().parents[1] / "examples"),
                "--port",
                str(port),
            ]
        ) as service:
            try:
                for _ in range(150):
                    if service.poll() is not None:
                        raise RuntimeError("workbench exited before accepting requests")
                    try:
                        with urlopen(url + "/api/content", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("workbench did not start")
                report = verify(url, args.browser)
            finally:
                service.terminate()
                service.wait(timeout=30)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
