"""Exercise the installed workbench through Chromium against its real HTTP server."""

import argparse
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from PIL import Image
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
        page.select_option("#run-kind", "run")
        page.select_option("#map-select", "builtin")
        page.select_option("#class-select", "assassin")
        page.select_option("#fault-select", "none")
        page.select_option("#mission-select", "talk-then-clear")
        page.click("#submit-run")
        expect(page.locator("#detail-status")).to_have_text("目标完成", timeout=30000)
        page.click('[data-evidence="summary"]')
        expect(page.locator("#evidence-json")).to_contain_text("mission_evidence")
        page.reload()
        expect(page.locator("#connection-label")).to_have_text("本机已连接")
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
        page.click('[data-view="routes"]')
        page.select_option("#route-graph", "cms_east_hill")
        page.select_option("#route-source", "henesys")
        page.select_option("#route-target", "mushroom_forest")
        page.locator("#route-form button").click()
        expect(page.locator("#route-steps li")).to_have_count(4)
        expect(page.locator("#route-steps li").last).to_contain_text("青蘑菇树林")
        page.click('[data-view="keys"]')
        page.fill("#keys-character", "示例区 / 一号")
        screenshot = io.BytesIO()
        Image.new("RGB", (32, 16), "navy").save(screenshot, format="PNG")
        page.locator("#keys-image-file").set_input_files(
            {"name": "fixture.png", "mimeType": "image/png", "buffer": screenshot.getvalue()}
        )
        expect(page.locator("#keys-image")).to_have_js_property("naturalWidth", 32)
        page.fill("#keys-list", "示例技能 = Ctrl")
        page.locator("#keys-form button").click()
        expect(page.locator("#keys-status")).to_contain_text("保存 1 项")
        page.fill("#keys-list", "示例技能 = Ctrl\n跳跃 = CTRL")
        page.locator("#keys-form button").click()
        expect(page.locator("#keys-status")).to_contain_text("出现了两次")
        page.fill("#keys-character", "示例区 / 二号")
        expect(page.locator("#keys-image")).to_be_hidden()
        assert page.locator("#keys-image").get_attribute("src") is None
        expect(page.locator("#keys-list")).to_have_value("")
        page.fill("#keys-character", "示例区 / 一号")
        expect(page.locator("#keys-list")).to_have_value("示例技能 = Ctrl")
        page.fill("#keys-list", "模拟范围攻击 = A")
        page.locator("#keys-form button").click()
        page.click('[data-view="runs"]')
        page.select_option("#run-kind", "run")
        page.select_option("#class-select", "dragon_knight")
        page.select_option("#map-select", "dead-forest-drill")
        page.select_option("#mission-select", "dead-forest-hunt")
        page.fill("#run-profile", "示例区 / 一号")
        page.click("#submit-run")
        expect(page.locator("#detail-status")).to_have_text("目标完成", timeout=30000)
        expect(page.locator("#evidence-json")).to_contain_text("input_dispatch_summary")
        page.click('[data-view="simulator"]')
        page.select_option("#sim-class", "dragon_knight")
        page.click("#sim-create")
        expect(page.locator("#sim-clock")).to_contain_text("第 0 帧")
        page.fill("#sim-profile", "示例区 / 一号")
        page.click("#sim-key")
        page.locator("#sim-key").press("a")
        expect(page.locator("#sim-key-status")).to_contain_text("已执行")
        page.click('[data-view="keys"]')
        page.fill("#keys-list", "模拟范围攻击 = S")
        page.locator("#keys-form button").click()
        page.click('[data-view="simulator"]')
        page.click("#sim-key")
        page.locator("#sim-key").press("a")
        expect(page.locator("#sim-key-status")).to_contain_text("unbound_key")
        page.click("#sim-reset")
        expect(page.locator("#sim-clock")).to_contain_text("第 0 帧")
        page.click("#sim-key")
        page.locator("#sim-key").press("s")
        expect(page.locator("#sim-key-status")).to_contain_text("已执行")
        page.click("#sim-key-history")
        expect(page.locator("#sim-key-log")).to_contain_text("s → 模拟范围攻击")
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
            "custom_mission_outcome": "completed",
            "evidence_operations": ["replay", "reinfer", "export_bundle", "verify_bundle"],
            "single_step_restore_after_reload": True,
            "keyed_two_map_task": "completed_and_evidenced",
            "local_key_image_preview_and_clear": True,
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
