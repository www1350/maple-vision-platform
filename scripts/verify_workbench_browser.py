"""Exercise the installed workbench through Chromium against its real HTTP server."""

import argparse
import hashlib
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

from PIL import Image, ImageDraw
from playwright.sync_api import Page, Route, expect, sync_playwright

from maple_vision.catalog import canonical_json
from maple_vision.input_bindings import load_input_profile
from maple_vision.recording_evaluation import review_events, summarize_recording_windows
from maple_vision.scene import ImageFrame


def recording_window(root: Path, name: str, start: int, pts: list[int], truncated: bool) -> Path:
    session = root / name
    frames = session / "frames"
    frames.mkdir(parents=True)
    rows = []
    for sequence, timestamp in enumerate(pts):
        picture = Image.new("RGB", (160, 90), (50 + 20 * (start + sequence), 80, 120))
        drawing = ImageDraw.Draw(picture)
        # Independent visible figures; the browser selects a region, not a model prediction.
        drawing.rectangle((34, 24, 41, 33), fill=(50, 210, 240))
        drawing.rectangle((32, 34, 43, 51), fill=(50, 210, 240))
        drawing.rectangle((111, 20, 118, 31), fill=(240, 90, 180))
        drawing.rectangle((108, 32, 121, 51), fill=(240, 90, 180))
        pixels = picture.tobytes()
        frame = ImageFrame(sequence, timestamp - pts[0], 160, 90, pixels)
        data = frame.png_bytes()
        file = f"frames/{sequence:06d}.png"
        (session / file).write_bytes(data)
        rows.append(
            {
                "sequence": sequence,
                "timestamp_ms": timestamp - pts[0],
                "file": file,
                "sha256": hashlib.sha256(data).hexdigest(),
                "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "source_name": f"video-frame-{start + sequence:06d}",
                "source_sha256": None,
                "width": 160,
                "height": 90,
                "pts": timestamp,
            }
        )
    manifest = {
        "schema_version": 2 if start == 0 else 3,
        "kind": "offline_recording",
        "scope": "user_supplied_recording",
        "source_kind": "mp4",
        "source_uri": "urn:self:browser-workbench",
        "permission": "self_authored",
        "license": "CC0-1.0",
        "source_name": "browser-fixture.mp4",
        "source_sha256": "b" * 64,
        "source_hash_basis": "mp4_file",
        "fps": None,
        "timebase": "video_presentation_ticks",
        "frame_count": len(rows),
        "width": 160,
        "height": 90,
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


def actor_image_geometry(page: Page) -> dict[str, float]:
    """Measure rendered pixels independently of the application's selection code."""
    return page.locator("#recording-frame").evaluate(
        """image => {
          const bounds = image.getBoundingClientRect();
          const scale = Math.min(bounds.width / image.naturalWidth,
                                 bounds.height / image.naturalHeight);
          return {left: bounds.left + (bounds.width - image.naturalWidth * scale) / 2,
                  top: bounds.top + (bounds.height - image.naturalHeight * scale) / 2,
                  scale, width: image.naturalWidth * scale,
                  height: image.naturalHeight * scale};
        }"""
    )


def draw_actor_box(page: Page, box: list[int], *, reverse: bool = False) -> None:
    expect(page.locator("#recording-target-start")).to_be_enabled()
    if page.locator("#recording-target-clear").is_enabled():
        page.click("#recording-target-clear")
    page.click("#recording-target-start")
    viewport = page.locator("#recording-frame-viewport")
    viewport.scroll_into_view_if_needed()
    geometry = actor_image_geometry(page)
    center = {
        "x": geometry["left"] + (box[0] + box[2]) / 2 * geometry["scale"],
        "y": geometry["top"] + (box[1] + box[3]) / 2 * geometry["scale"],
    }
    viewport.evaluate(
        """(element, center) => {
          const bounds = element.getBoundingClientRect();
          element.scrollBy(center.x - bounds.left - bounds.width / 2,
                           center.y - bounds.top - bounds.height / 2);
        }""",
        center,
    )
    geometry = actor_image_geometry(page)
    # Land inside the boundary pixels to avoid ambiguous fractional display edges.
    points = [(box[0] + 0.2, box[1] + 0.2), (box[2] - 0.2, box[3] - 0.2)]
    if reverse:
        points.reverse()
    first, last = [
        (
            geometry["left"] + x * geometry["scale"],
            geometry["top"] + y * geometry["scale"],
        )
        for x, y in points
    ]
    page.mouse.move(*first)
    page.mouse.down()
    page.mouse.move(*last, steps=5)
    page.mouse.up()
    expect(page.locator("#recording-target-preview")).to_be_visible()


def confirm_actor_box(page: Page) -> None:
    page.locator("#recording-target-reviewer").fill("independent browser review")
    page.locator("#recording-target-subject").fill("left cyan figure")
    expect(page.locator("#recording-target-export")).to_be_disabled()
    page.locator("#recording-target-confirm").check()
    expect(page.locator("#recording-target-export")).to_be_enabled()


def download_actor_box(page: Page) -> dict[str, object]:
    with page.expect_download() as downloaded:
        page.click("#recording-target-export")
    download = downloaded.value
    assert download.suggested_filename == "actor-selection.json"
    with tempfile.TemporaryDirectory(prefix="maple-actor-download-") as folder:
        path = Path(folder) / download.suggested_filename
        download.save_as(path)
        return json.loads(path.read_text(encoding="utf-8"))


def verify_actor_selection_ui(page: Page, url: str, screenshots: Path | None) -> None:
    """Exercise real pointer events, downloads, source clocks and stale responses."""
    expected_box = [32, 24, 44, 52]
    sessions = page.request.get(url + "/api/recordings").json()
    second = next(session for session in sessions if session.get("source_start_frame", 0) == 2)
    detail = page.request.get(url + "/api/recordings/" + second["session_id"]).json()
    draw_actor_box(page, expected_box)
    confirm_actor_box(page)
    if screenshots is not None:
        screenshots.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshots / "actor-selection-desktop.png"), full_page=True)
    selection = download_actor_box(page)
    assert selection["kind"] == "recording_actor_selection"
    assert selection["source"]["session_id"] == second["session_id"]
    assert selection["anchor"]["source_sequence"] == 3
    assert selection["anchor"]["timestamp_ms"] == 140  # Local clock, not coverage's 260 ms.
    assert selection["anchor"]["bbox"] == expected_box
    assert selection["anchor"]["rgb_sha256"] == detail["frames"][1]["rgb_sha256"]

    # Delay a previously requested image, then select a different frame before it arrives.
    pending_images: list[Route] = []
    pattern = "**/api/recording-coverages/*/frames/2"
    page.route(pattern, lambda route: pending_images.append(route))
    seek = page.locator("#recording-source-frame")
    seek.fill("2")
    seek.press("Enter")
    expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 2")
    expect(page.locator("#recording-target-start")).to_be_disabled()
    deadline = time.monotonic() + 5
    while not pending_images and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert pending_images, "frame request was not intercepted"
    seek.fill("3")
    seek.press("Enter")
    expect(page.locator("#recording-target-start")).to_be_enabled()
    draw_actor_box(page, expected_box)
    held_image = pending_images.pop()
    image_response = page.request.get(held_image.request.url)
    held_image.fulfill(response=image_response)
    page.wait_for_timeout(100)
    page.unroute(pattern)
    expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 3")
    assert page.locator("#recording-frame").evaluate(
        """image => {
          const canvas = document.createElement('canvas'); canvas.width=1; canvas.height=1;
          const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
          return Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3);
        }"""
    ) == [110, 80, 120]

    # Finish a real API response only after the user has changed the selected frame.
    pending_exports = []
    export_pattern = "**/api/recordings/*/actor-selection"

    def hold_export(route: Route) -> None:
        pending_exports.append((route, route.fetch()))

    page.route(export_pattern, hold_export)
    stale_downloads = []
    page.on("download", stale_downloads.append)
    confirm_actor_box(page)
    page.click("#recording-target-export")
    deadline = time.monotonic() + 5
    while not pending_exports and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert pending_exports, "selection request was not intercepted"
    seek.fill("2")
    seek.press("Enter")
    expect(page.locator("#recording-target-confirm")).not_to_be_checked()
    held_export, response = pending_exports.pop()
    assert response.status == 200
    held_export.fulfill(response=response)
    page.wait_for_timeout(150)
    assert not stale_downloads, "a stale actor selection was downloaded"
    page.remove_listener("download", stale_downloads.append)
    page.unroute(export_pattern)
    expect(page.locator("#recording-target-export")).to_be_disabled()

    # Repeat the exact source-pixel choice in a narrow viewport, in reverse drag order.
    page.set_viewport_size({"width": 390, "height": 844})
    draw_actor_box(page, expected_box, reverse=True)
    confirm_actor_box(page)
    if screenshots is not None:
        page.screenshot(path=str(screenshots / "actor-selection-mobile.png"), full_page=True)
    mobile_selection = download_actor_box(page)
    assert mobile_selection["anchor"]["bbox"] == expected_box
    assert mobile_selection["anchor"]["source_sequence"] == 2
    assert mobile_selection["anchor"]["timestamp_ms"] == 0
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    page.click("#recording-target-zoom")
    draw_actor_box(page, expected_box)
    confirm_actor_box(page)
    zoomed_selection = download_actor_box(page)
    assert zoomed_selection["anchor"]["bbox"] == expected_box
    assert zoomed_selection["anchor"]["rgb_sha256"] == detail["frames"][0]["rgb_sha256"]
    page.click("#recording-target-zoom")
    page.set_viewport_size({"width": 1280, "height": 850})


def verify(url: str, executable: str, screenshots: Path | None = None) -> dict[str, object]:
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
        page.click('[data-view="recordings"]')
        expect(page.locator("#recording-frame")).to_be_visible()
        expect(page.locator("#recording-evidence")).to_contain_text("attack_motion_visible")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 0")
        expect(page.locator("#recording-event-status")).to_contain_text("共 2 条")
        page.click("#recording-event-next")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 1")
        expect(page.locator("#recording-event-status")).to_contain_text("接绳可见")
        page.click("#recording-event-next")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 2")
        expect(page.locator("#recording-event-status")).to_contain_text("攻击动作可见")
        page.click("#recording-event-prev")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 1")
        seek = page.locator("#recording-source-frame")
        seek.fill("3")
        seek.press("Enter")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 3 · 窗口帧 1")
        try:
            verify_actor_selection_ui(page, url, screenshots)
        except Exception:
            if screenshots is not None:
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(
                    path=str(screenshots / "actor-selection-failure.png"), full_page=True
                )
            raise
        tail = (
            page.locator("#recording-select option")
            .filter(has_text="单窗口")
            .filter(has_text="2–3")
        )
        page.select_option("#recording-select", value=tail.get_attribute("value"))
        expect(seek).to_have_attribute("min", "2")
        seek.fill("3")
        seek.press("Enter")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 3 · 窗口帧 1")
        seek.fill("1")
        seek.press("Enter")
        assert not seek.evaluate("(field) => field.checkValidity()")
        expect(page.locator("#recording-frame-caption")).to_contain_text("源帧 3 · 窗口帧 1")
        page.click('[data-view="trial"]')
        expect(page.locator("#trial-status")).to_have_text("四栏已复验")
        expect(page.locator("#trial-lanes .content-group")).to_have_count(4)
        expect(page.locator("#trial-json")).to_contain_text('"raw_pixels_reverified": false')
        expect(page.locator("#trial-json")).to_contain_text('"wrong_profile_denials": 12')
        page.click('[data-view="runs"]')
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
        row_image = Image.new("RGB", (200, 32), "white")
        row_image.paste("navy", (82, 6, 98, 26))
        row_file = io.BytesIO()
        row_image.save(row_file, format="PNG")
        page.locator("#keys-image-file").set_input_files(
            {"name": "row.png", "mimeType": "image/png", "buffer": row_file.getvalue()}
        )
        page.click("#keys-row-start")
        image = page.locator("#keys-image")
        image.click(position={"x": 10, "y": 16})
        image.click(position={"x": 170, "y": 16})
        expect(page.locator(".key-review-cell")).to_have_count(9)
        expect(page.locator("#keys-row-status")).to_contain_text("深色内容候选 G。")
        page.locator(".key-review-cell").filter(has_text="G").click()
        page.locator(".key-review-cell").filter(has_text="F").click()
        page.click("#keys-row-confirm")
        expect(page.locator("#keys-row-status")).to_contain_text("九格复核已确认")
        page.fill("#keys-list", "示例技能 = F")
        page.locator("#keys-form button[type=submit]").click()
        expect(page.locator("#keys-status")).to_contain_text("保存 1 项")
        with page.expect_download() as reviewed_download_info:
            page.click("#keys-export-confirmation")
        with tempfile.TemporaryDirectory() as reviewed_dir:
            reviewed_path = Path(reviewed_dir) / "reviewed-confirmation.json"
            reviewed_download_info.value.save_as(reviewed_path)
            reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
            assert reviewed["schema_version"] == 2
            assert reviewed["key_grid_review"]["heuristic"]["candidate_keys"] == ["G"]
            assert reviewed["key_grid_review"]["player_review"]["content_keys"] == ["F"]
            assert reviewed["key_grid_review"]["player_review"]["added_keys"] == ["F"]
            assert reviewed["key_grid_review"]["player_review"]["removed_keys"] == ["G"]
            assert reviewed["recognition"]["automatic_skill_recognition"] is False
            assert reviewed["recognition"]["skill_candidates_generated"] is False
        screenshot = io.BytesIO()
        Image.new("RGB", (32, 16), "navy").save(screenshot, format="PNG")
        page.locator("#keys-image-file").set_input_files(
            {"name": "fixture.png", "mimeType": "image/png", "buffer": screenshot.getvalue()}
        )
        expect(page.locator("#keys-image")).to_have_js_property("naturalWidth", 32)
        expect(page.locator(".key-review-cell")).to_have_count(0)
        expect(page.locator("#keys-status")).to_contain_text("截图已在浏览器内计算哈希")
        page.check("#keys-detail")
        expect(page.locator("#keys-image")).to_have_css("max-width", "none")
        page.uncheck("#keys-detail")
        expect(page.locator("#keys-image")).to_have_css("max-width", "100%")
        page.fill("#keys-list", "示例技能 = Ctrl")
        page.locator("#keys-form button[type=submit]").click()
        expect(page.locator("#keys-status")).to_contain_text("保存 1 项")
        expect(page.locator("#keys-status")).to_contain_text("未自动识别技能")
        with page.expect_download() as confirmation_download_info:
            page.click("#keys-export-confirmation")
        confirmation_download = confirmation_download_info.value
        assert confirmation_download.suggested_filename == "input-binding-confirmation.json"
        with tempfile.TemporaryDirectory() as confirmation_dir:
            confirmation_path = Path(confirmation_dir) / confirmation_download.suggested_filename
            confirmation_download.save_as(confirmation_path)
            confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
            assert confirmation["profile"]["profile_id"] == "示例区 / 一号"
            assert confirmation["source_image"]["bytes"] == len(screenshot.getvalue())
            assert confirmation["source_image"]["width"] == 32
            assert confirmation["source_image"]["height"] == 16
            assert confirmation["source_image"]["image_bytes_uploaded"] is False
            assert confirmation["recognition"]["automatic_skill_recognition"] is False
        page.fill("#keys-list", "未确认技能 = X")
        with page.expect_download() as download_info:
            page.click("#keys-export")
        download = download_info.value
        assert download.suggested_filename == "input-bindings.json"
        with tempfile.TemporaryDirectory() as download_dir:
            exported = Path(download_dir) / download.suggested_filename
            download.save_as(exported)
            profile = load_input_profile(exported)
            assert profile.profile_id == "示例区 / 一号"
            assert [(item.skill, item.key) for item in profile.bindings] == [("示例技能", "Ctrl")]
            page.locator("#keys-import").set_input_files(exported)
            expect(page.locator("#keys-status")).to_contain_text("1 项草稿")
            expect(page.locator("#keys-list")).to_have_value("示例技能 = Ctrl")
            bad = {"name": "bad.json", "mimeType": "application/json", "buffer": b"invalid"}
            page.locator("#keys-import").set_input_files(bad)
            expect(page.locator("#keys-status")).to_contain_text("导入失败")
            expect(page.locator("#keys-list")).to_have_value("示例技能 = Ctrl")
        page.fill("#keys-list", "示例技能 = Ctrl\n跳跃 = Control")
        page.locator("#keys-form button[type=submit]").click()
        expect(page.locator("#keys-status")).to_contain_text("出现了两次")
        page.fill("#keys-character", "示例区 / 二号")
        expect(page.locator("#keys-image")).to_be_hidden()
        assert page.locator("#keys-image").get_attribute("src") is None
        expect(page.locator("#keys-list")).to_have_value("")
        page.fill("#keys-character", "示例区 / 一号")
        expect(page.locator("#keys-list")).to_have_value("示例技能 = Ctrl")
        draft = {
            "schema_version": 1,
            "profile_id": "示例区 / 一号",
            "bindings": [{"skill": "模拟范围攻击", "key": "A"}],
        }
        page.locator("#keys-import").set_input_files(
            {
                "name": "bindings.json",
                "mimeType": "application/json",
                "buffer": json.dumps(draft, ensure_ascii=False).encode(),
            }
        )
        expect(page.locator("#keys-status")).to_contain_text("1 项草稿")
        expect(page.locator("#keys-list")).to_have_value("模拟范围攻击 = A")
        assert (
            page.evaluate("localStorage.getItem('maple.keybindings.v1.示例区 / 一号')")
            == "示例技能 = Ctrl"
        )
        page.locator("#keys-form button[type=submit]").click()
        expect(page.locator("#keys-status")).to_contain_text("保存 1 项")
        page.click('[data-view="runs"]')
        page.select_option("#run-kind", "run")
        page.select_option("#class-select", "dragon_knight")
        page.select_option("#map-select", "dead-forest-drill")
        page.select_option("#mission-select", "dead-forest-hunt")
        page.fill("#run-profile", "示例区 / 一号")
        page.click("#submit-run")
        expect(page.locator("#detail-status")).to_have_text("目标完成", timeout=30000)
        expect(page.locator("#evidence-json")).to_contain_text("input_dispatch_summary")
        expect(page.locator("#evidence-json")).to_contain_text("input_confirmation")
        page.click('[data-view="simulator"]')
        page.select_option("#sim-class", "dragon_knight")
        page.click("#sim-create")
        expect(page.locator("#sim-clock")).to_contain_text("第 0 帧")
        page.fill("#sim-profile", "示例区 / 一号")
        page.click("#sim-key")
        page.locator("#sim-key").press("a")
        expect(page.locator("#sim-key-status")).to_contain_text("已执行")
        expect(page.locator("#sim-key-status")).to_contain_text("本步直接命中")
        page.click('[data-view="keys"]')
        page.fill("#keys-list", "模拟范围攻击 = S")
        page.locator("#keys-form button[type=submit]").click()
        expect(page.locator("#keys-status")).to_contain_text("保存 1 项")
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
        expect(page.locator("#sim-key-log")).to_contain_text("示例区 / 一号")
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
            "confirmed_profile_export_loaded": True,
            "binding_confirmation_receipt_exported": True,
            "task_bound_to_confirmation_receipt": True,
            "reviewed_key_grid_receipt_exported": True,
            "local_key_image_preview_and_clear": True,
            "read_only_recording_replay": True,
            "source_bound_actor_selection_download": True,
            "actor_selection_stale_image_and_response_rejected": True,
            "actor_selection_mobile_and_zoom_coordinates": True,
            "separated_trial_evidence_lanes": 4,
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
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    if not args.browser:
        parser.error("a local Chromium or Chrome executable is required")
    with tempfile.TemporaryDirectory(prefix="maple-browser-") as folder:
        root = Path(folder)
        recordings = root / "recordings"
        recordings.mkdir()
        first = recording_window(recordings, "window-zero", 0, [100, 140], True)
        final = recording_window(recordings, "window-two", 2, [220, 360], False)
        event_reports = root / "recording-events"
        event_reports.mkdir()
        review_events(
            first,
            event_reports / "first.json",
            reviewer="browser-fixture",
            events=[("rope_contact_visible", 1, 1)],
        )
        review_events(
            final,
            event_reports / "final.json",
            reviewer="browser-fixture",
            events=[("attack_motion_visible", 0, 1)],
        )
        coverage = root / "coverage.json"
        summarize_recording_windows([first, final], coverage)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        repository = Path(__file__).resolve().parents[1]
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
                "--recordings",
                str(recordings),
                "--recording-events",
                str(event_reports),
                "--recording-coverage",
                str(coverage),
                "--trial-evidence-root",
                str(repository),
                "--trial-evidence-ledger",
                str(repository / "reports/release-0.10.0/throwing-star-trial-evidence-ledger.json"),
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
                report = verify(url, args.browser, args.screenshots)
            finally:
                service.terminate()
                service.wait(timeout=30)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
