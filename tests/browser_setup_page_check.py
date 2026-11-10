"""在真实的 MaiBot Dashboard 中检查五步设置页；接口请求全部在浏览器内拦截，交给本地插件实例处理。

需要：Playwright、Chrome（或 Playwright 自带的 Chromium）、正在运行的 MaiBot WebUI（只用来提供 Dashboard 页面）
以及 MaiBot 源码（环境变量 MAIBOT_ROOT，或与本项目并列的 MaiBot 文件夹）。不会改动 MaiBot 及已安装插件的配置。

用法（项目根目录）：
    python -B tests/browser_setup_page_check.py              # 完整五步：保存参数与自动语音
    python -B tests/browser_setup_page_check.py --gating     # 全新配置：未完成第一步时不能进入后续步骤
    python -B tests/browser_setup_page_check.py --resume=3   # 恢复到第 2～5 步，后面的步骤保持锁定
    python -B tests/browser_setup_page_check.py --icon       # 插件配置列表显示插件图标与版本

环境变量：MAIBOT_WEBUI（默认 http://127.0.0.1:8001）、CHROME_PATH（默认 Windows 下的 Chrome 安装位置）。
"""
import asyncio
import json
import mimetypes
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import unquote, urlsplit

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_paths import MAIBOT_ROOT, SKIP_REASON, use_maibot  # noqa: E402
from plugin import create_plugin  # noqa: E402

WEBUI = os.environ.get("MAIBOT_WEBUI", "http://127.0.0.1:8001").rstrip("/")
CHROME = os.environ.get("CHROME_PATH", "C:/Program Files/Google/Chrome/Application/chrome.exe")
OUTPUT = Path(tempfile.gettempdir()) / "gsv-ui-check"


async def main():
    if MAIBOT_ROOT is None:
        sys.exit(SKIP_REASON)
    use_maibot()
    from src.plugin_runtime.webui_schema import load_webui_extension

    temporary = tempfile.TemporaryDirectory()
    base = Path(temporary.name).resolve()
    models_root = base / "GPT-SoVITS"
    models_root.mkdir()
    plugin = create_plugin()
    plugin._settings_path = base / "config.toml"
    config = plugin.build_default_config()
    config["models"]["root"] = models_root.as_posix()
    plugin.set_plugin_config(config)
    await plugin.on_load()
    resume_step = next((int(arg.split("=", 1)[1]) for arg in sys.argv if arg.startswith("--resume=")), None)
    gating_scenario = "--gating" in sys.argv
    icon_scenario = "--icon" in sys.argv
    # 其余场景使用前四步都已完成的配置（--resume 时只完成到指定步骤之前）。
    complete_steps = resume_step or (None if gating_scenario else 5)
    if complete_steps:
        assert complete_steps in (2, 3, 4, 5)
        cfg = plugin.config.model_dump()
        cfg["role"].update(mode="original", role_id="", emotions=[])
        cfg["models"].update(base_model="v5turbo", gpt_weights="fixture.ckpt", sovits_weights="fixture.pth")
        # 没有第二步记录时会按已保存的标准底模推断第二步已完成；恢复到第二步需要未选择底模。
        if complete_steps == 2:
            cfg["models"].update(base_model="custom", gpt_weights="", sovits_weights="")
        if complete_steps >= 4:
            cfg["voice"].update(ref_audio_path="fixture.wav", prompt_text="测试参考文本")
        else:
            cfg["voice"].update(ref_audio_path="", prompt_text="")
        plugin.set_plugin_config(cfg)
        from wizard_progress import WizardProgress
        for step in range(1, complete_steps):
            WizardProgress(plugin._settings_path).mark(step, plugin.config)
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    icon = manifest.get("display", {}).get("icon", {})
    extension = load_webui_extension(str(ROOT))
    page_meta = extension.pages[0]
    serialized = {"plugin_id": manifest["id"], **extension.model_dump(mode="json", exclude={"pages"}),
                  "pages": [{**page_meta.model_dump(mode="json", exclude={"document"}),
                             "document_revision": page_meta.document_revision}]}
    entry = {"id": manifest["id"], "manifest": manifest, "path": str(base), "enabled": True, "disabled": False,
             "loaded": True, "load_status": "success"}
    requests, errors, parameter_saves = [], [], []
    bridge = {"active": 0, "peak": 0}

    async def ensure_connection_fixture():
        plugin._mark_setup_connected(plugin.config.api.base_url, "已自动确认保存的接口连接")
        return {"success": True, "message": "已自动确认连接", "connection": dict(plugin._connection)}

    bindings = {"ensure_connection": ensure_connection_fixture, "state": plugin.setup_state, "catalog": plugin.setup_catalog,
                "check_config": plugin.setup_check_config, "roles": plugin.setup_roles, "llm_tasks": plugin.setup_llm_tasks,
                "save_parameters": plugin.setup_save_parameters, "save_auto_voice": plugin.setup_save_auto_voice}

    async def respond(route):
        request = route.request
        path = unquote(urlsplit(request.url).path)
        requests.append((request.method, path))
        if path.endswith("/auth/check"):
            value = {"authenticated": True, "requires_custom_token": False}
        elif path.endswith("/setup/status"):
            value = {"is_first_setup": False, "requires_custom_token": False}
        elif path.endswith("/plugins/installed"):
            value = {"success": True, "plugins": [entry]}
        elif path.endswith("/plugins/runtime/webui"):
            value = {"success": True, "schema_version": 2, "extensions": [serialized],
                     "capabilities": ["custom_pages", "structured_arguments", "declarative_basic", "image_upload"]}
        elif path.endswith("/document"):
            value = {"success": True, "document": page_meta.document, "revision": page_meta.document_revision}
        elif "/runtime/webui/" in path and request.method == "POST":
            name = path.rsplit("/", 1)[-1]
            args = request.post_data_json["args"]
            if name == "save_parameters":
                parameter_saves.append(args)
            bridge["active"] += 1
            bridge["peak"] = max(bridge["peak"], bridge["active"])
            try:
                if bridge["active"] > 1:
                    raise RuntimeError("page request queue allowed concurrent bridge calls")
                # 故意延迟初始化查询，重现用户立即点击配置检查的重叠窗口。
                if name in ("catalog", "roles", "check_config"):
                    await asyncio.sleep(.15)
                value = {"success": True, "result": await bindings[name](**args)}
            except Exception as exc:
                await route.fulfill(status=400, content_type="application/json", body=json.dumps({"detail": str(exc)}))
                return
            finally:
                bridge["active"] -= 1
        elif icon.get("type") == "local" and path.endswith("/plugins/icon/" + manifest["id"]):
            await route.fulfill(status=200, content_type=mimetypes.guess_type(icon["value"])[0] or "application/octet-stream",
                                body=(ROOT / icon["value"]).read_bytes())
            return
        elif path.endswith("/version"):
            value = {"version": "1.3.19", "version_major": 1, "version_minor": 3, "version_patch": 19}
        else:
            value = {"success": True, "plugins": [], "extensions": [], "pages": [], "enabled": False}
        await route.fulfill(status=200, content_type="application/json", body=json.dumps(value, ensure_ascii=False))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        async with async_playwright() as playwright:
            launch = {"headless": True}
            if Path(CHROME).is_file():
                launch["executable_path"] = CHROME
            browser = await playwright.chromium.launch(**launch)
            try:
                page = await browser.new_page(viewport={"width": 1450, "height": 1100}, device_scale_factor=2 if icon_scenario else 1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.route("**/api/**", respond)
                await page.route_web_socket("**/*", lambda ws: ws.close())
                start = "/plugin-config" if icon_scenario else f"/extensions/{manifest['id']}/{page_meta.id}"
                await page.goto(WEBUI + start, wait_until="domcontentloaded")
                try:
                    if icon_scenario:
                        source = "/api/webui/plugins/icon/" + manifest["id"]
                        image = page.locator(f'img[src="{source}"]').first
                        await expect(image).to_be_visible(timeout=20000)
                        await page.wait_for_function("s=>{const i=document.querySelector('img[src=\"'+s+'\"]');return Boolean(i&&i.complete&&i.naturalWidth)}",
                                                     arg=source, timeout=10000)
                        assert await image.evaluate("i=>[i.naturalWidth,i.naturalHeight]") == [256, 256]
                        tile = image.locator("xpath=..")
                        expected = "rgb({}, {}, {})".format(*bytes.fromhex(icon["background"][1:]))
                        assert await tile.evaluate("e=>getComputedStyle(e).backgroundColor") == expected
                        await expect(page.get_by_text("v" + manifest["version"], exact=True).first).to_be_visible()
                        await page.screenshot(path=str(OUTPUT / "plugin-icon-list.png"))
                        print("PASS: plugin list shows the manifest logo (256x256 PNG via the MaiBot icon route) on its background, version v" + manifest["version"])
                        return
                    frame = page.frame_locator(f'iframe[title="{page_meta.title}"]')
                    if await page.locator(f'iframe[title="{page_meta.title}"]').count() == 0:
                        frame = page.frame_locator("iframe").first
                    if gating_scenario:
                        await expect(frame.locator("#step-api")).to_have_attribute("aria-current", "step", timeout=20000)
                        for locked in ("#step-base", "#step-models", "#step-parameters", "#step-auto", "#next-step"):
                            await expect(frame.locator(locked)).to_be_disabled(timeout=15000)
                        await expect(frame.locator("#step-models")).to_have_attribute("title", "请先完成第1步「API 配置测试」")
                        await frame.locator("#status-step-4").click()
                        await expect(frame.locator("#global-message")).to_contain_text("请先完成第1步「API 配置测试」")
                        await expect(frame.locator("#step-api")).to_have_attribute("aria-current", "step")
                        await page.screenshot(path=str(OUTPUT / "gating-step1.png"))
                        print("PASS: fresh configuration cannot skip ahead; later steps stay locked until step 1 is complete")
                        return
                    # 每一步完成才能进入下一步：恢复到第二、三步时后面的步骤仍锁定。
                    partial = bool(resume_step and resume_step < 4)
                    if partial:
                        await expect(frame.locator("#step-auto")).to_be_disabled(timeout=20000)
                    else:
                        await expect(frame.locator("#step-parameters")).to_be_enabled(timeout=20000)
                    await expect(frame.locator("#step-auto")).to_be_visible()
                    if resume_step:
                        target = {2: "base", 3: "models", 4: "parameters", 5: "auto"}[resume_step]
                        await expect(frame.locator("#step-" + target)).to_have_attribute("aria-current", "step", timeout=15000)
                        await expect(frame.locator("#global-message")).to_contain_text("恢复", timeout=15000)
                        await expect(frame.locator("#next-step")).to_be_enabled(timeout=15000)
                    await expect(frame.locator("#check-config")).to_be_enabled(timeout=15000)
                    await frame.locator("#check-config").click()
                    await expect(frame.locator("#check-config-message")).not_to_contain_text("Two requests", timeout=15000)
                    await expect(frame.locator("#check-config")).to_be_enabled(timeout=15000)
                    assert await frame.locator(".layout").evaluate("e=>getComputedStyle(e).gridTemplateColumns.split(' ').length===2")
                    if partial:
                        # 第二步未完成时第三步起锁定；第三步（普通模式缺参考音频）未完成时第四步起锁定。
                        locked, label = {2: ("#step-models", "请先完成第2步「选择底模」"),
                                         3: ("#step-parameters", "请先完成第3步「声音模式」")}[resume_step]
                        await expect(frame.locator(locked)).to_be_disabled()
                        await expect(frame.locator(locked)).to_have_attribute("title", label)
                        if resume_step == 2:
                            # 未选择底模时“保存底模并继续”不可点击。
                            await expect(frame.locator("#save-base")).to_be_disabled()
                            await expect(frame.locator("#save-base")).to_have_attribute("title", "请先在下拉框中选择 SoVITS 底模或完整权重")
                        await frame.locator("#status-step-5").click()
                        await expect(frame.locator("#global-message")).to_contain_text("请先完成第" + str(resume_step) + "步")
                        if resume_step == 3:
                            # 参考音频在第三步：普通模式选择框和放置说明都在第三步。
                            await expect(frame.locator("#ref-audio-path")).to_be_visible()
                            await expect(frame.locator("#status-step-3")).to_contain_text("尚未选择参考音频")
                        await page.screenshot(path=str(OUTPUT / f"resume-step{resume_step}.png"))
                        assert not [error for error in errors if "WebSocket" not in error], errors
                        print("PASS: persisted partial configuration resumes step", resume_step, "and the following steps stay locked")
                        return
                    await frame.locator("#step-models").click()
                    await expect(frame.locator("#mode-selector")).to_have_attribute("role", "tablist")
                    await expect(frame.locator("#role-mode-btn")).to_have_attribute("role", "tab")
                    await expect(frame.locator("#original-mode-btn")).to_have_attribute("role", "tab")
                    await frame.locator("#step-parameters").click()
                    await frame.locator("#speed-factor").fill("1.2")
                    await expect(frame.locator("#save-parameters")).to_be_enabled(timeout=15000)
                    async with page.expect_response(lambda response: response.request.method == "POST" and "/save_parameters" in response.url):
                        await frame.locator("#save-parameters").click()
                    await expect(frame.locator("#save-parameters")).to_be_enabled()
                    assert parameter_saves and parameter_saves[-1]["speed_factor"] == 1.2
                    assert plugin.config.voice.speed_factor == 1.2
                    # 试听与参数在同一步（第四步），可选，不需要切换页面。
                    await expect(frame.locator("#preview-run")).to_be_visible()
                    await frame.locator("#step-auto").click()
                    await expect(frame.locator("#auto-card")).to_be_visible()
                    await frame.locator("#auto-enabled").check()
                    await frame.locator("#auto-frequency-mode").select_option("every_n")
                    await frame.locator("#auto-every-n").fill("3")
                    await expect(frame.locator("#auto-probability")).to_be_disabled()
                    async with page.expect_response(lambda response: response.request.method == "POST" and "/save_auto_voice" in response.url):
                        await frame.locator("#save-auto").click()
                    await expect(frame.locator("#save-auto")).to_be_enabled()
                    assert plugin.config.auto_voice.enabled and plugin.config.auto_voice.frequency_mode == "every_n"
                    assert plugin.config.auto_voice.every_n == 3
                    assert plugin._auto_stats["converted"] == 0
                    assert not [r for r in requests if r[0] == "PUT"], "the page must save through plugin APIs only"
                    await page.screenshot(path=str(OUTPUT / "five-steps.png"))
                except Exception:
                    print("PAGE DIAGNOSTIC", (await page.locator("body").inner_text())[:4000])
                    for candidate in page.frames:
                        if candidate != page.main_frame:
                            try:
                                print("IFRAME DIAGNOSTIC", (await candidate.locator("body").inner_text())[:7000])
                            except Exception:
                                pass
                    print("REQUESTS", requests[-20:])
                    print("JS", errors)
                    raise
                unexpected = [error for error in errors if "WebSocket" not in error]
                assert not unexpected, unexpected
                assert bridge["peak"] == 1, bridge["peak"]
                print("PASS: five-step setup page in the MaiBot workspace route saves parameters and auto voice through plugin APIs")
                print("PASS: optional preview stays on step four; page requests are serialized")
            finally:
                await browser.close()
    finally:
        await plugin.on_unload()
        temporary.cleanup()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
