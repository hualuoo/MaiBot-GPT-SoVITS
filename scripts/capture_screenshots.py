"""重新生成 README 与文档中的截图（docs/images/*.png）。

截图来自真实的 MaiBot Dashboard 页面和真实的插件后端代码：
- Dashboard 页面由本机正在运行的 MaiBot WebUI 提供（默认 http://127.0.0.1:8001）。页面发出的接口请求
  全部在浏览器内拦截，交给本脚本里单独创建的插件实例处理，不会改动正在运行的 MaiBot 和已安装插件的配置；
- 本机模型目录是临时创建的演示目录：文件以硬链接指向本机 GPT-SoVITS 中的真实底模、角色文件和参考音频
  （不复制、不修改原文件），页面上的路径统一显示为 D:/GPT-SoVITS；
- 第一步会对 GPT-SoVITS 的 api_v2.py 做一次只读连接测试；不加载模型、不合成语音。

用法（在 MaiBot 的 Python 环境中、项目根目录运行；需要 Playwright 和 Chrome）：
    python -B scripts/capture_screenshots.py --gsv <GPT-SoVITS 目录> --role 黍 [--maibot <MaiBot 目录>]
不传 --maibot 时使用环境变量 MAIBOT_ROOT，或与本项目并列的 MaiBot 文件夹。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

from PIL import Image
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images"
sys.path.insert(0, str(ROOT))
from plugin import create_plugin  # noqa: E402

DEFAULT_MAIBOT = os.environ.get("MAIBOT_ROOT", "").strip() or str(ROOT.parent / "MaiBot")
DISPLAY_ROOT = "D:/GPT-SoVITS"
DISPLAY_PLUGIN = "D:/MaiBot/plugins/github.hualuoo.maibot-gpt-sovits"
CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
# 演示目录中需要的标准底模（相对 GPT_SoVITS/pretrained_models，本机没有的会跳过）。
PRETRAINED = (
    "s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth", "v2Pro/s2Gv2Pro.pth", "sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "gsv-v5-pretrained/s2Gv5turbo.pth", "gsv-v5-pretrained/vocoder.pth",
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "gsv-v2final-pretrained/s2G2333k.pth",
)
EMOTION_NAMES = {"happy": "开心", "angry": "生气", "surprised": "吃惊", "neutral": "中立", "sad": "难过"}
# MaiBot 默认模型配置中的文本模型任务（截图中“情感判断模型任务”下拉框的选项）。
LLM_TASKS = ["replyer", "planner", "memory", "utils", "learner", "fast_model", "emoji", "vlm"]


def build_demo(gsv: Path, role: str, base: Path) -> tuple[Path, dict]:
    """在 base 下建立演示用 GPT-SoVITS 目录，所有大文件都是硬链接。"""
    root = base / "GPT-SoVITS"

    def link(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)

    pretrained = gsv / "GPT_SoVITS" / "pretrained_models"
    for relative in PRETRAINED:
        if (pretrained / relative).is_file():
            link(pretrained / relative, root / "GPT_SoVITS" / "pretrained_models" / relative)
    source_role = gsv / "maibot_roles" / role
    manifest = json.loads((source_role / "role.json").read_text(encoding="utf-8"))
    for item in source_role.iterdir():
        if item.is_file():
            target = root / "maibot_roles" / role / item.name
            if item.name == "role.json":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
            else:
                link(item, target)
    # 普通模式的参考音频：把角色各情感的参考音频放进 maibot_refs/<角色>/，文件名用中文情感名。
    for emotion, fields in manifest["emotions"].items():
        name = EMOTION_NAMES.get(emotion, emotion) + Path(fields["ref_audio"]["name"]).suffix
        link(source_role / fields["ref_audio"]["name"], root / "maibot_refs" / role / name)
    return root, manifest


class Alias:
    """页面上显示 D:/GPT-SoVITS，交给插件时换回演示目录的真实路径。"""

    def __init__(self, real_root: Path, real_plugin: Path):
        self.show_pairs = [(real_root.as_posix(), DISPLAY_ROOT), (str(real_root), DISPLAY_ROOT.replace("/", "\\")),
                           (real_plugin.as_posix(), DISPLAY_PLUGIN), (str(real_plugin), DISPLAY_PLUGIN.replace("/", "\\"))]
        self.real_pairs = [(DISPLAY_ROOT, real_root.as_posix()), (DISPLAY_PLUGIN, real_plugin.as_posix())]

    def _map(self, value, pairs):
        if isinstance(value, str):
            for old, new in pairs:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [self._map(item, pairs) for item in value]
        if isinstance(value, dict):
            return {key: self._map(item, pairs) for key, item in value.items()}
        return value

    def show(self, value):
        return self._map(value, self.show_pairs)

    def real(self, value):
        return self._map(value, self.real_pairs)


def optimize(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        image.save(path, optimize=True)


async def capture(args) -> None:
    base = Path(tempfile.mkdtemp(prefix="gsv-readme-")).resolve()
    plugin = create_plugin()
    try:
        root, role_manifest = build_demo(Path(args.gsv), args.role, base)
        plugin_dir = base / "plugin"
        plugin_dir.mkdir()
        plugin._settings_path = plugin_dir / "config.toml"
        values = plugin.build_default_config()
        values["api"]["base_url"] = args.api
        values["models"].update(root=root.as_posix())
        plugin.set_plugin_config(values)
        await plugin.on_load()
        alias = Alias(root, plugin_dir)

        sys.path.insert(0, str(args.maibot))
        from src.plugin_runtime.webui_schema import load_webui_extension

        manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
        schema = plugin.get_webui_config_schema(plugin_id=manifest["id"], plugin_name=manifest["name"],
                                                plugin_version=manifest["version"])
        extension = load_webui_extension(str(ROOT))
        page_meta = extension.pages[0]
        serialized = {"plugin_id": manifest["id"], **extension.model_dump(mode="json", exclude={"pages"}),
                      "pages": [{**page_meta.model_dump(mode="json", exclude={"document"}),
                                 "document_revision": page_meta.document_revision}]}
        apis = {**{name: item.api for name, item in page_meta.queries.items()},
                **{name: item.api for name, item in page_meta.actions.items()}}
        entry = {"id": manifest["id"], "manifest": manifest, "path": DISPLAY_PLUGIN, "enabled": True,
                 "disabled": False, "loaded": True, "load_status": "success"}
        icon = manifest["display"]["icon"]["value"]

        async def available_models():
            return list(LLM_TASKS)

        # 只为“情感判断模型任务”下拉框提供 MaiBot 模型任务列表；截图过程不调用 LLM、不发送消息。
        plugin._ctx = SimpleNamespace(llm=SimpleNamespace(get_available_models=available_models),
                                      logger=logging.getLogger("readme-capture"))

        async def respond(route):
            request = route.request
            path = unquote(urlsplit(request.url).path)
            if path.endswith("/auth/check"):
                value = {"authenticated": True, "requires_custom_token": False}
            elif path.endswith("/setup/status"):
                value = {"is_first_setup": False, "requires_custom_token": False}
            elif path.endswith("/plugins/installed"):
                value = {"success": True, "plugins": [entry]}
            elif path.endswith("/bundle"):
                value = {"success": True, "schema": alias.show(schema), "config": alias.show(plugin.config.model_dump(mode="json")),
                         "raw_config": "", "config_error": ""}
            elif path.endswith("/plugins/runtime/webui"):
                value = {"success": True, "schema_version": 2, "extensions": [serialized],
                         "capabilities": ["custom_pages", "structured_arguments", "declarative_basic", "image_upload"]}
            elif path.endswith("/document"):
                value = {"success": True, "document": page_meta.document, "revision": page_meta.document_revision}
            elif "/runtime/webui/" in path and request.method == "POST":
                name = path.rsplit("/", 1)[-1]
                try:
                    result = await getattr(plugin, apis[name])(**alias.real(request.post_data_json["args"]))
                except Exception as exc:
                    await route.fulfill(status=400, content_type="application/json", body=json.dumps({"detail": str(exc)}))
                    return
                value = {"success": True, "result": alias.show(result)}
            elif path.endswith("/plugins/icon/" + manifest["id"]):
                await route.fulfill(status=200, content_type=mimetypes.guess_type(icon)[0], body=(ROOT / icon).read_bytes())
                return
            elif path.endswith("/version"):
                value = {"version": "1.3.19", "version_major": 1, "version_minor": 3, "version_patch": 19}
            else:
                value = {"success": True, "plugins": [], "extensions": [], "pages": [], "enabled": False}
            await route.fulfill(status=200, content_type="application/json", body=json.dumps(value, ensure_ascii=False))

        OUTPUT.mkdir(parents=True, exist_ok=True)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True, executable_path=CHROME)
            try:
                # 高视口让每一步的卡片完整显示；最后的总览截图再缩回常见的窗口高度。
                page = await browser.new_page(viewport={"width": 1450, "height": 2400})
                await page.route("**/api/**", respond)
                await page.route_web_socket("**/*", lambda ws: ws.close())

                async def shot(locator, name):
                    await locator.screenshot(path=str(OUTPUT / name), animations="disabled")
                    optimize(OUTPUT / name)
                    print("saved", name)

                async def done(step: int, timeout: int = 20000) -> None:
                    """等待右栏第 step 步变绿；失败时打印页面上的原因，方便排查。"""
                    try:
                        await expect(frame.locator(f"#status-step-{step}")).to_have_attribute("data-tone", "success", timeout=timeout)
                    except AssertionError:
                        for selector in (f"#status-step-{step}-text", "#global-message", "#operation-error-message", "#check-config-message"):
                            print("DIAGNOSTIC", selector, await frame.locator(selector).inner_text())
                        raise

                async def idle() -> None:
                    """等待页面上的保存/检查请求结束（“立即检查”按钮恢复可用）。"""
                    await expect(frame.locator("#check-config")).to_be_enabled(timeout=20000)
                    await page.wait_for_timeout(300)

                async def shot_between(top, bottom, name, margin=16):
                    """从 top 元素顶部截到 bottom 元素底部，避免整张卡片过长。"""
                    await top.scroll_into_view_if_needed()
                    first, last = await top.bounding_box(), await bottom.bounding_box()
                    clip = {"x": first["x"], "y": first["y"], "width": first["width"],
                            "height": last["y"] + last["height"] + margin - first["y"]}
                    await page.screenshot(path=str(OUTPUT / name), clip=clip, animations="disabled")
                    optimize(OUTPUT / name)
                    print("saved", name)

                # 插件配置列表：图标、版本与简介。
                await page.set_viewport_size({"width": 1450, "height": 900})
                await page.goto(args.dashboard + "/plugin-config", wait_until="domcontentloaded")
                image = page.locator(f'img[src="/api/webui/plugins/icon/{manifest["id"]}"]').first
                await expect(image).to_be_visible(timeout=20000)
                await page.wait_for_function("s=>{const i=document.querySelector('img[src=\"'+s+'\"]');return Boolean(i&&i.complete&&i.naturalWidth)}",
                                             arg=f"/api/webui/plugins/icon/{manifest['id']}", timeout=10000)
                row = page.locator("div", has=image.locator("xpath=..")).filter(has_text=manifest["description"]).last
                box = await row.bounding_box()
                await page.screenshot(path=str(OUTPUT / "plugin-list.png"), animations="disabled",
                                      clip={"x": box["x"] - 12, "y": box["y"] - 12, "width": box["width"] + 24, "height": box["height"] + 24})
                optimize(OUTPUT / "plugin-list.png")
                print("saved plugin-list.png")

                await page.set_viewport_size({"width": 1450, "height": 2400})
                await page.goto(f"{args.dashboard}/extensions/{manifest['id']}/{page_meta.id}", wait_until="domcontentloaded")
                await expect(page.locator(f'iframe[title="{page_meta.title}"]')).to_be_visible(timeout=20000)
                frame = page.frame_locator(f'iframe[title="{page_meta.title}"]')

                # 第一步：保存并测试连接（对 api_v2.py 的只读探测）。
                await expect(frame.locator("#test-api")).to_be_enabled(timeout=20000)
                await frame.locator("#test-api").click()
                await done(1, 30000)
                await shot(frame.locator("#panel-api"), "step1-api.png")

                # 第二步：扫描演示目录，选择 v2ProPlus 标准底模。
                await frame.locator("#next-step").click()
                await expect(frame.locator("#step-base")).to_have_attribute("aria-current", "step", timeout=15000)
                await frame.locator("#scan-models").click()
                base_model = DISPLAY_ROOT + "/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
                await expect(frame.locator(f'#sovits-model-picker option[value="{base_model}"]')).to_have_count(1, timeout=20000)
                await frame.locator("#sovits-model-picker").select_option(base_model)
                await expect(frame.locator("#save-base")).to_be_enabled(timeout=15000)
                await shot(frame.locator("#base-card"), "step2-base-model.png")
                await frame.locator("#save-base").click()
                await done(2)

                # 第三步（普通模式）：标准 GPT + maibot_refs 中的参考音频，即零样本音色克隆。
                await frame.locator("#step-models").click()
                await expect(frame.locator("#step-models")).to_have_attribute("aria-current", "step", timeout=15000)
                reference = f"{DISPLAY_ROOT}/maibot_refs/{args.role}/{EMOTION_NAMES['neutral']}.mp3"
                await expect(frame.locator(f'#ref-audio-path option[value="{reference}"]')).to_have_count(1, timeout=20000)
                await frame.locator("#ref-audio-path").select_option(reference)
                await frame.locator("#prompt-text").fill(role_manifest["emotions"]["neutral"]["ref_text"])
                await frame.locator("#save-models").click()
                await done(3)
                await shot(frame.locator("#mode-card"), "step3-normal.png")

                # 第三步（角色模式）：读取 role.json，启用全部情感并保存情感判断设置。
                await frame.locator("#role-mode-btn").click()
                role_id = args.role + "/role.json"
                await expect(frame.locator(f'#role-picker option[value="{role_id}"]')).to_have_count(1, timeout=20000)
                await frame.locator("#role-picker").select_option(role_id)
                boxes = frame.locator("#emotion-options input[type=checkbox]")
                await expect(boxes).to_have_count(len(role_manifest["emotions"]), timeout=15000)
                for index in range(await boxes.count()):
                    await boxes.nth(index).check()
                await frame.locator("#emotion-picker").select_option("neutral")
                await frame.locator("#apply-role").click()
                await expect(frame.locator("#role-status")).to_contain_text("已就绪", timeout=120000)
                await expect(frame.locator("#llm-task")).to_be_enabled(timeout=15000)
                await frame.locator("#llm-task").select_option("fast_model")
                await frame.locator("#save-emotion-judge").click()
                await done(3)
                await idle()
                await shot_between(frame.locator("#mode-card"), frame.locator("#role-status"), "step3-role.png")
                await shot(frame.locator("#emotion-judge"), "emotion-judge.png")

                # 第四步：保存生成参数（试听可选，这里不加载模型）。
                await frame.locator("#next-parameters").click()
                await expect(frame.locator("#step-parameters")).to_have_attribute("aria-current", "step", timeout=15000)
                await idle()
                await frame.locator("#save-parameters").click()
                await done(4)
                await shot(frame.locator("#parameters-card"), "step4-parameters.png")

                # 第五步：开启自动语音回复（按概率 30%，最短间隔 10 秒）。
                await idle()
                await frame.locator("#next-auto").click()
                await expect(frame.locator("#step-auto")).to_have_attribute("aria-current", "step", timeout=15000)
                await idle()
                await frame.locator("#auto-enabled").check()
                await frame.locator("#auto-probability").fill("30")
                await frame.locator("#auto-min-interval").fill("10")
                await frame.locator("#save-auto").click()
                await done(5)
                await shot(frame.locator("#auto-card"), "step5-auto-voice.png")

                # 总览：回到第三步（角色模式），右侧五步状态全部为绿色。
                await page.set_viewport_size({"width": 1450, "height": 1000})
                await frame.locator("#step-models").click()
                await expect(frame.locator("#role-card")).to_be_visible(timeout=15000)
                await frame.locator("#check-config").click()
                await expect(frame.locator("#check-config")).to_be_enabled(timeout=20000)
                await page.wait_for_timeout(500)
                await page.screenshot(path=str(OUTPUT / "setup-overview.png"), animations="disabled")
                optimize(OUTPUT / "setup-overview.png")
                print("saved setup-overview.png")
                # 右栏：当前配置检查与当前状态（五步全部完成）。
                await page.set_viewport_size({"width": 1450, "height": 2400})
                await done(5, 15000)
                await shot_between(frame.locator("#check-config-card"), frame.locator("#status-step-5"), "status-panel.png", margin=10)
            finally:
                await browser.close()
    finally:
        await plugin.on_unload()
        # 只删除演示目录里的硬链接与临时配置，GPT-SoVITS 中的原文件不受影响。
        shutil.rmtree(base, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="重新生成 README 与文档中的截图")
    parser.add_argument("--gsv", required=True, help="本机 GPT-SoVITS 安装目录（需要 maibot_roles/<角色>/role.json）")
    parser.add_argument("--role", default="黍", help="maibot_roles 中用于截图的角色文件夹名")
    parser.add_argument("--api", default="http://127.0.0.1:9880", help="正在运行的 api_v2.py 地址（只做连接测试）")
    parser.add_argument("--maibot", default=DEFAULT_MAIBOT, help="MaiBot 根目录（读取页面声明）")
    parser.add_argument("--dashboard", default="http://127.0.0.1:8001", help="提供 Dashboard 页面的 MaiBot WebUI 地址")
    asyncio.run(capture(parser.parse_args()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
