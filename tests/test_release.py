"""发行一致性：插件图标、版本号、发行包与安装脚本保持一致。

python -B -m unittest discover -s tests -p test_release.py -v
"""
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_release


def manifest() -> dict:
    return json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))


class ReleaseTests(unittest.TestCase):
    def test_manifest_declares_local_png_logo_in_assets(self):
        icon = manifest()["display"]["icon"]
        self.assertEqual(icon["type"], "local")
        self.assertEqual(icon["value"], "assets/maibot-microphone-chibi-256.png")
        self.assertRegex(icon["background"], r"^#[0-9A-Fa-f]{6}$")
        # 后备图标必须是 MaiBot Dashboard 认得的 lucide 名称，否则会退回默认拼图图标。
        self.assertIn(icon["fallback"], {"bot", "settings", "image", "plug", "puzzle", "package", "search", "shield",
                                         "wrench", "cloud", "database", "book-open", "scroll-text", "bar-chart-3", "gamepad2"})
        data = (ROOT / icon["value"]).read_bytes()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", data[16:24]), (256, 256), "图标应为正方形，MaiBot 会裁切填满方框")
        self.assertLessEqual(len(data), build_release.MAX_ICON_BYTES)
        self.assertEqual(build_release.manifest_icon(manifest()), icon["value"])
        self.assertFalse((ROOT / PurePosixPath(icon["value"]).name).exists(), "图标只放在 assets/ 中")

    def test_logo_is_packaged_byte_for_byte_and_copied_by_install_script(self):
        icon = build_release.manifest_icon(manifest())
        self.assertIn(icon, build_release.FILES)
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = build_release.build_release(Path(temporary))
            self.assertEqual(archive_path.name, f"MaiBot-GPT-SoVITS-{manifest()['version']}.zip")
            with ZipFile(archive_path) as archive:
                self.assertEqual(archive.read(icon), (ROOT / icon).read_bytes())
                packaged = json.loads(archive.read("_manifest.json"))
                self.assertEqual(packaged["display"]["icon"]["value"], icon)
        script = (ROOT / "install.ps1").read_text(encoding="utf-8")
        self.assertIn(f'(Join-Path $source "{icon}")', script)
        self.assertIn('Join-Path $target "assets"', script)

    def test_build_refuses_a_declared_logo_that_would_not_be_packaged(self):
        with patch.object(build_release, "FILES", [name for name in build_release.FILES
                                                   if not name.startswith("assets/")]), \
                tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "插件图标 assets/maibot-microphone-chibi-256.png 不在 FILES"):
                build_release.build_release(Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])
        with patch.object(build_release, "MAX_ICON_BYTES", 10), tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "512 KB"):
                build_release.build_release(Path(temporary))

    def test_release_ships_usage_docs_but_no_development_tools(self):
        docs = sorted(path.relative_to(ROOT).as_posix() for pattern in ("docs/*.md", "docs/images/*.png")
                      for path in ROOT.glob(pattern))
        self.assertGreaterEqual(len(docs), 20)
        with tempfile.TemporaryDirectory() as temporary:
            with ZipFile(build_release.build_release(Path(temporary))) as archive:
                names = archive.namelist()
                for name in docs:
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)
        self.assertFalse([name for name in names if name.startswith(("scripts/", "tests/", "maibot-patch/", "dist/"))])
        script = (ROOT / "install.ps1").read_text(encoding="utf-8")
        self.assertIn('(Join-Path $source "docs/*.md")', script)
        self.assertIn('(Join-Path $source "docs/images/*.png")', script)

    def test_version_badge_matches_manifest_and_first_changelog_entry(self):
        version = manifest()["version"]
        self.assertEqual(version, "1.0.0")
        self.assertRegex(version, r"^\d+\.\d+\.\d+$", "MaiBot 要求 x.y.z，不能带 v 前缀")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f'<a href="_manifest.json"><img src="https://img.shields.io/badge/version-{version}-blue" alt="version"></a>', readme)
        changelog = readme[readme.index("## 更新日志"):]
        # 首次发布：更新日志只有当前版本一条。
        self.assertEqual(re.findall(r"^#+ (\d+\.\d+\.\d+)", changelog, re.M), [version])
        # 文档里的发行包名不写死版本号，发布新版本时只需改 _manifest.json、徽章和更新日志。
        for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"MaiBot-GPT-SoVITS-\d", path.name)
        self.assertIn("MaiBot-GPT-SoVITS-<版本>.zip", (ROOT / "docs" / "installation.md").read_text(encoding="utf-8"))

    def test_plugin_identity_matches_installation_and_log_documentation(self):
        plugin_id = manifest()["id"]
        self.assertEqual(plugin_id, "github.hualuoo.maibot-gpt-sovits")
        self.assertRegex(plugin_id, r"^[a-z0-9]+(?:[.-][a-z0-9]+)+$")
        installation = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
        self.assertIn(f"| 插件 ID | `{plugin_id}` |", installation)
        for name in ("docs/usage.md", "maibot-patch/sky-blue-log.patch"):
            self.assertIn(f"plugin.{plugin_id}", (ROOT / name).read_text(encoding="utf-8"), name)
        script = (ROOT / "install.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('$pluginId = (Get-Content -LiteralPath (Join-Path $source "_manifest.json")', script)
        self.assertIn(".id -eq $pluginId", script)
        self.assertIn('Join-Path $MaiBotPath ("plugins/" + $pluginId)', script)

    def test_publication_repository_and_author_are_consistent(self):
        info = manifest()
        self.assertEqual(info["urls"]["repository"], "https://github.com/hualuoo/MaiBot-GPT-SoVITS")
        self.assertFalse(info["urls"]["repository"].endswith(".git"))
        self.assertEqual(info["author"], {"name": "HuaLuoo", "url": "https://github.com/hualuoo/"})
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 HuaLuoo", license_text)

    def test_sdk_range_matches_requirements_and_installation_documentation(self):
        self.assertEqual(manifest()["sdk"], {"min_version": "2.20.0", "max_version": "2.21.99"})
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("maibot-plugin-sdk>=2.20.0,<2.22", requirements)
        installation = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
        self.assertIn("| 插件 SDK | maibot-plugin-sdk 2.20.0 ～ 2.21.99（验证版本 2.21.0） |", installation)

    def test_readme_title_is_centered(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        title = "MaiBot GPT-SoVITS 语音合成插件"
        self.assertIn(f'<h1 align="center">{title}</h1>', readme)
        self.assertEqual(len(re.findall(r"<h1\b", readme)), 1)
        self.assertNotIn(f"# {title}", readme)

    def test_badges_match_project_facts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r'<p align="center">(.*?)</p>', readme, re.S)
        badge_block = next(block for block in blocks if "img.shields.io/badge/" in block)
        badges = {label: url for url, label in re.findall(
            r'<img src="(https://img\.shields\.io/badge/[^\"]+)" alt="([^\"]+)">', badge_block)}
        self.assertEqual(list(badges), ["version", "MaiBot", "license"])
        self.assertEqual(readme.count("https://img.shields.io/badge/"), 3)
        host = manifest()["host_application"]
        self.assertEqual(host, {"min_version": "1.3.6", "max_version": "1.4.99"})
        self.assertIn(f"/MaiBot-{host['min_version']}--1.4.x-", badges["MaiBot"])
        self.assertNotIn("1131116861", badge_block)
        self.assertNotIn("[![python]", readme)
        self.assertNotIn("[![GPT-SoVITS]", readme)
        self.assertIn(f"/license-{manifest()['license']}-", badges["license"])
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8").startswith(f"{manifest()['license']} License"))

    def test_supported_versions_match_installation_documentation(self):
        installation = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
        self.assertIn("| MaiBot | 1.3.6 ～ 1.4.x |", installation)

    def test_community_group_and_video_are_centered_directly_below_badges(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        paragraphs = readme.split("\n\n")
        badges = next(index for index, paragraph in enumerate(paragraphs) if "img.shields.io/badge/" in paragraph)
        video_url = "https://www.bilibili.com/video/BV1dP4i68EbA/"
        community = ('<p align="center">\n'
                     f'  QQ 交流群：<strong>1131116861</strong> · <a href="{video_url}">效果预览视频</a>\n'
                     '</p>')
        self.assertEqual(paragraphs[badges + 1], community)
        self.assertTrue(paragraphs[badges + 2].startswith("基于 "))
        self.assertTrue(paragraphs[badges + 3].startswith("![设置页总览"))
        self.assertEqual(readme.count("1131116861"), 1)
        self.assertEqual(readme.count(video_url), 1)
        self.assertNotIn("## 交流", readme)
        self.assertNotIn("[![QQ", readme)

    def test_introduction_highlights_tts_local_deployment_voice_cloning_and_emotion_classification(self):
        keywords = ("文本转语音", "CPU/GPU 本地部署", "音色克隆", "情感分类")
        description = manifest()["description"]
        for keyword in keywords:
            self.assertIn(keyword, description)
        readme = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        # 项目介绍以“基于”开头，不依赖标题采用 Markdown 还是 HTML。
        intro = next(item for item in readme.split("\n\n") if item.startswith("基于 "))
        for keyword in keywords:
            self.assertIn(f"**{keyword}**", intro)
        features = re.search(r"^## 功能\n(.*?)^## ", readme, re.M | re.S).group(1)
        self.assertEqual(re.findall(r"^### (.+)$", features, re.M), [*keywords, "其他"])

    def test_docs_index_lists_every_document_with_its_title(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        documents = sorted(path for path in (ROOT / "docs").glob("*.md") if path.name != "README.md")
        self.assertGreaterEqual(len(documents), 9)
        for path in documents:
            title = re.search(r"^# (.+)$", path.read_text(encoding="utf-8"), re.M).group(1)
            self.assertIn(f"| [{title}](docs/{path.name}) |", readme)
            self.assertIn(f"| [{title}]({path.name}) |", index)

    def test_markdown_links_and_images_resolve(self):
        pages = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
        referenced = set()
        for page in pages:
            text = page.read_text(encoding="utf-8")
            targets = (re.findall(r"\]\(([^)\s]+)\)", text)
                       + re.findall(r'<img [^>]*src="([^"]+)"', text)
                       + re.findall(r'<a [^>]*href="([^"]+)"', text))
            for target in targets:
                if re.match(r"^(?:https?:|mailto:|#)", target):
                    continue
                resolved = (page.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"{page.relative_to(ROOT).as_posix()} 链接不存在：{target}")
                self.assertTrue(resolved.is_relative_to(ROOT), target)
                if resolved.suffix == ".png":
                    referenced.add(resolved.relative_to(ROOT).as_posix())
            for alt, target in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)\)", text):
                self.assertTrue(alt.strip(), f"{page.name} 中的图片 {target} 缺少说明文字")
        screenshots = {path.relative_to(ROOT).as_posix() for path in (ROOT / "docs" / "images").glob("*")}
        self.assertEqual(screenshots - referenced, set(), "docs/images 中没有未被引用的截图")
        self.assertIn("assets/maibot-microphone-chibi-256.png", referenced)
        # MaiBot 的插件页面只读取根目录 README.md；README 本身保持简短，详细说明放在 docs/。
        self.assertLess(len((ROOT / "README.md").read_text(encoding="utf-8")), 8000)

    def test_repository_has_no_machine_specific_paths(self):
        # 开源仓库不带开发机器上的个人路径：用户目录、本地 Conda 环境。
        pattern = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|" + "mini" + "forge|" + "ana" + "conda3", re.I)
        offenders = []
        for path in sorted(ROOT.rglob("*")):
            parts = set(path.relative_to(ROOT).parts)
            if not path.is_file() or parts & {"dist", "logs", ".git", "__pycache__", "images"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            offenders += [f"{path.relative_to(ROOT).as_posix()}:{text[:m.start()].count(chr(10)) + 1}"
                          for m in pattern.finditer(text)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
