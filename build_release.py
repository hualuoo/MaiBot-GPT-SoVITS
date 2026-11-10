"""生成可在 MaiBot WebUI「插件管理 → 从 ZIP 安装」导入的发行包。

用法（项目根目录）：
    python build_release.py                 # 输出 dist/MaiBot-GPT-SoVITS-<版本>.zip
    python build_release.py --verify-host <MaiBot 目录>

只打包运行所需文件和使用文档（docs/）；不包含 config.toml、data/、日志、测试、脚本或缓存。
ZIP 根目录直接放置 _manifest.json，符合 MaiBot zip_install 的目录约定。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent
FILES = [
    "_manifest.json", "plugin.py", "config_models.py", "tts_client.py", "setup_service.py",
    "native_config.py", "role_service.py", "role_library.py", "model_catalog.py",
    "wizard_progress.py", "auto_voice.py", "emotion_judge.py", "tts_log.py",
    "webui.json", "webui/setup.html", "assets/maibot-microphone-chibi-256.png",
    "config.example.toml", "requirements.txt", "README.md", "LICENSE",
]
# 与 MaiBot src/webui/routers/plugin/zip_install.py 的限制保持一致。
MAX_ZIP_BYTES = 100 * 1024 * 1024
# 使用文档随发行包附带，便于离线阅读。
DOC_PATTERNS = ("docs/*.md", "docs/images/*.png")
# 与 MaiBot src/webui/routers/plugin/icon_routes.py 的图标限制保持一致。
MAX_ICON_BYTES = 512 * 1024
ICON_SUFFIXES = {".jpg", ".jpeg", ".png", ".svg", ".webp"}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def manifest_icon(manifest: dict) -> str | None:
    """返回 _manifest.json 中 display.icon 声明的本地图标路径（未声明本地图标时为 None）。"""
    icon = (manifest.get("display") or {}).get("icon") or {}
    if icon.get("type") != "local":
        return None
    return PurePosixPath(str(icon.get("value", "")).strip().replace("\\", "/")).as_posix()


def release_files() -> list[str]:
    """发行包中的全部文件：FILES 加上 docs/ 中的文档与截图。"""
    docs = sorted({path.relative_to(ROOT).as_posix() for pattern in DOC_PATTERNS
                   for path in ROOT.glob(pattern) if path.is_file()})
    return FILES + [name for name in docs if name not in FILES]


def build_release(output_dir: Path | None = None) -> Path:
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    files = release_files()
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("发行文件缺失：" + "、".join(missing))
    icon = manifest_icon(manifest)
    if icon is not None:
        # 图标不在发行包里时 MaiBot 只能显示后备图标，构建时直接拒绝。
        if icon not in FILES:
            raise ValueError(f"_manifest.json 声明的插件图标 {icon} 不在 FILES 发行文件列表中")
        if PurePosixPath(icon).suffix.lower() not in ICON_SUFFIXES:
            raise ValueError("插件图标仅支持 jpg、jpeg、png、svg、webp")
        if (ROOT / icon).stat().st_size > MAX_ICON_BYTES:
            raise ValueError("插件图标超过 MaiBot 512 KB 上限")
    output_dir = output_dir or ROOT / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"MaiBot-GPT-SoVITS-{manifest['version']}.zip"
    temporary = target.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name in files:
            # 固定时间戳与普通文件权限：同一源码重复构建得到相同内容。
            info = ZipInfo(PurePosixPath(name).as_posix(), FIXED_TIME)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, (ROOT / name).read_bytes())
    if temporary.stat().st_size > MAX_ZIP_BYTES:
        temporary.unlink()
        raise ValueError("ZIP 超过 MaiBot 100 MB 上限")
    temporary.replace(target)
    return target


def verify_with_host(archive_path: Path, host: Path) -> str:
    """用 MaiBot 自身的 ZIP 路径检查与 Manifest 校验器验证，不安装插件。"""
    sys.path.insert(0, str(host))
    from src.plugin_runtime.runner.manifest_validator import ManifestValidator
    from src.webui.routers.plugin.zip_install import _archive_entries

    with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as temporary:
        entries = _archive_entries(archive)
        if PurePosixPath("_manifest.json") not in [path for _, path in entries]:
            raise ValueError("ZIP 根目录缺少 _manifest.json")
        archive.extractall(temporary)
        validator = ManifestValidator(log_errors=False)
        manifest = validator.load_from_plugin_path(Path(temporary))
        if manifest is None:
            raise ValueError("MaiBot Manifest 校验失败：" + "；".join(validator.errors))
        return f"{manifest.id} {manifest.version}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-host", type=Path, help="MaiBot 根目录；使用其 ZIP/Manifest 校验逻辑检查产物")
    args = parser.parse_args()
    target = build_release()
    print(target)
    if args.verify_host:
        print("MaiBot 校验通过：" + verify_with_host(target, args.verify_host))


if __name__ == "__main__":
    main()
