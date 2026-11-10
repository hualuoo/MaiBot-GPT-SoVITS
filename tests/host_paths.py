"""定位本机 MaiBot 源码：环境变量 MAIBOT_ROOT，或与本项目并列的 MaiBot 文件夹。

需要导入 MaiBot 源码的测试在找不到 MaiBot 时会跳过。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
SKIP_REASON = "需要 MaiBot 源码：设置环境变量 MAIBOT_ROOT，或把 MaiBot 放在本项目旁边的 MaiBot 文件夹"


def _find_maibot() -> Path | None:
    value = os.environ.get("MAIBOT_ROOT", "").strip()
    path = Path(value) if value else PROJECT.parent / "MaiBot"
    return path.resolve() if (path / "bot.py").is_file() else None


MAIBOT_ROOT = _find_maibot()


def use_maibot() -> Path:
    """把 MaiBot 源码目录加入 sys.path 并返回它；找不到时抛出 RuntimeError。"""
    if MAIBOT_ROOT is None:
        raise RuntimeError(SKIP_REASON)
    if str(MAIBOT_ROOT) not in sys.path:
        sys.path.insert(0, str(MAIBOT_ROOT))
    return MAIBOT_ROOT


def installed_plugin(plugin_id: str) -> Path | None:
    """按 manifest 中的插件 ID 查找 MaiBot plugins 目录下已安装的插件。"""
    plugins = MAIBOT_ROOT / "plugins" if MAIBOT_ROOT is not None else None
    if plugins is None or not plugins.is_dir():
        return None
    for folder in sorted(plugins.iterdir()):
        try:
            if json.loads((folder / "_manifest.json").read_text(encoding="utf-8")).get("id") == plugin_id:
                return folder
        except (OSError, ValueError, AttributeError):
            continue
    return None
