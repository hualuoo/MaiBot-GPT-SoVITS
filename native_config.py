"""原生配置页适配：只读 Schema 索引与显式异步刷新，不依赖 Host 内部。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import time
import tomllib
import uuid

from maibot_sdk import ConfigWidget, ConfigWidgetState

if __package__:
    from .config_models import NO_LORA, UNSELECTED, TTSConfig
    from .model_catalog import scan_models
    from .role_library import RoleLibrary
else:
    from config_models import NO_LORA, UNSELECTED, TTSConfig
    from model_catalog import scan_models
    from role_library import RoleLibrary


INDEX_FILENAME = "native-model-index.json"
_INDEX_LIMIT = 1024 * 1024
_CONFIG_LIMIT = 256 * 1024
_ROLE_CACHE_TTL = 5.0
ROLE_UNSELECTED = "（请选择角色）"
EMOTION_UNSELECTED = "（请选择情感）"
_ROLE_HINT = "先选择角色和情感并保存，再点击原生“读取已保存角色”按钮；该按钮只使用已保存配置并启动后台校验，不会提交未保存的草稿。即使当前是original原版模式，也可以先选择角色和情感保存。切换角色后请保存并重新进入本页以更新情感选项。"
_SAVED_HINT = "原生按钮仅使用已保存配置，不会提交本页未保存的草稿；请先保存，再测试或加载。"
_REFRESH_HINT = "修改目录后请先保存，再点击“刷新模型列表”；刷新后请重新进入本配置页更新下拉选项。"


def _safe_role_id(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 240:
        return False
    parts = value.split("/")
    return (len(parts) == 2 and not parts[0].startswith((".", "_")) and parts[1].lower() == "role.json"
            and all(part and part not in (".", "..") and not part.endswith((" ", "."))
                    and not any(ord(char) < 32 or char in '<>:"\\|?*' for char in part)
                    and not re.fullmatch(r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])",
                                         part.split(".", 1)[0].rstrip(" ."), re.I)
                    for part in parts))


def _directory_key(root: str) -> str:
    # 仅作词法比较，不 resolve、不检查目录，不触发扫描或网络访问。
    return os.path.normcase(os.path.normpath(root.strip().replace("\\", "/"))) if root.strip() else ""


def _catalog_data(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("模型索引不是有效对象")
    result = {"note": str(value.get("note", ""))[:2000]}
    for name in ("base_models", "gpt_weights", "sovits_weights", "lora_weights"):
        items = value.get(name, [])
        if not isinstance(items, list):
            raise ValueError("模型索引列表格式无效")
        key, limit = ("id", 80) if name == "base_models" else ("path", 1024)
        result[name] = []
        seen = set()
        for item in items[:200]:
            text = item.get(key) if isinstance(item, dict) else item
            if isinstance(text, str) and text.strip() and "\x00" not in text and len(text) <= limit:
                text = text.strip()
                if text not in seen:
                    result[name].append({key: text})
                    seen.add(text)
    return result


def _write_index(path: Path, root: str, catalog: dict) -> None:
    content = json.dumps({"version": 1, "root": root,
                          "generated_at": time.time(), "catalog": catalog},
                         ensure_ascii=False, separators=(",", ":"))
    if len(content.encode("utf-8")) > _INDEX_LIMIT:
        raise ValueError("模型索引超过大小上限")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class NativeConfigMixin:
    """放在 MaiBotPlugin 前；在 _init_setup(settings_path) 后调用 _init_native()。"""

    config_widgets = [
        ConfigWidget(id="api_status", kind="status", label="接口连接状态（已保存配置）"),
        ConfigWidget(id="test_saved_api", kind="action", label="测试已保存接口"),
        ConfigWidget(id="model_status", kind="status", label="模型与列表状态"),
        ConfigWidget(id="refresh_models", kind="action", label="刷新模型列表"),
        ConfigWidget(id="prepare_saved_role", kind="action", label="读取已保存角色"),
        ConfigWidget(id="load_saved_models", kind="action", label="加载已保存模型",
                     confirmation="将按已保存配置切换 GPT-SoVITS 共享服务的全局权重，可能影响其他调用方。确认已保存并继续？"),
    ]

    def _init_native(self) -> None:
        self._native_catalog = None
        self._native_catalog_key = None
        self._native_index_checked_key = None
        self._native_catalog_note = "尚未读取模型索引。" + _REFRESH_HINT
        self._native_catalog_tone = "neutral"
        self._native_refreshing = False
        self._native_action_busy = False
        self._native_last_error = ""
        self._native_role_cache = None
        self._native_role_cache_key = None
        self._native_role_cache_until = 0.0

    def _native_config(self, *, read_disk: bool = False) -> TTSConfig:
        # 冷配置检查没有 ctx/config 注入；状态轮询使用 read_disk=False，保证无 I/O。
        current = getattr(self, "_plugin_config_instance", None)
        if isinstance(current, TTSConfig):
            return current
        if read_disk:
            path = getattr(self, "_settings_path", None)
            if path is not None:
                try:
                    with Path(path).open("rb") as stream:
                        raw = stream.read(_CONFIG_LIMIT + 1)
                    if len(raw) > _CONFIG_LIMIT:
                        raise ValueError("配置文件超过只读检查大小上限")
                    return TTSConfig.model_validate(tomllib.loads(raw.decode("utf-8-sig")))
                except (OSError, ValueError, TypeError):
                    pass
        return TTSConfig()

    def _native_read_catalog(self, cfg: TTSConfig) -> tuple[dict, str]:
        key = _directory_key(cfg.models.root)
        if getattr(self, "_native_catalog_key", None) == key and self._native_catalog is not None:
            return self._native_catalog, self._native_catalog_note
        if getattr(self, "_native_index_checked_key", None) == key:
            return _catalog_data({}), self._native_catalog_note
        self._native_index_checked_key = key
        note = "尚无与当前目录匹配的模型索引。" + _REFRESH_HINT
        path = getattr(self, "_settings_path", None)
        if path is not None:
            try:
                with (Path(path).parent / INDEX_FILENAME).open("rb") as stream:
                    raw = stream.read(_INDEX_LIMIT + 1)
                if len(raw) > _INDEX_LIMIT:
                    raise ValueError("模型索引超过大小上限")
                data = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(data, dict) or data.get("version") != 1:
                    raise ValueError("模型索引版本无效")
                if not isinstance(data.get("root"), str):
                    raise ValueError("模型索引目录无效")
                if _directory_key(data["root"]) != key:
                    note = "模型目录已变更，旧索引未用于当前下拉列表。" + _REFRESH_HINT
                else:
                    catalog = _catalog_data(data.get("catalog"))
                    self._native_catalog = catalog
                    self._native_catalog_key = key
                    self._native_catalog_note = "已读取本地模型索引（不会在打开配置页时扫描）。" + catalog["note"] + _REFRESH_HINT
                    return catalog, self._native_catalog_note
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, RecursionError):
                note = "模型索引不存在、损坏或无法读取，已保留当前配置值。" + _REFRESH_HINT
        self._native_catalog_note = note
        return _catalog_data({}), note

    def _native_read_roles(self, cfg: TTSConfig) -> tuple[dict, str]:
        """只读 GPT-SoVITS 安装目录下 maibot_roles 的有界JSON清单；不扫描或校验任何模型资产。

        cfg 来自调用方：MaiBot 冷读取原生表单时插件配置尚未注入，只能使用从 TOML 读出的配置。
        """
        model_root = (cfg.models.root or "").strip()
        if not model_root:
            return ({"roles": [], "errors": ["尚未填写本机模型目录"]},
                    "角色目录：GPT-SoVITS 安装目录下的 maibot_roles。请先在设置页第二步填写并保存本机模型目录。")
        root = Path(os.path.abspath(model_root)) / "maibot_roles"
        key = str(root)
        now = time.monotonic()
        if (getattr(self, "_native_role_cache_key", None) == key
                and getattr(self, "_native_role_cache", None) is not None
                and now < getattr(self, "_native_role_cache_until", 0.0)):
            return self._native_role_cache
        try:
            result = (RoleLibrary(root).list_roles() if root.exists()
                      else {"roles": [], "errors": ["角色文件夹不存在：" + root.as_posix()]})
        except (OSError, ValueError, TypeError, RuntimeError, RecursionError) as exc:
            result = {"roles": [], "errors": [str(exc) or "角色JSON读取失败"]}
        note = "角色目录：" + root.as_posix() + "。只读取 maibot_roles/<角色>/role.json，不读取插件 data 或其他目录，不扫描或校验模型资产。"
        if not result["roles"]:
            note += "未发现可用角色JSON；请把包含 role.json 的角色文件夹放到上述目录。"
        if result["errors"]:
            note += "读取错误：" + "；".join(str(error) for error in result["errors"][:5])
        self._native_role_cache_key = key
        self._native_role_cache_until = now + _ROLE_CACHE_TTL
        self._native_role_cache = (result, note)
        return result, note

    def _native_role_schema(self, section: dict, cfg: TTSConfig) -> None:
        result, note = self._native_read_roles(cfg)
        fields = section.get("fields", {})
        roles = result["roles"]
        options = [ROLE_UNSELECTED] + [role["id"] for role in roles]
        if _safe_role_id(cfg.role.role_id):
            options.append(cfg.role.role_id)
        selected = next((role for role in roles if role["id"] == cfg.role.role_id), None)
        if selected is None and cfg.role.role_id.strip() in ("", ROLE_UNSELECTED) and len(roles) == 1:
            selected = roles[0]
            note += "当前尚未选择角色；情感选项为唯一角色的预览，请先选择角色和情感并保存。"
        emotions = [emotion["id"] for emotion in selected["emotions"]] if selected else []
        if "role_id" in fields:
            fields["role_id"].update(type="select", ui_type="select", choices=list(dict.fromkeys(options)),
                                     default=cfg.role.role_id, placeholder="请选择角色", hint=note + _ROLE_HINT)
        for name in ("emotion", "emotions"):
            if name in fields:
                fields[name].update(type="select", ui_type="select", choices=emotions or [EMOTION_UNSELECTED],
                                    default=getattr(cfg.role, name), multiple=name == "emotions",
                                    placeholder="请选择情感", hint=note + _ROLE_HINT)
        for name, field in fields.items():
            # original模式下也必须能先挑选、保存角色，不依赖模式条件隐藏字段。
            field.pop("visible_when", None)
            field.pop("show_when", None)
            field["hidden"] = False
            if _ROLE_HINT not in (field.get("hint") or ""):
                field["hint"] = ((field.get("hint") or "") + " " + _ROLE_HINT).strip()

    def get_webui_config_schema(self, *, plugin_id: str = "", plugin_name: str = "",
                                plugin_version: str = "", plugin_description: str = "",
                                plugin_author: str = "") -> dict:
        schema = super().get_webui_config_schema(
            plugin_id=plugin_id, plugin_name=plugin_name, plugin_version=plugin_version,
            plugin_description=plugin_description, plugin_author=plugin_author)
        cfg = self._native_config(read_disk=True)
        catalog, note = self._native_read_catalog(cfg)
        fields = schema.get("sections", {}).get("models", {}).get("fields", {})
        choices = {"base_model": ["custom"] + [item["id"] for item in catalog["base_models"]],
                   "gpt_weights": [UNSELECTED] + [item["path"] for item in catalog["gpt_weights"]],
                   "sovits_weights": [UNSELECTED] + [item["path"] for item in catalog["sovits_weights"]],
                   "lora_weights": [NO_LORA] + [item["path"] for item in catalog["lora_weights"]]}
        for name, options in choices.items():
            if name not in fields:
                continue
            current = getattr(cfg.models, name)
            if isinstance(current, str) and current.strip():
                options.append(current)
            fields[name].update(type="select", ui_type="select", choices=list(dict.fromkeys(options)),
                                placeholder="请选择模型", hint=note + _SAVED_HINT)
        for section in ("api", "models"):
            for field in schema.get("sections", {}).get(section, {}).get("fields", {}).values():
                if _SAVED_HINT not in (field.get("hint") or ""):
                    field["hint"] = ((field.get("hint") or "") + " " + _SAVED_HINT).strip()
        # 两个呈现节仍写回同一个models配置对象；SectionRenderer以name作为真实保存键。
        sections = schema.get("sections", {})
        if "api" in sections:
            sections["api"].update(name="api", title="第一步 · 接口配置与测试", order=1)
        if "models" in sections:
            ordinary = dict(sections["models"])
            ordinary.update(name="models", title="第三步 · 普通模式（单一LoRA或不使用）", order=4,
                            fields={name: fields[name] for name in ("gpt_weights", "sovits_weights", "lora_weights") if name in fields})
            sections["models"].update(name="models", title="第二步 · 只选择底模", order=2,
                                      fields={name: fields[name] for name in ("root", "base_model") if name in fields})
            sections["original_weights"] = ordinary
        if "role" in sections:
            sections["role"].update(name="role", title="第三步 · 角色模式与情感", order=3)
            self._native_role_schema(sections["role"], cfg)
        if "voice" in sections:
            sections["voice"].update(name="voice", title="第三步 · 普通模式参考音频与推理参数", order=5)
        if "behavior" in sections:
            sections["behavior"]["order"] = 6
        # Dashboard 支持 layout.editor_page 时，插件配置页直接打开五步设置页；不支持时忽略此字段。
        schema.setdefault("layout", {})["editor_page"] = "setup"
        return schema

    async def refresh_native_catalog(self) -> dict:
        """显式刷新/启动预热可调用；失败返回中文结果，不使插件 on_load 失败。"""
        if getattr(self, "_native_refreshing", False):
            return {"success": False, "message": "模型列表正在刷新，请稍后重试"}
        self._native_role_cache = None
        self._native_role_cache_until = 0.0
        self._native_refreshing = True
        self._native_catalog_tone = "neutral"
        self._native_catalog_note = "正在扫描已保存的本地模型目录…"
        cfg = self._native_config().model_copy(deep=True)
        root = cfg.models.root
        key = _directory_key(root)
        try:
            catalog = _catalog_data(await asyncio.to_thread(scan_models, root))
            current = self._native_config()
            if _directory_key(current.models.root) != key:
                raise ValueError("扫描期间模型目录已改变，请按新配置重新刷新")
            self._native_catalog, self._native_catalog_key = catalog, key
            self._native_index_checked_key = key
            path = Path(self._settings_path).parent / INDEX_FILENAME
            try:
                await asyncio.to_thread(_write_index, path, root, catalog)
            except (OSError, ValueError) as exc:
                self._native_catalog_tone = "warning"
                self._native_catalog_note = "模型列表已在内存更新，但索引未能持久化：" + str(exc) + "。" + _REFRESH_HINT
                return {"success": False, "message": self._native_catalog_note}
            self._native_catalog_tone = "success"
            self._native_catalog_note = "模型索引已更新。" + catalog["note"] + _REFRESH_HINT
            return {"success": True, "message": self._native_catalog_note}
        except asyncio.CancelledError:
            self._native_catalog_tone = "warning"
            self._native_catalog_note = "模型列表刷新已取消。" + _REFRESH_HINT
            raise
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self._native_catalog_tone = "error"
            self._native_catalog_note = "刷新模型列表失败：" + str(exc) + "。" + _REFRESH_HINT
            return {"success": False, "message": self._native_catalog_note}
        finally:
            self._native_refreshing = False

    async def get_config_widget_state(self) -> dict[str, ConfigWidgetState]:
        """仅内存状态：禁止文件读取、模型扫描、HTTP 请求与配置写入。"""
        cfg = self._native_config()
        enabled = bool(getattr(self, "_active", False) and cfg.plugin.enabled)
        busy = bool(getattr(self, "_busy", False) or getattr(self, "_native_refreshing", False)
                    or getattr(self, "_native_action_busy", False))
        disabled = not enabled or busy
        connection = getattr(self, "_connection", {})
        model = getattr(self, "_model_status", {})
        job = getattr(self, "_job", None)
        if isinstance(job, dict) and job.get("kind") in ("load_models", "prepare_role"):
            model = job
        tones = {"success": "success", "error": "error", "cancelled": "warning"}
        prefix = "插件尚未启用，启用后可使用原生按钮。" if not enabled else ""
        api_text = prefix + str(connection.get("message", "尚未测试接口")) + "\n" + _SAVED_HINT
        model_text = prefix + str(model.get("message", "尚未加载模型，服务端当前权重未知"))
        model_text += "\n" + getattr(self, "_native_catalog_note", "尚未读取模型索引。" + _REFRESH_HINT)
        key = _directory_key(cfg.models.root)
        cached_key = getattr(self, "_native_catalog_key", None)
        if cached_key is not None and cached_key != key:
            model_text += "\n模型目录已改变，请保存后刷新列表。"
        error = getattr(self, "_native_last_error", "")
        if error:
            model_text += "\n" + error
        model_text += "\n" + _SAVED_HINT
        return {
            "api_status": ConfigWidgetState(text=api_text[:2000], tone=tones.get(connection.get("status"), "neutral")),
            "test_saved_api": ConfigWidgetState(disabled=disabled),
            "model_status": ConfigWidgetState(text=model_text[:2000], tone="error" if error else tones.get(
                model.get("status"), getattr(self, "_native_catalog_tone", "neutral"))),
            "refresh_models": ConfigWidgetState(disabled=disabled),
            "load_saved_models": ConfigWidgetState(disabled=disabled),
            "prepare_saved_role": ConfigWidgetState(disabled=disabled),
        }

    async def _native_action(self, action_id: str, cfg: TTSConfig) -> dict:
        if action_id == "test_saved_api":
            return await self.setup_test_api()
        if action_id == "load_saved_models":
            return await self.setup_start_job(kind="load_models")
        if action_id == "prepare_saved_role":
            role = cfg.role
            if not _safe_role_id(role.role_id):
                raise ValueError("请先选择有效角色和情感并保存，再读取已保存角色")
            if (not role.emotion.strip() or role.emotion == EMOTION_UNSELECTED
                    or any(value == EMOTION_UNSELECTED or not value.strip() for value in role.emotions)):
                raise ValueError("请先选择有效情感并保存，再读取已保存角色")
            # setup_prepare_role仅启动后台任务即返回；此处绝不同步resolve/stat/hash资产。
            return await self.setup_prepare_role(role.role_id, role.emotion, role.emotions or [role.emotion])
        if action_id == "refresh_models":
            return await self.refresh_native_catalog()
        raise ValueError("未知的原生配置页操作")

    async def on_config_widget_action(self, action_id: str) -> None:
        if action_id not in {"test_saved_api", "refresh_models", "load_saved_models", "prepare_saved_role"}:
            raise ValueError("未知的原生配置页操作")
        cfg = self._native_config()
        if not getattr(self, "_active", False) or not cfg.plugin.enabled:
            raise ValueError("请先在 MaiBot 中启用插件")
        if (getattr(self, "_busy", False) or getattr(self, "_native_refreshing", False)
                or getattr(self, "_native_action_busy", False)):
            raise ValueError("已有任务正在运行，请等待结束后重试")
        self._native_action_busy = True
        self._native_last_error = ""
        try:
            result = await self._native_action(action_id, cfg)
            if not isinstance(result, dict) or result.get("success") is not True:
                message = result.get("message", "原生配置页操作失败") if isinstance(result, dict) else "原生配置页操作失败"
                raise ValueError(str(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._native_last_error = str(exc) or "原生配置页操作失败"
            raise ValueError(self._native_last_error) from exc
        finally:
            self._native_action_busy = False
