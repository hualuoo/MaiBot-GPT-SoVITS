"""角色模式 API：只读取 GPT-SoVITS 安装目录下 maibot_roles 中用户自行放置的角色文件；不提供网页上传。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
import uuid

from maibot_sdk import API

if __package__:
    from .role_library import RoleLibrary
    from .config_models import NO_LORA
    from .model_catalog import ROLES_DIRNAME
    from .tts_client import TTSError
else:
    from role_library import RoleLibrary
    from config_models import NO_LORA
    from model_catalog import ROLES_DIRNAME
    from tts_client import TTSError


class RoleService:
    def _init_roles(self) -> None:
        self._role_library = None
        self._role_library_root = None
        self._role_status = {"status": "idle", "message": "请选择角色和语气，读取JSON配置", "profile": None}

    @property
    def _role_status(self) -> dict:
        return self._role_status_value

    @_role_status.setter
    def _role_status(self, value: dict) -> None:
        # Older synthesis/cancel callers still assign the original three fields.
        value = dict(value)
        profile = value.get("profile")
        previous = getattr(self, "_role_status_value", {})
        profiles = value.get("profiles")
        if profiles is None:
            old = previous.get("profile")
            if profile and old and profile.get("role_id") == old.get("role_id"):
                profiles = dict(previous.get("profiles", {}))
            else:
                profiles = {}
            if profile:
                profiles[profile["emotion"]] = profile
        value["profiles"] = profiles
        value["emotions"] = list(value.get("emotions", profiles))
        value.setdefault("profile", None)
        self._role_status_value = value

    def _plugin_data_dir(self) -> Path:
        """插件自己的 data 目录只保存状态快照（向导进度、普通模式配置），角色文件放在 GPT-SoVITS 目录。"""
        return self._settings_path.parent / "data"

    def _role_data_dir(self) -> Path:
        """角色文件夹固定在 GPT-SoVITS 安装目录（第二步保存的本机模型目录）下的 maibot_roles。

        合成时由 GPT-SoVITS 服务直接读取角色的模型与参考音频，放在它的目录里才能保证读得到；
        不读取插件 data 或其他目录。
        """
        root = (self.config.models.root or "").strip()
        if not root:
            raise ValueError(f"请先在第二步填写并保存本机模型目录（GPT-SoVITS 安装目录），角色放在其中的 {ROLES_DIRNAME} 文件夹")
        return Path(os.path.abspath(root)) / ROLES_DIRNAME

    def _roles(self, root: Path | None = None) -> RoleLibrary:
        own = self._role_data_dir()
        if root is not None and root != own:
            raise ValueError(f"角色只从 GPT-SoVITS 目录下的 {ROLES_DIRNAME} 读取")
        root = own
        if self._role_library is None or self._role_library_root != root:
            self._role_library = RoleLibrary(root)
            self._role_library_root = root
        return self._role_library

    def _original_snapshot_path(self) -> Path:
        return self._plugin_data_dir() / ".state" / "original-config.json"

    @staticmethod
    def _write_snapshot(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_snapshot(self) -> dict:
        path = self._original_snapshot_path()
        if not path.exists():
            return {}
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("原版模式配置快照超过上限")
        result = json.loads(raw.decode("utf-8"))
        if not isinstance(result, dict) or set(result) != {"models", "voice"}:
            raise ValueError("原版模式配置快照损坏")
        return result

    async def _capture_original(self) -> None:
        if self.config.role.mode == "original":
            value = {"models": self.config.models.model_dump(), "voice": self.config.voice.model_dump()}
            await asyncio.to_thread(self._write_snapshot, self._original_snapshot_path(), value)

    @API("setup_roles", description="扫描 GPT-SoVITS 目录下 maibot_roles 中的角色 JSON 清单")
    async def setup_roles(self) -> dict:
        def scan():
            try:
                root = self._role_data_dir()
            except ValueError as exc:
                return {"roles": [], "errors": [str(exc)], "data_dir": "", "scanned_paths": [],
                        "message": f"角色列表为空：{exc}"}
            if not root.exists():
                reason = (f"角色文件夹不存在：{root.as_posix()}；请在 GPT-SoVITS 安装目录下新建 {ROLES_DIRNAME} 文件夹，"
                          "再放入角色文件夹")
                return {"roles": [], "errors": [reason], "data_dir": root.as_posix(),
                        "scanned_paths": [root.as_posix()], "message": f"角色列表为空：{reason}"}
            result = self._roles(root).list_roles()
            result.update(data_dir=root.as_posix(), scanned_paths=[root.as_posix()])
            if result["roles"]:
                result["message"] = f"已读取 GPT-SoVITS 目录下 {ROLES_DIRNAME} 中的角色清单；不扫描插件 data 或其他目录"
            else:
                reason = "；".join(result["errors"][:3]) or f"未发现角色，每个角色文件夹应包含 {ROLES_DIRNAME}/<角色>/role.json"
                result["message"] = f"角色列表为空：{reason}；角色目录：{root.as_posix()}"
            return result
        result = await asyncio.to_thread(scan)
        while len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 450 * 1024:
            if result["errors"]:
                result["errors"].pop()
            elif result["roles"]:
                result["roles"].pop()
            else:
                break
        return result

    @API("setup_set_mode", description="切换角色模式或原版手动模式")
    async def setup_set_mode(self, mode: str) -> dict:
        try:
            self._require_setup()
            if mode not in ("original", "role"):
                raise TTSError("未知模式，请选择角色模式或原版模式")
            if self._busy:
                raise TTSError("任务正在运行，结束后再切换模式")
            if self.config.role.mode == mode:
                return {"success": True, "message": "模式保持不变"}
            await self._capture_original()
            updates = {"role": {"mode": mode}}
            if mode == "original":
                updates.update(await asyncio.to_thread(self._read_snapshot))
            await self._save_setup(updates)
            self._loaded_signature = None
            self._preview_audio = b""
            return {"success": True, "message": "已切换为角色模式，请读取角色JSON" if mode == "role" else "已恢复原版手动模式配置，请按需加载模型"}
        except (TTSError, ValueError, OSError) as exc:
            return {"success": False, "message": str(exc)}

    @API("setup_prepare_role", description="校验角色JSON资源并自动套用模型与参考音频")
    async def setup_prepare_role(self, role_id: str, emotion: str, emotions: list[str] | None = None) -> dict:
        try:
            self._require_setup()
            if self._busy:
                raise TTSError("已有任务正在运行，请稍后读取角色")
            if not isinstance(role_id, str) or not role_id or len(role_id) > 240:
                raise TTSError("请选择有效角色清单")
            selected = self._validate_role_emotions(emotion, emotions)
            self._busy = True
            self._preview_audio = b""
            self._role_status = {"status": "running", "message": "正在读取 role.json 并校验角色文件与底模…", "profile": None}
            self._job = {"id": uuid.uuid4().hex, "kind": "prepare_role", "status": "running",
                         "message": self._role_status["message"], "audio_bytes": 0, "duration_seconds": 0}
            self._job_task = asyncio.create_task(self._run_prepare_role(role_id, emotion, selected), name="gpt-sovits-role")
            self._tasks.add(self._job_task)
            self._job_task.add_done_callback(self._finish_setup_task)
            return {"success": True, "message": self._job["message"], "job": dict(self._job)}
        except (TTSError, ValueError) as exc:
            return {"success": False, "message": str(exc)}

    @staticmethod
    def _validate_role_emotions(emotion: str, emotions: list[str] | None) -> list[str]:
        if not isinstance(emotion, str) or not emotion.strip() or len(emotion) > 80:
            raise TTSError("请选择有效默认试听语气（最多80字符）")
        selected = [emotion] if emotions is None else emotions
        if not isinstance(selected, list) or not 1 <= len(selected) <= 50:
            raise TTSError("请勾选1至50个角色语气")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 80 for item in selected):
            raise TTSError("角色语气必须为非空字符串，每项最多80字符")
        if len(set(selected)) != len(selected):
            raise TTSError("勾选的角色语气不能重复")
        if emotion not in selected:
            raise TTSError("默认试听语气必须在勾选语气中")
        return list(selected)

    async def _run_prepare_role(self, role_id: str, emotion: str, emotions: list[str] | None = None) -> None:
        job, started = self._job, time.monotonic()
        try:
            selected = self._validate_role_emotions(emotion, emotions)
            cfg = self.config.model_copy(deep=True)
            library = await asyncio.to_thread(self._roles)
            profiles, failures = {}, []
            for item in selected:
                try:
                    profile = await asyncio.to_thread(library.resolve, role_id, item, cfg.models.root)
                    profile["voice"]["aux_ref_audio_paths"] = []
                    profiles[item] = profile
                except (ValueError, OSError, TTSError) as exc:
                    failures.append(f"{item}：{exc}")
            if failures:
                raise TTSError("；".join(failures))
            versions = {profile["models"]["base_model"] for profile in profiles.values()}
            if len(versions) != 1:
                raise TTSError("同一角色所选语气的底模版本必须一致：" + "、".join(sorted(versions)))
            version = next(iter(versions))
            if cfg.models.base_model != "custom" and cfg.models.base_model != version:
                raise TTSError(f"第二步所选底模{cfg.models.base_model}与角色 role.json 的底模{version}不匹配，请确认后重新选择；未覆盖第二步底模")
            if (cfg.api.base_url, cfg.models.root, cfg.models.base_model) != (
                    self.config.api.base_url, self.config.models.root, self.config.models.base_model):
                raise TTSError("校验期间接口、模型目录或底模已改变，请重新读取角色")
            profile = profiles[emotion]
            model_values = dict(profile["models"])
            model_values["lora_weights"] = model_values.get("lora_weights") or NO_LORA
            # Nothing touches cfg/config.toml until every selected profile passes.
            await self._capture_original()
            await self._save_setup({"models": model_values, "voice": profile["voice"],
                                   "role": {"mode": "role", "role_id": role_id, "emotion": emotion,
                                            "emotions": selected}}, allow_busy=True)
            checked = len({asset["path"] for item in profiles.values() for asset in item["assets"]})
            message = (f"角色{profile['role_name']}的{len(selected)}个情感已就绪（{version}）；默认{profile['emotion_label']}，"
                       f"已按 role.json 校验 {checked} 个文件。模型尚未加载。")
            self._role_status = {"status": "success", "message": message, "profile": profile,
                                 "profiles": profiles, "emotions": selected}
            await self._mark_progress(3)
            self._loaded_signature = None
            job.update(status="success", message=message)
        except asyncio.CancelledError:
            self._role_status = {"status": "error", "message": "角色读取已取消；没有确认配置应用成功", "profile": None}
            job.update(status="cancelled", message=self._role_status["message"])
        except Exception as exc:
            message = f"角色读取失败：{exc}"
            self._role_status = {"status": "error", "message": message, "profile": None}
            job.update(status="error", message=message)
        finally:
            job["elapsed_seconds"] = round(time.monotonic() - started, 2)
            self._busy = False
