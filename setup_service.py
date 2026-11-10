"""中文设置向导后端；仅通过已声明的 MaiBot WebUI API 调用。"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
import wave
from pathlib import Path

import tomlkit
from maibot_sdk import API

if __package__:
    from . import tts_log
    from .model_catalog import scan_models, scan_reference_audios, inspect_sovits_file
    from .config_models import NO_LORA, UNSELECTED
    from .tts_client import GPTSoVITSClient, TTSError
    from .wizard_progress import WizardProgress
else:
    import tts_log
    from model_catalog import scan_models, scan_reference_audios, inspect_sovits_file
    from config_models import NO_LORA, UNSELECTED
    from tts_client import GPTSoVITSClient, TTSError
    from wizard_progress import WizardProgress


class SetupService:
    """混入插件，命令合成和设置任务共用单并发标志。"""

    def _init_setup(self, settings_path: Path) -> None:
        self._settings_path = settings_path
        self._settings_lock = asyncio.Lock()
        self._connection = {"status": "idle", "message": "尚未测试接口"}
        self._model_status = {"status": "idle", "message": "尚未通过本页加载模型；服务端当前权重未知"}
        self._loaded_signature = None
        self._known_api_url = None
        self._known_model_signature = None
        self._ensure_connection_task = None
        self._ensure_connection_url = None
        self._connection_generation = 0
        self._connection_checked_at = 0.0
        self._job = None
        self._job_task = None
        self._preview_audio = b""

    def _progress_manager(self):
        # 测试/冷实例可改变settings_path；进度与实际配置文件同根且不依赖ctx。
        return WizardProgress(self._settings_path)

    async def _progress_state(self) -> dict:
        cfg = self.config.model_copy(deep=True)
        return await asyncio.to_thread(self._progress_manager().compute, cfg,
            dict(self._connection), dict(self._role_status), dict(self._job) if self._job else None)

    async def _mark_progress(self, step: int) -> None:
        try:
            await asyncio.to_thread(self._progress_manager().mark, step, self.config.model_copy(deep=True))
        except (OSError, ValueError):
            self._get_logger().warning("配置已保存，但未能记录配置进度；重新打开时将按保存值检查")

    @staticmethod
    def _model_signature(cfg) -> tuple:
        return (cfg.api.base_url.rstrip("/"), cfg.models.base_model,
                cfg.models.gpt_weights, cfg.models.sovits_weights,
                SetupService._selected_path(cfg.models.lora_weights),
                cfg.role.mode, cfg.role.role_id if cfg.role.mode == "role" else "",
                cfg.role.emotion if cfg.role.mode == "role" else "",
                tuple(cfg.role.emotions) if cfg.role.mode == "role" else ())

    @staticmethod
    def _selected_path(value: str) -> str:
        value = value.strip()
        return "" if value in (NO_LORA, UNSELECTED) else value

    def _original_weights_configured(self, cfg) -> bool:
        return cfg.models.base_model != "custom" or bool(
            self._selected_path(cfg.models.gpt_weights) and self._selected_path(cfg.models.sovits_weights))

    def _loads_on_demand(self, cfg) -> bool:
        """角色模式，或普通模式开启自动语音且已配置权重时，合成前会自动加载所选权重。"""
        return cfg.role.mode == "role" or (cfg.auto_voice.enabled and self._original_weights_configured(cfg))

    def _invalidate_setup(self, old, new) -> None:
        old_url = self._known_api_url or old.api.base_url.rstrip("/")
        old_signature = self._known_model_signature or self._model_signature(old)
        self._known_api_url = new.api.base_url.rstrip("/")
        self._known_model_signature = self._model_signature(new)
        if old_url != self._known_api_url:
            self._connection_generation += 1
            self._connection_checked_at = 0.0
            self._connection = {"status": "idle", "message": "接口地址已变更，请重新测试连接"}
        if old_signature != self._known_model_signature:
            self._loaded_signature = None
            self._model_status = {"status": "idle", "message": "模型选择已变更，请重新加载并测试"}
            self._preview_audio = b""
            if self._job and self._job["status"] != "running":
                self._job = None

    def _require_setup(self) -> None:
        if not self._active or not self.config.plugin.enabled:
            raise TTSError("请先在 MaiBot 中启用插件")

    async def _save_setup(self, updates: dict, *, allow_busy: bool = False) -> None:
        """原子更新唯一 config.toml，保留注释及不相关配置。"""
        async with self._settings_lock:
            self._require_setup()
            if self._busy and not allow_busy:
                raise TTSError("任务正在运行，请等待结束后保存设置")
            old = self.config.model_copy(deep=True)
            # Merge the latest disk contents so an unrelated native config edit is preserved.
            path = self._settings_path
            if path.exists():
                # 记事本等编辑器可能写入 UTF-8 BOM；tomlkit 不接受 BOM，回写时去掉。
                document = tomlkit.parse(path.read_text(encoding="utf-8-sig"))
                raw = document.unwrap()
                data, _ = self.normalize_plugin_config(raw)
            else:
                document = tomlkit.document()
                data = old.model_dump(mode="python")
            for section, values in updates.items():
                data.setdefault(section, {}).update(values)
            validated = self.config_model.model_validate(data)
            data = validated.model_dump(mode="python")
            # Persist all default sections on first save, targeted fields thereafter.
            for section, values in data.items():
                if section not in document:
                    document[section] = values
                elif section in updates:
                    for key, value in updates[section].items():
                        document[section][key] = data[section][key]
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(tomlkit.dumps(document))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            self.set_plugin_config(data)
            self._last_sent.clear()
            self._invalidate_setup(old, self.config)

    @API("setup_state", description="读取中文设置向导状态")
    async def setup_state(self) -> dict:
        # 读取设置不执行推理或切换；即使插件暂未启用，也应允许查看配置。
        return {"config": self.config.model_dump(mode="json"),
                "connection": dict(self._connection), "model_status": dict(self._model_status),
                "job": dict(self._job) if self._job else None,
                "role_status": dict(self._role_status), "progress": await self._progress_state(),
                "auto_stats": dict(self._auto_stats),
                "emotion_judge": dict(getattr(self, "_emotion_last", {}) or {})}

    @staticmethod
    def _saved_connection_configured(progress: dict) -> bool:
        # Saved navigation is evidence of configuration, never evidence of liveness.
        return (1 in progress.get("completed_steps", []) or
                bool(progress.get("markers", {}).get("1")) or
                type(progress.get("next_step")) is int and 2 <= progress["next_step"] <= 5)

    def _setup_connection_valid(self, url: str) -> bool:
        checked_url = self._connection.get("checked_url", self._known_api_url)
        checked_at = self._connection_checked_at
        return (self._connection.get("status") == "success" and checked_url == url and
                (not checked_at or 0 <= time.monotonic() - checked_at < 300))

    def _mark_setup_connected(self, url: str, message: str = "接口连接成功，已确认 POST /tts") -> bool:
        """Record a real successful probe/TTS for this saved endpoint, not a save."""
        url = url.rstrip("/")
        if not self._active or not self.config.plugin.enabled or url != self.config.api.base_url.rstrip("/"):
            return False
        self._connection = {"status": "success", "message": message, "checked_url": url}
        self._connection_checked_at = time.monotonic()
        return True

    def _connection_response(self, success: bool, message: str = "") -> dict:
        return {"success": success, "message": message or self._connection["message"],
                "connection": dict(self._connection)}

    async def _cancel_setup_connection(self) -> None:
        """Unload hook: invalidate late results and release the shared probe task."""
        self._connection_generation += 1
        task = self._ensure_connection_task
        was_running = task and not task.done()
        if was_running:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if was_running:
            self._busy = False
        self._ensure_connection_task = None
        self._ensure_connection_url = None
        self._connection_checked_at = 0.0
        self._connection = {"status": "idle", "message": "连接确认已结束，重新进入时自动确认"}

    async def _run_ensure_connection(self, url: str, generation: int) -> dict:
        try:
            # The host permits 10s; cap the entire read-only network probe below 4s.
            await asyncio.wait_for(GPTSoVITSClient(url, timeout=3.5).probe(), timeout=3.5)
            if (generation != self._connection_generation or not self._active or
                    not self.config.plugin.enabled or url != self.config.api.base_url.rstrip("/")):
                return self._connection_response(False, "确认期间接口地址或插件状态已改变，旧连接结果已忽略，请重试")
            self._mark_setup_connected(url, "已自动确认保存的接口连接，POST /tts 可用；未加载或切换模型")
            return self._connection_response(True)
        except asyncio.CancelledError:
            if generation == self._connection_generation and url == self.config.api.base_url.rstrip("/"):
                self._connection = {"status": "idle", "message": "自动连接确认已取消", "checked_url": url}
            raise
        except Exception as exc:
            detail = "确认连接超时（3.5秒），请检查服务是否已启动" if isinstance(exc, asyncio.TimeoutError) else str(exc)
            message = f"自动确认连接失败：{detail}"
            if (generation == self._connection_generation and self._active and
                    url == self.config.api.base_url.rstrip("/")):
                self._connection = {"status": "error", "message": message, "checked_url": url}
            return self._connection_response(False, message)
        finally:
            if asyncio.current_task() is self._ensure_connection_task:
                self._busy = False

    @API("setup_ensure_connection", description="只读自动确认已保存的接口连接，不保存配置或加载模型")
    async def setup_ensure_connection(self) -> dict:
        try:
            self._require_setup()
            url = self.config.api.base_url.rstrip("/")
            if self._setup_connection_valid(url):
                return self._connection_response(True)
            task = self._ensure_connection_task
            if task and not task.done():
                if self._ensure_connection_url != url:
                    raise TTSError("接口地址已改变，正在结束旧连接确认，请稍后重试")
                return await asyncio.shield(task)
            if self._busy:
                raise TTSError("当前有任务正在运行，请结束后重试自动确认连接")
            progress = await self._progress_state()
            if not self._saved_connection_configured(progress):
                raise TTSError("尚未完成接口配置，请先在第一步保存并测试连接")
            self._require_setup()
            if url != self.config.api.base_url.rstrip("/"):
                raise TTSError("接口地址已改变，请重试自动确认连接")
            # compute yields to a thread: recheck the shared slot before creating it.
            task = self._ensure_connection_task
            if task and not task.done():
                return await asyncio.shield(task)
            if self._setup_connection_valid(url):
                return self._connection_response(True)
            if self._busy:
                raise TTSError("当前有任务正在运行，请结束后重试自动确认连接")
            self._connection = {"status": "running", "message": "正在自动确认连接…", "checked_url": url}
            self._busy = True
            self._ensure_connection_url = url
            task = asyncio.create_task(self._run_ensure_connection(url, self._connection_generation),
                                       name="gpt-sovits-ensure-connection")
            self._ensure_connection_task = task
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return await asyncio.shield(task)
        except (TTSError, ValueError, OSError) as exc:
            return self._connection_response(False, str(exc))

    @API("setup_catalog", description="扫描本机模型目录中的底模、权重，以及 maibot_refs 中的参考音频")
    async def setup_catalog(self, root: str = "") -> dict:
        # 只扫描本机模型目录（GPT-SoVITS 安装目录）；参考音频固定在其中的 maibot_refs。
        def scan() -> dict:
            result = scan_models(root)
            result.update(scan_reference_audios(root))
            return result
        return await asyncio.to_thread(scan)

    @API("setup_save_api", description="保存 GPT-SoVITS 接口设置")
    async def setup_save_api(self, base_url: str, timeout: float = 180) -> dict:
        try:
            self._require_setup()
            if not isinstance(base_url, str):
                raise TTSError("请输入有效接口地址")
            base_url = base_url.strip().rstrip("/")
            GPTSoVITSClient(base_url, timeout=timeout)
            await self._save_setup({"api": {"base_url": base_url, "timeout": timeout}})
            return {"success": True, "message": "接口设置已保存，请点击测试接口"}
        except (TTSError, ValueError, OSError) as exc:
            return {"success": False, "message": f"接口设置未保存：{exc}"}

    @API("setup_test_api", description="测试已保存的 HTTP 接口，不执行推理")
    async def setup_test_api(self) -> dict:
        try:
            self._require_setup()
            if self._busy:
                raise TTSError("当前有任务正在运行，请稍后测试接口")
            self._busy = True
            cfg = self.config.model_copy(deep=True)
            task = asyncio.current_task()
            if task:
                self._tasks.add(task)
            try:
                self._connection = {"status": "running", "message": "正在测试接口连接…"}
                await GPTSoVITSClient(cfg.api.base_url, timeout=5).probe()
                if cfg.api.base_url.rstrip("/") != self.config.api.base_url.rstrip("/"):
                    raise TTSError("测试期间接口地址已改变，请重新测试")
                self._mark_setup_connected(cfg.api.base_url, "接口连接成功，已确认 POST /tts；可进行第二步")
                await self._mark_progress(1)
            except asyncio.CancelledError:
                self._connection = {"status": "idle", "message": "接口测试已取消"}
                raise
            except Exception as exc:
                self._connection = {"status": "error", "message": f"接口测试失败：{exc}"}
            finally:
                self._busy = False
                if task:
                    self._tasks.discard(task)
            return {"success": self._connection["status"] == "success", "message": self._connection["message"]}
        except (TTSError, ValueError) as exc:
            return {"success": False, "message": str(exc)}

    @staticmethod
    def _path_key(value: str) -> str:
        value = value.strip().replace("\\", "/")
        return os.path.normcase(os.path.normpath(value)) if value else ""

    def _detect_base_model(self, catalog: dict, sovits_path: str) -> tuple[str, dict | None]:
        """根据所选 SoVITS 权重自动识别底模版本，与设置页显示规则一致。

        1. 路径正是某个已安装标准底模配对的 SoVITS 文件 → 该版本；
        2. 扫描结果中的完整权重 → 权重头版本（需本机已安装该版本的标准底模配对）；
        3. 不在扫描结果中的本机文件 → 只读文件头识别；已保存的远程路径沿用已保存版本；
        其余无法确认的情况按 custom（自定义权重）处理。LoRA 不能作为底模。
        """
        key = self._path_key(sovits_path)
        presets = [item for item in catalog.get("base_models", []) if isinstance(item, dict)]
        for item in presets:
            if self._path_key(str(item.get("sovits_path", ""))) == key:
                return str(item["id"]), item
        if any(isinstance(item, dict) and self._path_key(str(item.get("path", ""))) == key
               for item in catalog.get("lora_weights", [])):
            raise TTSError("所选文件是 LoRA，请在第三步普通模式中选择；第二步只选择底模或完整权重")
        version = next((item.get("version") for item in catalog.get("sovits_weights", [])
                        if isinstance(item, dict) and self._path_key(str(item.get("path", ""))) == key), None)
        if version is None:
            local = Path(sovits_path)
            if local.is_absolute() and local.is_file():
                try:
                    info = inspect_sovits_file(local)
                except (OSError, ValueError) as exc:
                    raise TTSError("无法识别所选 SoVITS 权重的文件头，请重新扫描并选择列表中的权重") from exc
                if info["mode"] == "lora":
                    raise TTSError("所选文件是 LoRA，请在第三步普通模式中选择；第二步只选择底模或完整权重")
                version = info["version"]
            elif key == self._path_key(self._selected_path(self.config.models.sovits_weights)):
                # 已保存但本机读不到的路径（例如远程服务路径）：沿用已保存的版本。
                version = self.config.models.base_model
            else:
                raise TTSError("所选 SoVITS 权重不在本机模型目录的扫描结果中，请重新扫描后选择")
        preset = next((item for item in presets if item.get("id") == version), None)
        return (str(version), preset) if preset is not None else ("custom", None)

    @API("setup_save_base", description="第二步保存模型目录与 SoVITS 底模，底模版本自动识别")
    async def setup_save_base(self, root: str, sovits_weights: str) -> dict:
        try:
            self._require_setup()
            if self._busy:
                raise TTSError("任务正在运行，请稍后选择底模")
            if not isinstance(root, str) or not isinstance(sovits_weights, str):
                raise TTSError("底模配置必须是文本")
            root, sovits = root.strip(), sovits_weights.strip()
            if not root:
                raise TTSError("请填写本机模型目录并扫描")
            if not sovits:
                raise TTSError("请选择 SoVITS 底模或完整权重")
            if not sovits.lower().endswith(".pth"):
                raise TTSError("SoVITS 底模或完整权重必须是 .pth 文件")
            catalog = await asyncio.to_thread(scan_models, root)
            base_model, preset = await asyncio.to_thread(self._detect_base_model, catalog, sovits)
            current = self.config.models
            changed = base_model != current.base_model
            updates = {"root": root, "base_model": base_model, "sovits_weights": sovits}
            if preset is not None:
                # 版本变化或尚未选择/仍是标准 GPT 时，同步为该版本的配套 GPT；保留同版本下手选的 GPT。
                preset_gpts = {self._path_key(str(item.get("gpt_path", ""))) for item in catalog.get("base_models", [])
                               if isinstance(item, dict)}
                gpt_key = self._path_key(self._selected_path(current.gpt_weights))
                if changed or not gpt_key or gpt_key in preset_gpts:
                    updates["gpt_weights"] = preset["gpt_path"]
            # LoRA 只能叠加在同版本的标准底模上；底模或版本变化时清除，避免残留不匹配的 LoRA。
            keeps_lora = (not changed and preset is not None
                          and self._path_key(str(preset.get("sovits_path", ""))) == self._path_key(sovits))
            if not keeps_lora:
                updates["lora_weights"] = NO_LORA
            await self._save_setup({"models": updates})
            self._role_status = {"status": "idle", "message": "底模已选择，请在第三步读取角色或配置普通LoRA", "profile": None}
            # 原版快照跟随第二步的公共底模；切回普通模式不恢复已经过时的底模版本。
            if self.config.role.mode == "original":
                await self._capture_original()
            else:
                snapshot = await asyncio.to_thread(self._read_snapshot)
                if snapshot:
                    snapshot["models"].update(self.config.models.model_dump())
                    await asyncio.to_thread(self._write_snapshot, self._original_snapshot_path(), snapshot)
            await self._mark_progress(2)
            if base_model == "custom":
                message = "已保存 SoVITS 权重；未识别为本机已安装的标准底模版本，按自定义权重处理（使用 LoRA 需要标准底模）。请继续第三步"
            else:
                message = f"已保存底模：{base_model}（根据所选 SoVITS 权重自动识别）；第二步不加载模型，请继续第三步选择角色模式或普通模式"
            return {"success": True, "base_model": base_model, "message": message}
        except (ValueError, OSError, TTSError) as exc:
            return {"success": False, "message": str(exc)}

    @API("setup_save_models", description="第三步保存普通模式单一LoRA及参考配置")
    async def setup_save_models(self, root: str = "", base_model: str = "custom",
                                gpt_weights: str = "", sovits_weights: str = "",
                                ref_audio_path: str = "", prompt_text: str = "",
                                prompt_lang: str = "zh", text_lang: str = "zh",
                                lora_weights: str | None = None) -> dict:
        try:
            self._require_setup()
            if lora_weights is None:
                lora_weights = self.config.models.lora_weights
            values = [root, base_model, gpt_weights, sovits_weights, lora_weights,
                      ref_audio_path, prompt_text, prompt_lang, text_lang]
            if any(not isinstance(value, str) for value in values):
                raise TTSError("模型和参考音频配置必须是文本")
            if (not self._selected_path(gpt_weights) or not self._selected_path(sovits_weights)) and base_model.strip() in ("", "custom"):
                raise TTSError("请选择或填写 GPT 权重及 SoVITS 权重")
            await self._save_setup({
                "models": {"root": root.strip(), "base_model": base_model.strip() or "custom",
                           "gpt_weights": self._selected_path(gpt_weights),
                           "sovits_weights": self._selected_path(sovits_weights),
                           "lora_weights": self._selected_path(lora_weights) or NO_LORA},
                "voice": {"ref_audio_path": ref_audio_path.strip(), "prompt_text": prompt_text.strip(),
                          "prompt_lang": prompt_lang.strip().lower(), "text_lang": text_lang.strip().lower()}})
            if self.config.role.mode == "original":
                await self._mark_progress(3)
            return {"success": True, "message": "GPT、LoRA 与参考音频已保存；保存本身不会切换服务端权重，可在第四步加载并试听"}
        except (TTSError, ValueError, OSError) as exc:
            return {"success": False, "message": f"模型设置未保存：{exc}"}

    @API("setup_save_parameters", description="第四步保存推理和行为参数，不修改角色模型与参考资源")
    async def setup_save_parameters(self, speed_factor: float, top_k: int, top_p: float,
                                    temperature: float, seed: int, text_split_method: str,
                                    parallel_infer: bool, max_text_length: int,
                                    cooldown_seconds: float, send_text: bool) -> dict:
        try:
            await self._save_setup({"voice": {"speed_factor": speed_factor, "top_k": top_k,
                "top_p": top_p, "temperature": temperature, "seed": seed,
                "text_split_method": text_split_method, "parallel_infer": parallel_infer},
                "behavior": {"max_text_length": max_text_length, "cooldown_seconds": cooldown_seconds,
                             "send_text": send_text}})
            await self._mark_progress(4)
            return {"success": True, "message": "推理与行为参数已保存；角色模型、LoRA和参考音频未改变"}
        except (ValueError, OSError, TTSError) as exc:
            return {"success": False, "message": f"参数未保存：{exc}"}

    @API("setup_save_auto_voice", description="第五步配置自动文字转语音频率和角色情感选择")
    async def setup_save_auto_voice(self, enabled: bool, frequency_mode: str,
                                     probability_percent: int, every_n: int,
                                     min_interval_seconds: float, follow_emotion: bool,
                                     keep_text: bool) -> dict:
        try:
            await self._save_setup({"auto_voice": {"enabled": enabled, "frequency_mode": frequency_mode,
                "probability_percent": probability_percent, "every_n": every_n,
                "min_interval_seconds": min_interval_seconds, "follow_emotion": follow_emotion,
                "keep_text": keep_text}})
            await self._mark_progress(6)
            return {"success": True, "message": "自动语音设置已保存；下一条合格机器人文字按频率处理，失败保留文字" if enabled else "自动语音已关闭，命令和语音工具仍可使用"}
        except (OSError, ValueError, TTSError) as exc:
            return {"success": False, "message": str(exc)}

    @API("setup_check_config", description="检查已保存底模、角色资源和参数配置，不推理或切换模型")
    async def setup_check_config(self) -> dict:
        cfg = self.config.model_copy(deep=True)
        issues = []
        def issue(level, field, message, step):
            # step 对应设置页的步骤，页面据此着色并支持点击跳转。
            issues.append({"level": level, "field": field, "message": message, "step": step})
        try:
            GPTSoVITSClient(cfg.api.base_url, timeout=cfg.api.timeout, max_audio_bytes=cfg.api.max_audio_bytes)
        except ValueError as exc:
            issue("error", "api.base_url", f"接口配置无效：{exc}", 1)
        if not cfg.plugin.enabled or not self._active:
            issue("warning", "plugin.enabled", "插件尚未运行，启用后才能执行接口测试和试听", 1)
        catalog = {"base_models": []}
        try:
            catalog = await asyncio.to_thread(scan_models, cfg.models.root)
        except (OSError, ValueError) as exc:
            issue("error", "models.root", f"模型目录不可用：{exc}", 2)
        preset = next((item for item in catalog["base_models"] if item["id"] == cfg.models.base_model), None)
        if cfg.models.base_model != "custom" and preset is None:
            issue("error", "models.base_model", "所选标准底模尚未在本地找到完整配对", 2)
        if cfg.role.mode == "role":
            roles = await self.setup_roles()
            role = next((item for item in roles["roles"] if item["id"] == cfg.role.role_id), None)
            if role is None:
                issue("error", "role.role_id", "所选角色不在 GPT-SoVITS 目录下的 maibot_roles 中："
                      + (roles["data_dir"] or "；".join(roles["errors"][:1]) or "请先在第二步保存本机模型目录"), 3)
            else:
                enabled = cfg.role.emotions or [cfg.role.emotion]
                available = {item["id"] for item in role["emotions"]}
                missing = [emotion for emotion in enabled if emotion not in available]
                if missing:
                    issue("error", "role.emotions", "JSON未定义所选情感：" + "、".join(missing), 3)
                if cfg.role.emotion not in enabled:
                    issue("error", "role.emotion", "默认试听情感不在启用列表", 3)
                if self._role_status["status"] != "success":
                    issue("warning", "role", "角色选择有效，但尚未读取 role.json 校验并套用；角色模式的模型和参考音频由 role.json 提供，无需手填", 3)
                else:
                    issue("info", "role", "角色配置已读取；使用 role.json 中各情感的模型和参考音频", 3)
                if len(enabled) > 1:
                    last = getattr(self, "_emotion_last", None) or {}
                    if not cfg.role.llm_emotion:
                        issue("info", "role.llm_emotion", "已启用多个情感，但关闭了 LLM 情感判断：合成时使用默认情感", 3)
                    elif last.get("source") == "llm_failed":
                        issue("warning", "role.llm_emotion", "最近一次 LLM 情感判断失败，已改用默认情感：" + str(last.get("detail", "")), 3)
                    else:
                        issue("info", "role.llm_emotion", f"多情感时由 MaiBot 的 {cfg.role.llm_task} 判断每句话的情感", 3)
        else:
            if not (self._selected_path(cfg.models.gpt_weights) or preset):
                issue("error", "models.gpt_weights", "普通模式缺少GPT权重", 3)
            if not (self._selected_path(cfg.models.sovits_weights) or preset):
                issue("error", "models.sovits_weights", "普通模式缺少SoVITS底模或完整权重", 2)
            # 参考音频在第三步（普通模式）设置。
            if not cfg.voice.ref_audio_path.strip():
                issue("error", "voice.ref_audio_path", "普通模式尚未选择参考音频", 3)
            if cfg.models.base_model in ("v3", "v4", "v5dev", "v5turbo") and not cfg.voice.prompt_text.strip():
                issue("error", "voice.prompt_text", "当前版本必须提供参考音频对应文本", 3)
        if self._connection["status"] != "success":
            issue("warning", "api", "尚未在本次插件运行中测试接口连接", 1)
        if self._loaded_signature != self._model_signature(cfg):
            if self._loads_on_demand(cfg):
                issue("info", "models", "当前选择尚未加载；首次合成时会自动加载，也可在第四步手动加载并试听", 4)
            else:
                issue("warning", "models", "当前选择尚未加载；普通模式未开启自动语音时，命令会沿用服务端当前权重，请在第四步加载模型", 4)
        ready = not any(item["level"] == "error" for item in issues)
        progress = await self._progress_state()
        return {"success": True, "ready": ready, "progress": progress, "next_step": progress["next_step"],
                "message": "保存配置检查通过；按提示完成角色读取和模型加载后可试听" if ready else "配置仍有缺项，请按提示修正",
                "issues": issues, "summary": {"mode": cfg.role.mode, "base_model": cfg.models.base_model,
                    "role_id": cfg.role.role_id, "emotion": cfg.role.emotion,
                    "emotions": cfg.role.emotions or [cfg.role.emotion], "api_url": cfg.api.base_url,
                    "auto_voice": cfg.auto_voice.model_dump()}}

    @API("setup_start_job", description="启动模型加载或语音试听后台测试")
    async def setup_start_job(self, kind: str, text: str = "") -> dict:
        try:
            self._require_setup()
            if not isinstance(text, str):
                raise TTSError("测试文本必须是字符串")
            if kind not in ("load_models", "synthesize"):
                raise TTSError("未知测试类型")
            ensuring = self._ensure_connection_task and not self._ensure_connection_task.done()
            if self._busy and not ensuring:
                raise TTSError("已有任务正在运行，请等待结束")
            if self.config.role.mode == "role" and (self._role_status["status"] != "success" or
                not self._role_status.get("profile") or
                self._role_status["profile"]["role_id"] != self.config.role.role_id or
                self._role_status["profile"]["emotion"] != self.config.role.emotion):
                raise TTSError("角色模式请先读取并校验所选角色JSON")
            if not self._setup_connection_valid(self.config.api.base_url.rstrip("/")):
                connection = await self.setup_ensure_connection()
                if not connection["success"]:
                    raise TTSError(connection["message"])
            if self._busy:
                raise TTSError("已有任务正在运行，请等待结束")
            cfg = self.config.model_copy(deep=True)
            if (not self._selected_path(cfg.models.gpt_weights) or not self._selected_path(cfg.models.sovits_weights)) and cfg.models.base_model == "custom":
                raise TTSError("请先选择并保存两份模型权重")
            if kind == "synthesize":
                if self._loaded_signature != self._model_signature(cfg):
                    raise TTSError("请先加载并测试当前选择的模型权重")
                if not isinstance(text, str) or not text.strip():
                    raise TTSError("请输入试听文本")
                if len(text.strip()) > min(cfg.behavior.max_text_length, 300):
                    raise TTSError("试听文本过长，请缩短到配置上限且不超过300字")
                if not cfg.voice.ref_audio_path.strip():
                    raise TTSError("请填写服务端可读的参考音频路径")
            self._busy = True
            self._preview_audio = b""
            self._job = {"id": uuid.uuid4().hex, "kind": kind, "status": "running",
                         "message": "正在启动模型加载…" if kind == "load_models" else "正在合成试听音频…",
                         "audio_bytes": 0, "duration_seconds": 0}
            self._job_task = asyncio.create_task(self._run_setup_job(cfg, kind, text.strip()),
                                                  name="gpt-sovits-setup")
            self._tasks.add(self._job_task)
            self._job_task.add_done_callback(self._finish_setup_task)
            return {"success": True, "message": self._job["message"], "job": dict(self._job)}
        except (TTSError, ValueError) as exc:
            return {"success": False, "message": str(exc)}

    def _finish_setup_task(self, task) -> None:
        self._tasks.discard(task)
        if task is self._job_task and task.cancelled():
            self._busy = False
            self._job.update(status="cancelled", message="测试已取消；请重新加载模型确认服务端状态。")
            if self._job["kind"] == "load_models":
                self._loaded_signature = None
                self._model_status = {"status": "error", "message": self._job["message"]}
            elif self._job["kind"] == "prepare_role":
                self._role_status = {"status": "error", "message": self._job["message"], "profile": None}

    async def _resolve_load_paths(self, cfg) -> tuple[str, str]:
        """只选择真实API可加载的路径；LoRA由服务端自动寻找标准底模。"""
        gpt = self._selected_path(cfg.models.gpt_weights)
        base = self._selected_path(cfg.models.sovits_weights)
        lora = self._selected_path(cfg.models.lora_weights)
        preset = None
        if cfg.models.base_model != "custom":
            catalog = await asyncio.to_thread(scan_models, cfg.models.root)
            preset = next((item for item in catalog["base_models"]
                           if item["id"] == cfg.models.base_model), None)
            if preset is not None:
                gpt = gpt or preset["gpt_path"]
                base = base or preset["sovits_path"]
        if not gpt or not base:
            raise TTSError("请先选择并保存GPT权重和SoVITS底模；未安装的底模不能加载")
        if not lora:
            return gpt, base
        # 不把角色GPT当LoRA，也不声称API能叠加到任意自选底模。
        if not lora.lower().endswith(".pth"):
            raise TTSError("LoRA必须是.pth角色权重，GPT的.ckpt不是LoRA")
        if cfg.models.base_model == "custom":
            raise TTSError("使用LoRA时请选择对应底模版本；官方API不支持指定任意自定义底模叠加LoRA")
        if preset is None:
            raise TTSError("LoRA需要已安装的标准底模，请刷新模型列表或检查服务端标准目录")
        if Path(base).as_posix().casefold() != Path(preset["sovits_path"]).as_posix().casefold():
            raise TTSError("LoRA加载会使用服务端该版本的标准底模，不支持替换为其他SoVITS权重")
        local = Path(lora)
        if local.is_file():
            info = await asyncio.to_thread(inspect_sovits_file, local)
            if info["mode"] != "lora":
                raise TTSError("所选文件未被识别为LoRA，请选择角色LoRA .pth，而不是完整底模或GPT权重")
            if info["version"] != cfg.models.base_model:
                raise TTSError(f"LoRA版本{info['version']}与所选底模{cfg.models.base_model}不匹配")
        return gpt, lora

    async def _run_setup_job(self, cfg, kind: str, text: str) -> None:
        job = self._job
        started = time.monotonic()
        # 设置页的加载与试听同样在 MaiBot 日志里记录开始与结果。
        trace = tts_log.Trace("load" if kind == "load_models" else "preview", text=text if kind != "load_models" else "")
        trace.set_role(cfg)
        if cfg.role.mode == "role" and kind != "load_models":
            trace.set_emotion(cfg.role.emotion, "default")
        tts_log.log_start(self._get_logger(), trace)
        try:
            client = GPTSoVITSClient(cfg.api.base_url, timeout=cfg.api.timeout,
                                     max_audio_bytes=cfg.api.max_audio_bytes)
            if kind == "load_models":
                self._loaded_signature = None
                self._model_status = {"status": "running", "message": "正在加载 GPT 权重…"}
                job["message"] = self._model_status["message"]
                gpt_path, acoustic_path = await self._resolve_load_paths(cfg)
                switch_started = time.monotonic()
                await client.set_gpt_weights(gpt_path)
                job["message"] = "GPT 权重加载成功，正在加载 SoVITS 底模或角色LoRA…"
                self._model_status = {"status": "running", "message": job["message"]}
                # 官方API直接接收LoRA路径，内部按版本加载默认底模并合并；不能任意叠加。
                await client.set_sovits_weights(acoustic_path)
                lora = self._selected_path(cfg.models.lora_weights)
                trace.record_switch(gpt_path, acoustic_path, bool(lora) and acoustic_path == lora, time.monotonic() - switch_started)
                if self._model_signature(cfg) != self._model_signature(self.config):
                    raise TTSError("加载期间配置已改变，请按新配置重新加载权重")
                self._loaded_signature = self._model_signature(cfg)
                self._model_status = {"status": "success", "message": "GPT 与 SoVITS 权重均已加载成功，可进行语音试听。服务端没有当前权重查询接口，外部切换不在本页跟踪范围。"}
                job.update(status="success", message=self._model_status["message"])
            else:
                payload = cfg.voice.model_dump(mode="python")
                payload.update(text=text, text_lang=cfg.voice.text_lang.lower(),
                               prompt_lang=cfg.voice.prompt_lang.lower(), media_type="wav",
                               streaming_mode=False, batch_size=1, gpt_streaming=False,
                               use_cuda_graph=False, use_flash_attention=False)
                trace.weights_note = "使用已加载的模型"
                synth_started = time.monotonic()
                self._preview_audio = await client.synthesize(payload)
                trace.synth_seconds = time.monotonic() - synth_started
                trace.audio = self._preview_audio
                trace.outcome = "仅本地试听，未发送"
                if self._model_signature(cfg) != self._model_signature(self.config):
                    raise TTSError("合成期间模型配置已改变，请重新加载并测试")
                with wave.open(io.BytesIO(self._preview_audio), "rb") as audio:
                    duration = round(audio.getnframes() / audio.getframerate(), 2)
                await self._mark_progress(5)
                job.update(status="success", message="试听合成成功，请点击播放器播放；不会发送聊天消息。",
                           audio_bytes=len(self._preview_audio), duration_seconds=duration)
            tts_log.log_success(self._get_logger(), trace)
        except asyncio.CancelledError:
            tts_log.log_cancelled(self._get_logger(), trace)
            job.update(status="cancelled", message="任务已取消；服务端可能仍在计算或已部分切换，请重新加载模型确认。")
            self._preview_audio = b""
            if kind == "load_models":
                self._loaded_signature = None
                self._model_status = {"status": "error", "message": job["message"]}
        except Exception as exc:
            trace.outcome = ""
            tts_log.log_failure(self._get_logger(), trace, tts_log.reason(exc))
            message = f"测试失败：{exc}"
            if kind == "load_models":
                message += "；两次切换不是原子操作，服务端可能已部分切换，请修正后重新加载两份权重。"
                self._loaded_signature = None
                self._model_status = {"status": "error", "message": message}
            self._preview_audio = b""
            job.update(status="error", message=message)
        finally:
            job["elapsed_seconds"] = round(time.monotonic() - started, 2)
            self._busy = False

    def _lookup_job(self, job_id: str) -> dict:
        self._require_setup()
        if not self._job or self._job["id"] != job_id:
            raise TTSError("任务不存在或已被新的测试替代，请刷新页面")
        return self._job

    @API("setup_job", description="读取后台测试进度")
    async def setup_job(self, job_id: str) -> dict:
        return dict(self._lookup_job(job_id))

    @API("setup_audio", description="分块读取试听 WAV，避免超过 WebUI 返回值上限")
    async def setup_audio(self, job_id: str, offset: int = 0) -> dict:
        job = self._lookup_job(job_id)
        if job["kind"] != "synthesize" or job["status"] != "success" or not self._preview_audio:
            raise TTSError("该任务尚无可播放的试听音频")
        if type(offset) is not int or offset < 0 or offset > len(self._preview_audio):
            raise TTSError("无效音频分块偏移")
        chunk = self._preview_audio[offset:offset + 131072]
        next_offset = offset + len(chunk)
        return {"base64": base64.b64encode(chunk).decode("ascii"), "next_offset": next_offset,
                "done": next_offset == len(self._preview_audio), "total_bytes": len(self._preview_audio)}

    @API("setup_cancel_job", description="取消本插件后台测试")
    async def setup_cancel_job(self, job_id: str) -> dict:
        job = self._lookup_job(job_id)
        if job["status"] == "running" and self._job_task:
            self._job_task.cancel()
            # Do not await server-side computation within the 10s WebUI budget.
            return {"success": True, "message": "已请求取消，请等待状态更新；服务端操作可能仍在继续。"}
        return {"success": True, "message": "任务已结束"}
