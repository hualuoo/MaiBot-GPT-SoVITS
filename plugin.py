"""GPT-SoVITS TTS for MaiBot SDK v2. No host internals are imported."""
from __future__ import annotations

import asyncio
import base64
import re
import time
from pathlib import Path

from maibot_sdk import Command, MaiBotPlugin, Tool

# Host loads the entry as a package; direct import remains available for tests.
if __package__:
    from .config_models import TTSConfig
    from .setup_service import SetupService
    from .native_config import NativeConfigMixin
    from .role_service import RoleService
    from .auto_voice import AutoVoiceMixin, _enabled_emotion
    from .emotion_judge import EmotionJudgeMixin
    from . import tts_log
    from .tts_client import GPTSoVITSClient, TTSError
else:
    from config_models import TTSConfig
    from setup_service import SetupService
    from native_config import NativeConfigMixin
    from role_service import RoleService
    from auto_voice import AutoVoiceMixin, _enabled_emotion
    from emotion_judge import EmotionJudgeMixin
    import tts_log
    from tts_client import GPTSoVITSClient, TTSError


TTS_PATTERN = r"(?s)^/tts(?:\s+(?P<text>.*))?$"


class GPTSoVITSPlugin(NativeConfigMixin, AutoVoiceMixin, EmotionJudgeMixin, RoleService, SetupService, MaiBotPlugin):
    config_model = TTSConfig

    def __init__(self):
        super().__init__()
        self._busy = False
        self._active = False
        self._last_sent: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()
        self._init_setup(Path(__file__).resolve().parent / "config.toml")
        self._init_native()
        self._init_roles()
        self._init_auto_voice()
        self._init_emotion_judge()

    async def on_load(self) -> None:
        self._active = True
        self._known_api_url = self.config.api.base_url.rstrip("/")
        self._known_model_signature = self._model_signature(self.config)
        # Do not contact the model server on plugin startup: it may be started later.

    async def on_unload(self) -> None:
        self._active = False
        await self._cancel_setup_connection()
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._last_sent.clear()
        self._preview_audio = b""
        self._loaded_signature = None

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        del version
        if scope == "self":
            old = self.config.model_copy(deep=True)
            self.set_plugin_config(config_data)
            self._invalidate_setup(old, self.config)
            self._last_sent.clear()

    def _check_access(self, stream_id: str, cfg: TTSConfig) -> None:
        if not self._active or not cfg.plugin.enabled:
            raise TTSError("TTS 插件未启用")
        if not stream_id:
            raise TTSError("缺少聊天流 ID")
        if cfg.behavior.allowed_stream_ids and stream_id not in cfg.behavior.allowed_stream_ids:
            raise TTSError("此聊天流未获准使用 TTS")

    def _client(self, cfg: TTSConfig) -> GPTSoVITSClient:
        try:
            return GPTSoVITSClient(cfg.api.base_url, timeout=cfg.api.timeout,
                                  max_audio_bytes=cfg.api.max_audio_bytes)
        except ValueError as exc:
            # 配置错误应给出可操作提示，而不是记录异常栈后显示“执行异常”。
            raise TTSError("GPT-SoVITS 接口地址或超时配置无效，请在设置页第一步修正") from exc

    async def _speak(self, stream_id: str, text: str, *, from_tool: bool = False, emotion: str = "",
                     chat: str = "") -> str:
        """命令与语音工具的合成入口；每次调用在 MaiBot 日志输出开始与结果。"""
        stream_id = stream_id if isinstance(stream_id, str) else ""
        trace = tts_log.Trace("tool" if from_tool else "command", stream_id=stream_id,
                              chat=chat or tts_log.known_chat(self, stream_id),
                              text=text if isinstance(text, str) else "")
        token = tts_log.activate(trace)
        try:
            message = await self._speak_once(stream_id, text, from_tool=from_tool, emotion=emotion, trace=trace)
        except asyncio.CancelledError:
            tts_log.log_cancelled(self._get_logger(), trace)
            raise
        except Exception as exc:
            tts_log.log_failure(self._get_logger(), trace, tts_log.reason(exc))
            raise
        else:
            tts_log.log_success(self._get_logger(), trace)
            return message
        finally:
            tts_log.deactivate(token)

    async def _speak_once(self, stream_id: str, text: str, *, from_tool: bool, emotion: str, trace) -> str:
        # Snapshot configuration; a hot reload cannot change an in-flight request.
        cfg = self.config.model_copy(deep=True)
        self._check_access(stream_id, cfg)
        if not isinstance(emotion, str):
            raise TTSError("情感参数必须是文本")
        judge = False
        if cfg.role.mode == "role":
            enabled = cfg.role.emotions or [cfg.role.emotion]
            requested = emotion.strip()
            # 模型工具常传入大小写/中文别名（Happy、开心）；只映射到已启用情感，绝不越权。
            selected = _enabled_emotion(requested, enabled) if requested else cfg.role.emotion
            if not selected or selected not in enabled:
                raise TTSError("该情感尚未启用，请在角色模式勾选并读取配置")
            cfg.role.emotion = selected
            emotion = selected
            # 未显式指定情感且启用了多个情感：持有忙碌标志后再请 LLM 判断，避免并发推理。
            judge = not requested and self._llm_emotion_active(cfg)
        elif emotion.strip():
            raise TTSError("普通模式只使用单一LoRA，不支持多情感选择")
        if from_tool and not cfg.behavior.tool_enabled:
            raise TTSError("模型语音工具已关闭，请使用 /tts 命令")
        if not isinstance(text, str) or not text.strip():
            raise TTSError("请输入要朗读的文本，例如：/tts 你好，很高兴认识你。")
        text = text.strip()
        if len(text) > cfg.behavior.max_text_length:
            raise TTSError(f"文本超过 {cfg.behavior.max_text_length} 字，请缩短后重试")
        if cfg.role.mode == "original" and not cfg.voice.ref_audio_path.strip():
            raise TTSError("尚未配置 voice.ref_audio_path；请设置 GPT-SoVITS 服务端参考音频路径")
        remaining = cfg.behavior.cooldown_seconds - (time.monotonic() - self._last_sent.get(stream_id, -1e12))
        if remaining > 0:
            raise TTSError(f"语音冷却中，请 {remaining:.0f} 秒后重试")
        # Fail fast instead of accumulating unbounded GPU/CPU inference queues.
        # There is no await between testing and setting this process-local flag.
        if self._busy:
            raise TTSError("正在合成其他语音或执行设置任务，请稍后重试")
        self._busy = True
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        try:
            trace.set_role(cfg)
            if cfg.role.mode == "role" and not judge:
                trace.set_emotion(emotion, "requested" if requested else "default")
            tts_log.log_start(self._get_logger(), trace)
            if judge:
                emotion, _ = await self._judge_emotion(text, cfg, cfg.role.emotion)
                cfg.role.emotion = emotion
            started = time.monotonic()
            audio = await self._synthesize_audio(cfg, text, emotion=emotion)
            trace.audio = audio if isinstance(audio, bytes) else b""
            if trace.synth_seconds is None:
                trace.synth_seconds = time.monotonic() - started
            encoded = base64.b64encode(audio).decode("ascii")
            result = await self.ctx.send.custom("voice", encoded, stream_id,
                                                processed_plain_text=text)
            sent = result.get("sent", False) if isinstance(result, dict) else bool(result)
            if not sent:
                raise TTSError("音频已合成，但语音发送失败；请检查适配器是否支持 voice/WAV")
            now = time.monotonic()
            # Expire cooldown entries to bound long-running process memory.
            self._last_sent = {k: v for k, v in self._last_sent.items()
                               if now - v < cfg.behavior.cooldown_seconds}
            self._last_sent[stream_id] = now
            trace.outcome = "已发送"
            if cfg.behavior.send_text:
                try:
                    self._auto_voice_prune(time.monotonic())
                    self._auto_voice_suppress[(stream_id, text)] = time.monotonic() + 60
                    self._auto_voice_prune(time.monotonic())
                    await self.ctx.send.text(text, stream_id)
                    trace.outcome = "已发送（附带原文）"
                except Exception:
                    trace.outcome = "已发送（附带原文失败）"
                    self._get_logger().warning("语音已发送，但附加原文发送失败")
            return "语音已发送，请勿重复朗读或再次调用工具。"
        finally:
            self._busy = False
            if task:
                self._tasks.discard(task)

    async def _synthesize_audio(self, cfg: TTSConfig, text: str, emotion: str = "") -> bytes:
        """仅合成；供命令、工具和自动发送前Hook共用。调用者持有_busy。"""
        if cfg.role.mode == "role" and emotion:
            if emotion not in (cfg.role.emotions or [cfg.role.emotion]):
                raise TTSError("该情感尚未启用")
            cfg.role.emotion = emotion
        client = self._client(cfg)
        trace = tts_log.current()
        if cfg.role.mode == "role":
            profile = await asyncio.to_thread(lambda: self._roles().resolve(cfg.role.role_id, cfg.role.emotion,
                                                                            cfg.models.root))
            if cfg.models.base_model != "custom" and cfg.models.base_model != profile["models"]["base_model"]:
                raise TTSError("角色 role.json 与第二步所选底模版本不匹配，请重新选择底模并读取角色")
            for key, value in profile["models"].items():
                setattr(cfg.models, key, value)
            cfg.voice.aux_ref_audio_paths = []
            for key, value in profile["voice"].items():
                setattr(cfg.voice, key, value)
            await self._ensure_weights(client, cfg, trace)
            # 工具可按单次请求切情感；页面默认情感与全部就绪配置不被这次请求覆盖。
            current_status = dict(self._role_status)
            current_status.update(status="success", message="角色情感已校验并应用")
            if not current_status.get("profile"):
                current_status["profile"] = profile
            current_status.setdefault("profiles", {})[cfg.role.emotion] = profile
            current_status["emotions"] = list(cfg.role.emotions or [cfg.role.emotion])
            self._role_status = current_status
        elif cfg.auto_voice.enabled and self._original_weights_configured(cfg):
            # 自定义模式未填写权重时沿用服务端当前权重（与关闭自动语音时一致），
            # 否则每次自动语音/命令都会因“请先选择权重”失败。
            await self._ensure_weights(client, cfg, trace)
        elif trace is not None:
            trace.weights_note = "使用服务端当前加载的模型"
        payload = cfg.voice.model_dump(mode="python")
        payload.update(text=text, text_lang=cfg.voice.text_lang.lower(),
                       prompt_lang=cfg.voice.prompt_lang.lower(), media_type="wav",
                       streaming_mode=False, batch_size=1, gpt_streaming=False,
                       use_cuda_graph=False, use_flash_attention=False)
        started = time.monotonic()
        audio = await client.synthesize(payload)
        if trace is not None:
            trace.synth_seconds = time.monotonic() - started
        self._mark_setup_connected(cfg.api.base_url, "语音接口已实际合成成功，无需重复手动测试")
        return audio

    async def _ensure_weights(self, client, cfg: TTSConfig, trace) -> None:
        """当前选择与已加载的不同才切换 GPT 与 SoVITS/LoRA 权重。"""
        signature = self._model_signature(cfg)
        if self._loaded_signature == signature:
            if trace is not None:
                trace.weights_note = "沿用已加载的模型"
            return
        gpt_path, acoustic_path = await self._resolve_load_paths(cfg)
        started = time.monotonic()
        self._loaded_signature = None
        await client.set_gpt_weights(gpt_path)
        await client.set_sovits_weights(acoustic_path)
        self._loaded_signature = signature
        if trace is not None:
            lora = self._selected_path(cfg.models.lora_weights)
            trace.record_switch(gpt_path, acoustic_path, bool(lora) and acoustic_path == lora, time.monotonic() - started)

    @Command("gpt_sovits_tts", description="朗读文本：/tts 文本", pattern=TTS_PATTERN, timeout_ms=300000)
    async def handle_tts(self, stream_id: str = "", **kwargs):
        chat = tts_log.chat_label(kwargs.get("message"))
        tts_log.remember_chat(self, stream_id, chat)
        groups = kwargs.get("matched_groups") or {}
        text = groups.get("text", "") if isinstance(groups, dict) else ""
        if not text:
            match = re.match(TTS_PATTERN, str(kwargs.get("text") or ""))
            if match:
                text = match.group("text") or ""
        try:
            message = await self._speak(stream_id, text, chat=chat)
            return True, message, True
        except TTSError as exc:
            message = str(exc)
        except Exception:
            self._get_logger().exception("TTS 命令执行异常")
            message = "TTS 执行异常，请查看插件日志。"
        if stream_id:
            await self.ctx.send.text(f"TTS：{message}", stream_id)
        return False, message, True

    @Command("gpt_sovits_tts_status", description="检查 TTS 接口（不合成）：/tts_status", pattern=r"^/tts_status$", timeout_ms=15000)
    async def handle_status(self, stream_id: str = "", **kwargs):
        del kwargs
        try:
            cfg = self.config.model_copy(deep=True)
            self._check_access(stream_id, cfg)
            try:
                client = GPTSoVITSClient(cfg.api.base_url, timeout=5)
            except ValueError as exc:
                raise TTSError("GPT-SoVITS 接口地址配置无效，请在设置页第一步修正") from exc
            status = await client.probe()
            if cfg.role.mode == "role":
                reference = "由角色JSON提供" if cfg.role.role_id.strip() else "未选择角色"
            else:
                reference = "已配置" if cfg.voice.ref_audio_path.strip() else "未配置"
            message = f"{status}；参考音频{reference}；合成状态：{'忙碌' if self._busy else '空闲'}。接口可达不代表模型或参考音频可用。"
            success = True
        except TTSError as exc:
            message, success = str(exc), False
        except Exception:
            self._get_logger().exception("TTS 接口探测异常")
            message, success = "接口探测异常，请查看插件日志", False
        if stream_id:
            await self.ctx.send.text(f"TTS：{message}", stream_id)
        return success, message, True

    @Tool("gpt_sovits_speak", description="将文本合成为语音并发送到当前聊天",
          detailed_description="当用户要求用语音说话、朗读或发送语音时调用。text 为要实际朗读的完整自然文本，不含 Markdown 或动作描述。成功后已向聊天发送音频，不要再重复发送相同内容。",
          parameters={"text": {"type": "string", "description": "实际朗读的完整文本", "required": True},
                      "emotion": {"type": "string", "description": "可选，仅角色模式：使用已启用的情感ID，如happy/neutral/sad；省略时多情感角色由 MaiBot LLM 自动判断，单情感使用默认情感", "required": False}},
          timeout_ms=300000)
    async def handle_speak(self, text: str = "", stream_id: str = "", emotion: str = "", **kwargs):
        del kwargs
        try:
            message = await self._speak(stream_id, text, from_tool=True, emotion=emotion)
            return {"name": "gpt_sovits_speak", "content": message, "success": True}
        except TTSError as exc:
            return {"name": "gpt_sovits_speak", "content": str(exc), "success": False}
        except Exception:
            self._get_logger().exception("TTS 工具执行异常")
            return {"name": "gpt_sovits_speak", "content": "TTS 异常，请查看插件日志；没有确认发送成功。", "success": False}


def create_plugin() -> GPTSoVITSPlugin:
    return GPTSoVITSPlugin()
